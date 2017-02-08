/*
 * Copyright © 2016 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <stddef.h>
#include <string.h>

#include "ksim.h"
#include "avx-builder.h"

static void *shader_pool, *shader_end;
const size_t shader_pool_size = 64 * 1024;
static void *constant_pool;
const size_t constant_pool_size = 4096;
uint32_t constant_pool_index;

void
reset_shader_pool(void)
{
	if (shader_pool == NULL) {
		int fd = memfd_create("jit", MFD_CLOEXEC);
		ftruncate(fd, shader_pool_size);
		shader_pool = mmap(NULL, shader_pool_size,
				   PROT_WRITE | PROT_READ | PROT_EXEC,
				   MAP_SHARED, fd, 0);
		close(fd);
	}

	constant_pool = shader_pool;
	constant_pool_index = 0;
	shader_end = shader_pool + constant_pool_size;
}

void *
get_const_data(size_t size, size_t align)
{
	int offset = align_u64(constant_pool_index, align);

	constant_pool_index = offset + size;
	ksim_assert(offset + size <= constant_pool_size);

	return constant_pool + offset;
}

static int
builder_disasm_printf(void *_bld, const char *fmt, ...)
{
	struct builder *bld = _bld;
	va_list va;
	int length;

	va_start(va, fmt);
	length = vsnprintf(bld->disasm_output + bld->disasm_length,
			   sizeof(bld->disasm_output) - bld->disasm_length, fmt, va);
	va_end(va);
	bld->disasm_length += length;

	return length;
}

void
builder_init(struct builder *bld)
{
	bld->shader = align_ptr(shader_end, 64);
	bld->p = (uint8_t *) bld->shader;

	bld->disasm_tail = bld->p - (uint8_t *) shader_pool;
	init_disassemble_info(&bld->info, bld, builder_disasm_printf);
	bld->info.arch = bfd_arch_i386;
	bld->info.mach = bfd_mach_x86_64;
	bld->info.buffer_vma = 0;
	bld->info.buffer_length = shader_pool_size;
	bld->info.buffer = shader_pool;
	bld->info.section = NULL;
	disassemble_init_for_target(&bld->info);
}

shader_t
builder_finish(struct builder *bld)
{
	shader_end = bld->p;

	ksim_assert(shader_end - shader_pool < shader_pool_size);

	return bld->shader;
}

bool
builder_disasm(struct builder *bld)
{
	const int end = bld->p - (uint8_t *) shader_pool;

	bld->disasm_length = 0;
	if (bld->disasm_tail < end) {
		bld->disasm_last = bld->disasm_tail;
		bld->disasm_length += snprintf(bld->disasm_output + bld->disasm_length,
					       sizeof(bld->disasm_output) - bld->disasm_length,
					       "%08x  ", bld->disasm_tail);

		bld->disasm_tail +=
			print_insn_i386(bld->disasm_tail, &bld->info);
		return true;
	} else {
		return false;
	}
}

void
builder_align(struct builder *bld)
{
	int len = (-(intptr_t) bld->p) & 15;
	static const uint8_t nops[][10] = {
		{ },
		{ 0x90 },
		{ 0x66, 0x90},
		{ 0x0f, 0x1f, 0x00 },
		{ 0x0f, 0x1f, 0x40, 0x00 },
		{ 0x0f, 0x1f, 0x44, 0x00, 0x00, },
		{ 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00},
		{ 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,},
		{ 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,},
		{ 0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00 },
	};

	if (len >= 9) {
		memcpy(bld->p, nops[9], 9);
		bld->p += 9;
		len -= 9;
	}

	memcpy(bld->p, nops[len], len);
	bld->p += len;
}

#ifdef TEST_AVX_BUILDER

uint32_t trace_mask = 0;
uint32_t breakpoint_mask = 0;
FILE *trace_file;

static void
test_fail(struct builder *bld, const char *fmt, ...)
{
	const uint8_t *code = (uint8_t *) bld->shader;
	const int size = bld->p - code;
	va_list va;

	va_start(va, fmt);
	vprintf(fmt, va);
	va_end(va);

	for (int i = 0; i < size; i++)
		printf("%02x ", code[i]);
	printf("%s\n", bld->disasm_output);
	exit(EXIT_FAILURE);
}

static void
check_reg_imm_emit_function(const char *fmt,
			    void (*func)(struct builder *bld, int reg, int imm), int delta)
{
	const int imm = 100;
	struct builder bld;

	for (int reg = 0; reg < 16; reg++) {
		int count, actual_reg, actual_imm;

		reset_shader_pool();
		builder_init(&bld);

		func(&bld, reg, imm);
		builder_disasm(&bld);

		count = sscanf(bld.disasm_output, fmt, &actual_reg, &actual_imm);

		if (count != 2 || reg != actual_reg || imm - delta != actual_imm)
			test_fail(&bld, "fmt='%s' reg=%d imm=%d:\n    ", fmt, reg, imm);
	}
}

static void
check_unop_emit_function(const char *fmt,
			 void (*func)(struct builder *bld, int dst))
{
	struct builder bld;

	for (int dst = 0; dst < 16; dst++) {
		int count, actual_dst;

		reset_shader_pool();
		builder_init(&bld);

		func(&bld, dst);
		builder_disasm(&bld);

		count = sscanf(bld.disasm_output, fmt, &actual_dst);

		if (count != 1 || dst != actual_dst)
			test_fail(&bld, "fmt='%s' dst=%d:\n    ", fmt, dst);
	}
}



static void
check_binop_emit_function(const char *fmt,
			  void (*func)(struct builder *bld, int dst, int src))
{
	struct builder bld;

	for (int dst = 0; dst < 16; dst++) {
		for (int src = 0; src < 16; src++) {
			int count, actual_dst, actual_src;

			reset_shader_pool();
			builder_init(&bld);

			func(&bld, dst, src);
			builder_disasm(&bld);

			count = sscanf(bld.disasm_output, fmt,
				       &actual_src, &actual_dst);

			if (count != 2 ||
			    dst != actual_dst || src != actual_src)
				test_fail(&bld, "fmt='%s' dst=%d src=%d:\n    ", fmt, dst, src);
		}
	}
}

static void
check_triop_emit_function(const char *fmt,
			  void (*func)(struct builder *bld, int dst, int src0, int src1))
{
	struct builder bld;

	for (int dst = 0; dst < 16; dst++)
		for (int src0 = 0; src0 < 16; src0++)
			for (int src1 = 0; src1 < 16; src1++) {
				int count, actual_dst, actual_src0, actual_src1;

				reset_shader_pool();
				builder_init(&bld);

				func(&bld, dst, src0, src1);
				builder_disasm(&bld);

				count = sscanf(bld.disasm_output,
					       fmt, &actual_src0, &actual_src1, &actual_dst);

				if (count != 3 ||
				    dst != actual_dst || src0 != actual_src0 || src1 != actual_src1)
					test_fail(&bld, "fmt='%s' dst=%d src0=%d src1=%d:\n    ",
						  fmt, dst, src0, src1);
			}
}

static void
check_quadop_emit_function(const char *fmt,
			   void (*func)(struct builder *bld, int dst, int mask, int src0, int src1))
{
	struct builder bld;

	for (int dst = 0; dst < 16; dst++)
		for (int mask = 0; mask < 16; mask++)
			for (int src0 = 0; src0 < 16; src0++)
				for (int src1 = 0; src1 < 16; src1++) {
					int count, actual_dst, actual_mask;
					int actual_src0, actual_src1;

					reset_shader_pool();
					builder_init(&bld);

					func(&bld, dst, mask, src0, src1);
					builder_disasm(&bld);

					count = sscanf(bld.disasm_output,
						       fmt, &actual_mask,
						       &actual_src0, &actual_src1, &actual_dst);

					if (count != 4 ||
					    dst != actual_dst || mask != actual_mask ||
					    src0 != actual_src0 || src1 != actual_src1)
						test_fail(&bld, "fmt='%s' dst=%d src0=%d src1=%d:\n    ",
							  fmt, dst, src0, src1);
				}
}

static inline void
emit_vpgatherdd_scale1(struct builder *bld, int dst, int index, int mask)
{
	builder_emit_vpgatherdd(bld, dst, index, mask, 1, 0);
}

static inline void
emit_vpgatherdd_scale2(struct builder *bld, int dst, int index, int mask)
{
	builder_emit_vpgatherdd(bld, dst, index, mask, 2, 0);
}

static inline void
emit_vpgatherdd_scale4(struct builder *bld, int dst, int index, int mask)
{
	builder_emit_vpgatherdd(bld, dst, index, mask, 4, 0);
}

static inline void
emit_vpgatherdd_scale1_offset24(struct builder *bld, int dst, int index, int mask)
{
	builder_emit_vpgatherdd(bld, dst, index, mask, 1, 24);
}

static inline void
emit_vpmaskmovd(struct builder *bld, int mask, int src)
{
	builder_emit_vpmaskmovd(bld, mask, src, 0x300);
}

static inline void
emit_vmovdqa_from_rax(struct builder *bld, int mask, int src)
{
}

static void
emit_vpmaskmovd_to_rax(struct builder *bld, int src, int mask)
{
	builder_emit_vpmaskmovd_to_rax(bld, src, mask, 0);
}

static void
emit_vpmaskmovd_to_rax_20(struct builder *bld, int src, int mask)
{
	builder_emit_vpmaskmovd_to_rax(bld, src, mask, 20);
}

static void
emit_vpmaskmovd_to_rax_500(struct builder *bld, int src, int mask)
{
	builder_emit_vpmaskmovd_to_rax(bld, src, mask, 500);
}

int main(int argc, char *argv[])
{
	check_reg_imm_emit_function("vpbroadcastd 0x%2$x(%%rip),%%ymm%1$d",
				    builder_emit_vpbroadcastd_rip_relative, 9);

	check_reg_imm_emit_function("vmovdqa 0x%2$x(%%rdi),%%ymm%1$d",
				    builder_emit_m256i_load, 0);
	check_reg_imm_emit_function("vmovdqa %%ymm%1$d,0x%2$x(%%rdi)",
				    builder_emit_m256i_store, 0);
	check_reg_imm_emit_function("vmovdqa 0x%2$x(%%rdi),%%xmm%1$d",
				    builder_emit_m128i_load, 0);
	check_reg_imm_emit_function("vmovdqa %%xmm%1$d,0x%2$x(%%rdi)",
				    builder_emit_m128i_store, 0);
	check_reg_imm_emit_function("vpbroadcastd 0x%2$x(%%rdi),%%ymm%1$d",
				    builder_emit_vpbroadcastd, 0);

	check_reg_imm_emit_function("vmovdqa 0x%2$x(%%rip),%%ymm%1$d",
				    builder_emit_m256i_load_rip_relative, 8);

	check_reg_imm_emit_function("vmovd %%xmm%1$d, 0x%2$x(%%rdi)",
				    builder_emit_u32_store, 0);

	check_triop_emit_function("vpaddd %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vpaddd);
	check_triop_emit_function("vpsubd %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vpsubd);
	check_triop_emit_function("vpmulld %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vpmulld);
	check_triop_emit_function("vaddps %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vaddps);
	check_triop_emit_function("vmulps %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vmulps);
	check_triop_emit_function("vdivps %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vdivps);
	check_triop_emit_function("vsubps %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vsubps);
	check_triop_emit_function("vpand %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vpand);
	check_triop_emit_function("vpandn %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vpandn);
	check_triop_emit_function("vpxor %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vpxor);
	check_triop_emit_function("vpor %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vpor);
	check_triop_emit_function("vpsrlvd %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vpsrlvd);
	check_triop_emit_function("vpsravd %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vpsravd);
	check_triop_emit_function("vpsllvd %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vpsllvd);
	check_triop_emit_function("vfmadd132ps %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vfmadd132ps);
	check_triop_emit_function("vfmadd231ps %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vfmadd231ps);
	check_triop_emit_function("vfnmadd132ps %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vfnmadd132ps);
	check_triop_emit_function("vfnmadd231ps %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vfnmadd231ps);
	check_triop_emit_function("vpgatherdd %%ymm%2$d,(%%rax,%%ymm%1$d,1),%%ymm%3$d", emit_vpgatherdd_scale1);
	check_triop_emit_function("vpgatherdd %%ymm%2$d,(%%rax,%%ymm%1$d,2),%%ymm%3$d", emit_vpgatherdd_scale2);
	check_triop_emit_function("vpgatherdd %%ymm%2$d,(%%rax,%%ymm%1$d,4),%%ymm%3$d", emit_vpgatherdd_scale4);
	check_triop_emit_function("vpgatherdd %%ymm%2$d,0x18(%%rax,%%ymm%1$d,1),%%ymm%3$d", emit_vpgatherdd_scale1_offset24);

	check_triop_emit_function("vpsrld $0x%2$x,%%ymm%1$d,%%ymm%3$d", builder_emit_vpsrld);
	check_triop_emit_function("vpslld $0x%2$x,%%ymm%1$d,%%ymm%3$d", builder_emit_vpslld);

	check_binop_emit_function("vpabsd %%ymm%d,%%ymm%d", builder_emit_vpabsd); 
	check_binop_emit_function("vrsqrtps %%ymm%d,%%ymm%d", builder_emit_vrsqrtps);
	check_binop_emit_function("vsqrtps %%ymm%d,%%ymm%d", builder_emit_vsqrtps);
	check_binop_emit_function("vrcpps %%ymm%d,%%ymm%d", builder_emit_vrcpps);
	check_binop_emit_function("vpmaskmovd %%ymm%2$d,%%ymm%1$d,0x300(%%rdi)", emit_vpmaskmovd);
	check_binop_emit_function("vpmovsxwd %%xmm%d,%%ymm%d", builder_emit_vpmovsxwd);
	check_binop_emit_function("vpmovzxwd %%xmm%d,%%ymm%d", builder_emit_vpmovzxwd);
	check_binop_emit_function("vmovdqa %%ymm%d,%%ymm%d", builder_emit_vmovdqa);
	check_binop_emit_function("vpmaskmovd %%ymm%2$d,%%ymm%1$d,(%%rax)", emit_vpmaskmovd_to_rax);
	check_binop_emit_function("vpmaskmovd %%ymm%2$d,%%ymm%1$d,20(%%rax)", emit_vpmaskmovd_to_rax_20);
	check_binop_emit_function("vpmaskmovd %%ymm%2$d,%%ymm%1$d,500(%%rax)", emit_vpmaskmovd_to_rax_500);

	/* check_triop_emit_function("vcmpps", builder_emit_vcmpps); */

	check_triop_emit_function("vmaxps %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vmaxps);
	check_triop_emit_function("vminps %%ymm%d,%%ymm%d,%%ymm%d", builder_emit_vminps);

	check_quadop_emit_function("vpblendvb %%ymm%d,%%ymm%d,%%ymm%d,%%ymm%d",
				   builder_emit_vpblendvb);

	check_unop_emit_function("vmovdqa (%%rax),%%ymm%d", builder_emit_vmovdqa_from_rax);
	check_unop_emit_function("vmovdqa %%ymm%d,(%%rax)", builder_emit_vmovdqa_to_rax);

#if 0
	/* xmm regs */
	check_triop_emit_function("vpackssdw", builder_emit_vpackssdw);
#endif

	/* builder_emit_vextractf128 */

	return EXIT_SUCCESS;
}

#endif
