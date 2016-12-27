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
builder_init(struct builder *bld, uint64_t surfaces, uint64_t samplers)
{
	bld->shader = align_ptr(shader_end, 64);
	bld->p = (uint8_t *) bld->shader;
	bld->binding_table_address = surfaces;
	bld->sampler_state_address = samplers;

	bld->disasm_tail = bld->p - (uint8_t *) shader_pool;
	init_disassemble_info(&bld->info, bld, builder_disasm_printf);
	/* info.print_address_func = override_print_address; */
	bld->info.arch = bfd_arch_i386;
	bld->info.mach = bfd_mach_x86_64;
	bld->info.buffer_vma = 0;
	bld->info.buffer_length = shader_pool_size;
	bld->info.buffer = shader_pool;
	bld->info.section = NULL;
	disassemble_init_for_target(&bld->info);

	builder_invalidate_all(bld);
}

shader_t
builder_finish(struct builder *bld)
{
	shader_end = bld->p;

	ksim_assert(shader_end - shader_pool < shader_pool_size);

	return bld->shader;
}

void *
builder_get_const_data(struct builder *bld, size_t size, size_t align)
{
	int offset = align_u64(constant_pool_index, align);

	constant_pool_index = offset + size;
	ksim_assert(offset + size <= constant_pool_size);

	return constant_pool + offset;
}

void
builder_invalidate_all(struct builder *bld)
{
	list_init(&bld->regs_lru_list);
	list_init(&bld->used_regs_list);

	for (int i = 0; i < ARRAY_LENGTH(bld->regs); i++) {
		list_insert(&bld->regs_lru_list, &bld->regs[i].link);
		bld->regs[i].contents = 0;
	}
}

int
builder_use_reg(struct builder *bld, struct avx2_reg *reg)
{
	list_remove(&reg->link);
	list_insert(&bld->used_regs_list, &reg->link);

	return reg - bld->regs;
}

int
builder_get_reg_with_uniform(struct builder *bld, uint32_t ud)
{
	struct avx2_reg *reg;
	void *p;
	int reg_num;

	if (list_find(reg, &bld->regs_lru_list, link,
		      (reg->contents & BUILDER_REG_CONTENTS_UNIFORM) &&
		      reg->uniform == ud))
		return builder_use_reg(bld, reg);

	reg_num = builder_get_reg(bld);
	reg = &bld->regs[reg_num];
	reg->contents |= BUILDER_REG_CONTENTS_UNIFORM;
	reg->uniform = ud;

	if (ud == 0) {
		builder_emit_vpxor(bld, reg_num, reg_num, reg_num);
	} else {
		p = builder_get_const_ud(bld, ud);
		builder_emit_vpbroadcastd_rip_relative(bld, reg_num, builder_offset(bld, p));
	}

	return reg_num;
}

int
builder_get_reg(struct builder *bld)
{
	struct avx2_reg *reg = container_of(bld->regs_lru_list.prev, reg, link);

	ksim_assert(!list_empty(&bld->regs_lru_list));

	reg->contents = 0;

	return builder_use_reg(bld, reg);
}

void
builder_release_regs(struct builder *bld)
{
	list_insert_list(&bld->regs_lru_list, &bld->used_regs_list);
	list_init(&bld->used_regs_list);
}

void
builder_dump_register_cache(struct builder *bld, FILE *fp)
{
	for (int i = 0; i < ARRAY_LENGTH(bld->regs); i++) {
		if (bld->regs[i].contents & BUILDER_REG_CONTENTS_EU_REG)
			fprintf(fp, "  ymm%d: g%d.%d<%d,%d,%d>\n",
				i,
				bld->regs[i].region.offset / 32,
				bld->regs[i].region.offset & 31,
				bld->regs[i].region.vstride,
				bld->regs[i].region.width,
				bld->regs[i].region.hstride);
	}
}

bool
builder_disasm(struct builder *bld)
{
	const int end = bld->p - (uint8_t *) shader_pool;

	bld->disasm_length = 0;
	if (bld->disasm_tail < end) {
		bld->disasm_tail +=
			print_insn_i386(bld->disasm_tail, &bld->info);
		return true;
	} else {
		return false;
	}
}

#ifdef TEST_AVX_BUILDER

void
check_reg_imm_emit_function(const char *fmt,
			    void (*func)(struct builder *bld, int reg, int imm), int delta)
{
	const int imm = 100;
	struct builder bld;

	for (int reg = 0; reg < 16; reg++) {
		int count, actual_reg, actual_imm;

		reset_shader_pool();
		builder_init(&bld, 0, 0);

		func(&bld, reg, imm);
		builder_disasm(&bld);

		count = sscanf(bld.disasm_output, fmt, &actual_reg, &actual_imm);

		if (count != 2 || reg != actual_reg || imm - delta != actual_imm) {
			const uint8_t *code = (uint8_t *) bld.shader;
			const int size = bld.p - code;
			printf("fmt='%s' reg=%d imm=%d:\n    ", fmt, reg, imm);
			for (int i = 0; i < size; i++)
				printf("%02x ", code[i]);
			printf("%s\n", bld.disasm_output);
			exit(EXIT_FAILURE);
		}
	}
}

void
check_binop_emit_function(const char *fmt,
			  void (*func)(struct builder *bld, int reg, int imm))
{
	struct builder bld;

	for (int dst = 0; dst < 16; dst++) {
		for (int src = 0; src < 16; src++) {
			int count, actual_dst, actual_src;

			reset_shader_pool();
			builder_init(&bld, 0, 0);

			func(&bld, dst, src);
			builder_disasm(&bld);

			count = sscanf(bld.disasm_output, fmt,
				       &actual_src, &actual_dst);

			if (count != 2 ||
			    dst != actual_dst || src != actual_src) {
				const uint8_t *code = (uint8_t *) bld.shader;
				const int size = bld.p - code;
				printf("fmt='%s' dst=%d src=%d:\n    ", fmt, dst, src);
				for (int i = 0; i < size; i++)
					printf("%02x ", code[i]);
				printf("%s", bld.disasm_output);
				exit(EXIT_FAILURE);
			}
		}
	}
}

void
check_triop_emit_function(const char *name,
			  void (*func)(struct builder *bld, int dst, int src0, int src1))
{
	struct builder bld;

	for (int dst = 0; dst < 16; dst++)
		for (int src0 = 0; src0 < 16; src0++)
			for (int src1 = 0; src1 < 16; src1++) {
				int count, actual_dst, actual_src0, actual_src1;
				char actual_name[16];

				reset_shader_pool();
				builder_init(&bld, 0, 0);

				func(&bld, dst, src0, src1);
				builder_disasm(&bld);

				count = sscanf(bld.disasm_output,
					       "%16s %%ymm%d,%%ymm%d,%%ymm%d",
					       actual_name, &actual_src0, &actual_src1, &actual_dst);

				if (count != 4 || strcmp(name, actual_name) != 0 ||
				    dst != actual_dst || src0 != actual_src0 || src1 != actual_src1) {

					const uint8_t *code = (uint8_t *) bld.shader;

					const int size = bld.p - code;

					printf("%s dst=%d src0=%d src1=%d:\n    ",
					       name, dst, src0, src1);
					for (int i = 0; i < size; i++)
						printf("%02x ", code[i]);
					printf("%s", bld.disasm_output);

					exit(EXIT_FAILURE);
				}
			}
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

	check_triop_emit_function("vpaddd", builder_emit_vpaddd);
	check_triop_emit_function("vpsubd", builder_emit_vpsubd);
	check_triop_emit_function("vpmulld", builder_emit_vpmulld);
	check_triop_emit_function("vaddps", builder_emit_vaddps);
	check_triop_emit_function("vmulps", builder_emit_vmulps);
	check_triop_emit_function("vdivps", builder_emit_vdivps);
	check_triop_emit_function("vsubps", builder_emit_vsubps);
	check_triop_emit_function("vpand", builder_emit_vpand);
	check_triop_emit_function("vpxor", builder_emit_vpxor);
	check_triop_emit_function("vpor", builder_emit_vpor);
	check_triop_emit_function("vpsrlvd", builder_emit_vpsrlvd);
	check_triop_emit_function("vpsravd", builder_emit_vpsravd);
	check_triop_emit_function("vpsllvd", builder_emit_vpsllvd);
	check_triop_emit_function("vfmadd132ps", builder_emit_vfmadd132ps);
	check_triop_emit_function("vfmadd231ps", builder_emit_vfmadd231ps);
	check_triop_emit_function("vfnmadd132ps", builder_emit_vfnmadd132ps);

	check_binop_emit_function("vpabsd %%ymm%d,%%ymm%d", builder_emit_vpabsd); 
	check_binop_emit_function("vrsqrtps %%ymm%d,%%ymm%d", builder_emit_vrsqrtps);
	check_binop_emit_function("vsqrtps %%ymm%d,%%ymm%d", builder_emit_vsqrtps);
	check_binop_emit_function("vrcpps %%ymm%d,%%ymm%d", builder_emit_vrcpps);

	/* check_triop_emit_function("vcmpps", builder_emit_vcmpps); */

	check_triop_emit_function("vmaxps", builder_emit_vmaxps);
	check_triop_emit_function("vminps", builder_emit_vminps);

	/* builder_emit_vpblendvb */

#if 0
	/* xmm regs */
	check_triop_emit_function("vpackssdw", builder_emit_vpackssdw);
#endif
	check_binop_emit_function("vpmovsxwd %%xmm%d,%%ymm%d", builder_emit_vpmovsxwd);
	check_binop_emit_function("vpmovzxwd %%xmm%d,%%ymm%d", builder_emit_vpmovzxwd);

	/* builder_emit_vextractf128 */

	return EXIT_SUCCESS;
}

#endif
