/*
 * Copyright © 2015 Intel Corporation
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

#include "ksim.h"
#include "libdisasm/gen_disasm.h"

uint32_t
load_constants(struct thread *t, struct curbe *c, uint32_t start)
{
	uint32_t grf = start;
	struct reg *regs;
	uint64_t base, range;

	for (uint32_t b = 0; b < 4; b++) {
		if (b == 0 && gt.curbe_dynamic_state_base)
			base = gt.dynamic_state_base_address;
		else
			base = 0;

		if (c->buffer[b].length > 0) {
			regs = map_gtt_offset(c->buffer[b].address + base, &range);
			ksim_assert(c->buffer[b].length * sizeof(regs[0]) <= range);
		}

		for (uint32_t i = 0; i < c->buffer[b].length; i++)
			t->grf[grf++] = regs[i];
	}

	return grf;
}


static struct gen_disasm *
get_disasm(void)
{
	const int gen = 8;
	static struct gen_disasm *disasm;

	if (disasm == NULL)
		disasm = gen_disasm_create(gen);

	return disasm;
}

void
print_inst(void *p)
{
	gen_disasm_disassemble_insn(get_disasm(), p, stdout);
}

void
prepare_shaders(void)
{
	uint64_t ksp, range;
	char cache[64 * 1024];
	void *kernel, *end;

	static void *pool;
	const size_t size = 64 * 1024;
	if (pool == NULL) {
		int fd = memfd_create("jit", MFD_CLOEXEC);
		ftruncate(fd, size);
		pool = mmap(NULL, size, PROT_WRITE | PROT_READ | PROT_EXEC, MAP_SHARED, fd, 0);
		close(fd);
	}

	end = pool;
	if (gt.vs.enable) {
		ksp = gt.vs.ksp + gt.instruction_base_address;
		kernel = map_gtt_offset(ksp, &range);
		gen_disasm_uncompact(get_disasm(), kernel,
				     cache, sizeof(cache));

		gt.vs.avx_shader = end;
		end = compile_shader(cache, gt.vs.avx_shader,
				     gt.vs.binding_table_address,
				     gt.vs.sampler_state_address);
	}

	ksp = gt.ps.ksp0 + gt.instruction_base_address;
	kernel = map_gtt_offset(ksp, &range);
	gen_disasm_uncompact(get_disasm(), kernel,
			     cache, sizeof(cache));

	gt.ps.avx_shader = align_ptr(end, 64);
	end = compile_shader(cache, gt.ps.avx_shader,
			     gt.ps.binding_table_address,
			     gt.ps.sampler_state_address);

	ksim_assert(end - pool < size);
}

#define GEN5_SAMPLER_MESSAGE_SAMPLE              0
#define GEN5_SAMPLER_MESSAGE_SAMPLE_BIAS         1
#define GEN5_SAMPLER_MESSAGE_SAMPLE_LOD          2
#define GEN5_SAMPLER_MESSAGE_SAMPLE_COMPARE      3
#define GEN5_SAMPLER_MESSAGE_SAMPLE_DERIVS       4
#define GEN5_SAMPLER_MESSAGE_SAMPLE_BIAS_COMPARE 5
#define GEN5_SAMPLER_MESSAGE_SAMPLE_LOD_COMPARE  6
#define GEN5_SAMPLER_MESSAGE_SAMPLE_LD           7
#define GEN7_SAMPLER_MESSAGE_SAMPLE_GATHER4      8
#define GEN5_SAMPLER_MESSAGE_LOD                 9
#define GEN5_SAMPLER_MESSAGE_SAMPLE_RESINFO      10
#define GEN7_SAMPLER_MESSAGE_SAMPLE_GATHER4_C    16
#define GEN7_SAMPLER_MESSAGE_SAMPLE_GATHER4_PO   17
#define GEN7_SAMPLER_MESSAGE_SAMPLE_GATHER4_PO_C 18
#define HSW_SAMPLER_MESSAGE_SAMPLE_DERIV_COMPARE 20
#define GEN7_SAMPLER_MESSAGE_SAMPLE_LD_MCS       29
#define GEN7_SAMPLER_MESSAGE_SAMPLE_LD2DMS       30
#define GEN7_SAMPLER_MESSAGE_SAMPLE_LD2DSS       31

/* for GEN5 only */
#define BRW_SAMPLER_SIMD_MODE_SIMD4X2                   0
#define BRW_SAMPLER_SIMD_MODE_SIMD8                     1
#define BRW_SAMPLER_SIMD_MODE_SIMD16                    2
#define BRW_SAMPLER_SIMD_MODE_SIMD32_64                 3

void
sfid_sampler(struct thread *t, const struct sfid_sampler_args *args)
{
#if 0
	/* Clamp */
	struct reg u;
	u.reg = _mm256_min_ps(t->grf[args->src].reg, _mm256_set1_ps(1.0f));
	u.reg = _mm256_max_ps(u.reg, _mm256_setzero_ps());

	struct reg v;
	v.reg = _mm256_min_ps(t->grf[args->src + 1].reg, _mm256_set1_ps(1.0f));
	v.reg = _mm256_max_ps(v.reg, _mm256_setzero_ps());
#endif

	/* Wrap */
	struct reg u;
	u.reg = _mm256_floor_ps(t->grf[args->src].reg);
	u.reg = _mm256_sub_ps(t->grf[args->src].reg, u.reg);

	struct reg v;
	v.reg = _mm256_floor_ps(t->grf[args->src + 1].reg);
	v.reg = _mm256_sub_ps(t->grf[args->src + 1].reg, v.reg);

	u.reg = _mm256_mul_ps(u.reg, _mm256_set1_ps(args->tex.width - 1));
	v.reg = _mm256_mul_ps(v.reg, _mm256_set1_ps(args->tex.height - 1));

	u.ireg = _mm256_cvttps_epi32(u.reg);
	v.ireg = _mm256_cvttps_epi32(v.reg);

	struct reg offsets;
	offsets.ireg =
		_mm256_add_epi32(_mm256_mullo_epi32(u.ireg, _mm256_set1_epi32(args->tex.cpp)),
				 _mm256_mullo_epi32(v.ireg, _mm256_set1_epi32(args->tex.stride)));
	struct reg argb32;
	argb32.ireg = _mm256_i32gather_epi32(args->tex.pixels, offsets.ireg, 1);

	/* Unpack RGBX */
	__m256i mask = _mm256_set1_epi32(0xff);
	__m256 scale = _mm256_set1_ps(1.0f / 255.0f);
	t->grf[args->dst + 0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(argb32.ireg, mask)), scale);
	argb32.ireg = _mm256_srli_epi32(argb32.ireg, 8);
	t->grf[args->dst + 1].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(argb32.ireg, mask)), scale);
	argb32.ireg = _mm256_srli_epi32(argb32.ireg, 8);
	t->grf[args->dst + 2].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(argb32.ireg, mask)), scale);

	t->grf[args->dst + 3].reg = _mm256_set1_ps(1.0f);
}

void
sfid_urb_simd8_write(struct thread *t, struct sfid_urb_args *args)
{
	if (trace_mask & TRACE_URB) {
		ksim_trace(TRACE_URB,
			   "urb simd8 write, src g%d, global offset %d, mlen %lu, mask %02x\n",
			   args->src, args->offset, args->len, t->mask);

		ksim_trace(TRACE_URB, "  grf%d:", args->src);
		for (int c = 0; c < 8; c++)
			ksim_trace(TRACE_URB, "  %6d", t->grf[args->src].d[c]);
		ksim_trace(TRACE_URB, "\n");

		for (int i = 1; i < args->len; i++) {
			ksim_trace(TRACE_URB, "  grf%d:", args->src + i);
			for (int c = 0; c < 8; c++)
				ksim_trace(TRACE_URB,
					   "  %6.1f", t->grf[args->src + i].f[c]);
			ksim_trace(TRACE_URB, "\n");
		}
	}

	uint32_t c;
	for_each_bit (c, t->mask) {
		struct value *vue = urb_handle_to_entry(t->grf[args->src].ud[c]);
		for (int i = 0; i < args->len - 1; i++)
			vue[args->offset + i / 4].v[i % 4] =
				t->grf[args->src + 1 + i].ud[c];
	}
}
