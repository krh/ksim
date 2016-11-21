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

#define NO_KERNEL 1

void
prepare_shaders(void)
{
	uint64_t ksp_simd8 = NO_KERNEL, ksp_simd16 = NO_KERNEL, ksp_simd32 = NO_KERNEL;

	reset_shader_pool();

	if (gt.vs.enable) {
		ksim_trace(TRACE_EU | TRACE_AVX, "jit vs\n");
		gt.vs.avx_shader =
			compile_shader(gt.vs.ksp,
				       gt.vs.binding_table_address,
				       gt.vs.sampler_state_address);
	}

	if (gt.ps.enable) {
		if (gt.ps.enable_simd8) {
			ksp_simd8 = gt.ps.ksp0;
			if (gt.ps.enable_simd16) {
				ksp_simd16 = gt.ps.ksp2;
				if (gt.ps.enable_simd32)
					ksp_simd32 = gt.ps.ksp1;
			} else {
				ksp_simd32 = gt.ps.ksp2;
			}
		} else {
			if (gt.ps.enable_simd16) {
				if(gt.ps.enable_simd32) {
					ksp_simd16 = gt.ps.ksp2;
					ksp_simd32 = gt.ps.ksp1;
				} else {
					ksp_simd16 = gt.ps.ksp0;
				}
			} else {
				ksp_simd32 = gt.ps.ksp0;
			}
		}

		if (ksp_simd8 != NO_KERNEL) {
			ksim_trace(TRACE_EU | TRACE_AVX, "jit simd8 ps\n");
			gt.ps.avx_shader_simd8 =
				compile_shader(ksp_simd8,
					       gt.ps.binding_table_address,
					       gt.ps.sampler_state_address);
		}
		if (ksp_simd16 != NO_KERNEL) {
			ksim_trace(TRACE_EU | TRACE_AVX, "jit simd16 ps\n");
			gt.ps.avx_shader_simd16 =
				compile_shader(ksp_simd16,
					       gt.ps.binding_table_address,
					       gt.ps.sampler_state_address);
		}
		if (ksp_simd32 != NO_KERNEL) {
			ksim_trace(TRACE_EU | TRACE_AVX, "jit simd32 ps\n");
			gt.ps.avx_shader_simd32 =
				compile_shader(ksp_simd32,
					       gt.ps.binding_table_address,
					       gt.ps.sampler_state_address);
		}
	}
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

	static const struct reg offsets = { .d = {  0, 32, 64, 96, 128, 160, 192, 224 } };
	uint32_t *handles = t->grf[args->src].ud;
	uint32_t *values = t->grf[args->src + 1].ud;
	uint32_t c;

	if (args->len == 9) {
		for_each_bit (c, t->mask) {
			struct value *vue = urb_handle_to_entry(handles[c]);
			__m256i e = _mm256_i32gather_epi32((void *) &values[c],
							   offsets.ireg, 1);
			_mm256_storeu_si256((void *) &vue[args->offset], e);
		}
	} else if (args->len == 5) {
		for_each_bit (c, t->mask) {
			struct value *vue = urb_handle_to_entry(handles[c]);
			__m128i e = _mm_i32gather_epi32((void *) &values[c],
							offsets.ihreg, 1);
			_mm_storeu_si128((void *) &vue[args->offset], e);
		}
	} else {
		for_each_bit (c, t->mask) {
			struct value *vue = urb_handle_to_entry(handles[c]);
			for (int i = 0; i < args->len - 1; i++)
				vue[args->offset + i / 4].v[i % 4] =
					t->grf[args->src + 1 + i].ud[c];
		}
	}
}
