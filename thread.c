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
