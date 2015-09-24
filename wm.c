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

#include <stdlib.h>
#include <string.h>

#include "ksim.h"
#include "write-png.h"

struct payload {
	int area;
	float inv_area;
	int w2[8], w0[8], w1[8];
	int w2_offsets[8], w0_offsets[8], w1_offsets[8];
	int a01, b01, c01;
	int a12, b12, c12;
	int a20, b20, c20;

	int max_w0_delta, max_w1_delta, max_w2_delta;
	int max_group_w0_delta, max_group_w1_delta, max_group_w2_delta;

	struct reg attribute_deltas[64];
};

struct rt {
	const uint32_t *state;
	void *pixels;
	int width;
	int height;
	int stride;
	int cpp;
};

static bool
get_render_target(uint32_t binding_table_offset, int i, struct rt *rt)
{
	uint64_t offset, range;
	const uint32_t *binding_table;

	binding_table = map_gtt_offset(binding_table_offset +
				       gt.surface_state_base_address, &range);
	if (range < 4)
		return false;

	rt->state = map_gtt_offset(binding_table[i] +
				   gt.surface_state_base_address, &range);
	if (range < 16 * 4)
		return false;

	rt->width = field(rt->state[2], 0, 13) + 1;
	rt->height = field(rt->state[2], 16, 29) + 1;
	rt->stride = field(rt->state[3], 0, 17) + 1;
	rt->cpp = format_size(field(rt->state[0], 18, 26));

	offset = get_u64(&rt->state[8]);
	rt->pixels = map_gtt_offset(offset, &range);
	if (range < rt->height * rt->stride)
		return false;

	return true;
}

void
sfid_render_cache(struct thread *t, const struct send_args *args)
{
	uint32_t opcode = field(args->function_control, 14, 17);
	uint32_t type = field(args->function_control, 8, 10);
	uint32_t surface = field(args->function_control, 0, 7);
	uint32_t binding_table_offset;
	struct rt rt;
	bool rt_valid;
	uint32_t *p;
	int src = args->src;

	binding_table_offset = t->grf[0].ud[4];
	rt_valid = get_render_target(binding_table_offset, surface, &rt);
	ksim_assert(rt_valid);

	int x = t->grf[1].ud[2] & 0xffff;
	int y = t->grf[1].ud[2] >> 16;
	int sx, sy;

	switch (opcode) {
	case 12: /* rt write */
		switch (type) {
		case 4: /* simd8 */ {
			__m256i r, g, b, a, shift;
			struct reg argb;
			__m256 scale;
			scale = _mm256_set1_ps(255.0f);

			r = _mm256_cvtps_epi32(_mm256_mul_ps(t->grf[src + 0].reg, scale));
			g = _mm256_cvtps_epi32(_mm256_mul_ps(t->grf[src + 1].reg, scale));
			b = _mm256_cvtps_epi32(_mm256_mul_ps(t->grf[src + 2].reg, scale));
			a = _mm256_cvtps_epi32(_mm256_mul_ps(t->grf[src + 3].reg, scale));

			shift = _mm256_set1_epi32(8);
			argb.ireg = _mm256_sllv_epi32(a, shift);
			argb.ireg = _mm256_or_si256(argb.ireg, r);
			argb.ireg = _mm256_sllv_epi32(argb.ireg, shift);
			argb.ireg = _mm256_or_si256(argb.ireg, g);
			argb.ireg = _mm256_sllv_epi32(argb.ireg, shift);
			argb.ireg = _mm256_or_si256(argb.ireg, b);

			for (int i = 0; i < 8; i++) {
				if ((t->mask & (1 << i)) == 0)
					continue;
				sx = x + (i & 1) + (i / 2 & 2);
				sy = y + (i / 2 & 1);
				p = rt.pixels + sy * rt.stride + sx * rt.cpp;
				*p = argb.ud[i];
			}
			break;
		}
		default:
			stub("rt write type");
			break;
		}
		break;
	default:
		stub("render cache message type");
		break;
	}
}

static void
dispatch_ps(struct payload *p, uint32_t mask, int x, int y, int w1, int w2)
{
	struct thread t;
	uint32_t g;

	assert(gt.ps.enable_simd8);

	/* Not sure what we should make this. */
	uint32_t fftid = 0;

	t.mask = mask;
	/* Fixed function header */
	t.grf[0] = (struct reg) {
		.ud = {
			/* R0.0 */
			gt.ia.topology |
			0 /*  FIXME: More here */,
			/* R0.1 */
			gt.cc.state,
			/* R0.2: MBZ */
			0,
			/* R0.3: per-thread scratch space, sampler ptr */
			gt.ps.sampler_state_address |
			gt.ps.scratch_size,
			/* R0.4: binding table pointer */
			gt.ps.binding_table_address,
			/* R0.5: fftid, scratch offset */
			gt.ps.scratch_pointer | fftid,
			/* R0.6: thread id */
			gt.ps.tid++ & 0xffffff,
			/* R0.7: Reserved */
			0,
		}
	};

	t.grf[1] = (struct reg) {
		.ud = {
			/* R1.0-1: MBZ */
			0,
			0,
			/* R1.2: x, y for subspan 0  */
			(y << 16) | x,
			/* R1.3: x, y for subspan 1  */
			(y << 16) | (x + 2),
			/* R1.4: x, y for subspan 2 (SIMD16) */
			0 | 0,
			/* R1.5: x, y for subspan 3 (SIMD16) */
			0 | 0,
			/* R1.6: MBZ */
			0 | 0,
			/* R1.7: Pixel sample mask and copy */
			mask | (mask << 16)

		}
	};

	g = 2;
	if (gt.wm.barycentric_mode & BIM_PERSPECTIVE_PIXEL) {
		for (int i = 0; i < 8; i++) {
			t.grf[g].f[i] = p->w1[i] * p->inv_area;
			t.grf[g + 1].f[i] = p->w2[i] * p->inv_area;
		}
		g += 2;
		/* if (simd16) ... */
	}

	if (gt.wm.barycentric_mode & BIM_PERSPECTIVE_CENTROID) {
		for (int i = 0; i < 8; i++) {
			t.grf[g].f[i] = p->w1[i] * p->inv_area;
			t.grf[g + 1].f[i] = p->w2[i] * p->inv_area;
		}
		g += 2;
		/* if (simd16) ... */
	}

	if (gt.wm.barycentric_mode & BIM_PERSPECTIVE_SAMPLE) {
		for (int i = 0; i < 8; i++) {
			t.grf[g].f[i] = p->w1[i] * p->inv_area;
			t.grf[g + 1].f[i] = p->w2[i] * p->inv_area;
		}
		g += 2;
		/* if (simd16) ... */
	}

	if (gt.wm.barycentric_mode & BIM_LINEAR_PIXEL) {
		g++; /* barycentric[1], slots 0-7 */
		g++; /* barycentric[2], slots 0-7 */
		/* if (simd16) ... */
	}

	if (gt.wm.barycentric_mode & BIM_LINEAR_CENTROID) {
		g++; /* barycentric[1], slots 0-7 */
		g++; /* barycentric[2], slots 0-7 */
		/* if (simd16) ... */
	}

	if (gt.wm.barycentric_mode & BIM_LINEAR_SAMPLE) {
		g++; /* barycentric[1], slots 0-7 */
		g++; /* barycentric[2], slots 0-7 */
		/* if (simd16) ... */
	}

	if (gt.ps.uses_source_depth) {
		g++; /* interpolated depth, slots 0-7 */
	}

	if (gt.ps.uses_source_w) {
		g++; /* interpolated w, slots 0-7 */
	}

	if (gt.ps.position_offset_xy == POSOFFSET_CENTROID) {
		g++;
	} else if (gt.ps.position_offset_xy == POSOFFSET_SAMPLE) {
		g++;
	}

	if (gt.ps.uses_input_coverage_mask) {
		g++;
	}

	if (gt.ps.push_constant_enable)
		g = load_constants(&t, &gt.ps.curbe, gt.ps.grf_start0);
	else
		g = gt.ps.grf_start0;

	if (gt.ps.attribute_enable) {
		memcpy(&t.grf[g], p->attribute_deltas,
		       gt.sbe.num_attributes * 2 * sizeof(t.grf[0]));
	}

	if (gt.ps.statistics)
		gt.ps_invocation_count++;

#if 1
	void (*f)(struct thread *t) = (void *) gt.ps.avx_shader->code;
	f(&t);
#else
	run_thread(&t, gt.ps.shader, TRACE_PS);
#endif
}

const int cpp = 4;
const int tile_width = 128 / 4;
const int tile_height = 32;

static void
rasterize_tile(struct payload *p, int x0, int y0,
	       int start_w2, int start_w0, int start_w1, void *tile, uint32_t stride)
{
	int row_w2 = start_w2;
	int row_w0 = start_w0;
	int row_w1 = start_w1;

	for (int y = 0; y < tile_height; y += 2) {
		int w2 = row_w2;
		int w0 = row_w0;
		int w1 = row_w1;

		for (int x = 0; x < tile_width; x += 4) {
			int max_w2 = w2 + p->max_group_w2_delta;
			int max_w0 = w0 + p->max_group_w0_delta;
			int max_w1 = w1 + p->max_group_w1_delta;
			if ((max_w2 | max_w0 | max_w1) < 0)
				goto next;

			for (int i = 0; i < 8; i++) {
				p->w2[i] = w2 + p->w2_offsets[i];
				p->w0[i] = w0 + p->w0_offsets[i];
				p->w1[i] = p->area - p->w2[i] - p->w0[i];
			}

			uint32_t mask = 0;
			for (int i = 0; i < 8; i++)
				if ((p->w1[i] | p->w0[i] | p->w2[i]) > 0)
					mask |= (1 << i);

			if (mask)
				dispatch_ps(p, mask, x0 + x, y0 + y, w1, w2);

		next:
			w2 += p->a01 * 4;
			w0 += p->a12 * 4;
			w1 += p->a20 * 4;
		}
		row_w2 += 2 * p->b01;
		row_w0 += 2 * p->b12;
		row_w1 += 2 * p->b20;
	}
}

void
rasterize_primitive(struct primitive *prim)
{
	struct rt rt;

	if (!get_render_target(gt.ps.binding_table_address, 0, &rt)) {
		spam("assuming binding_table[0], but nothing valid there\n");
		return;
	}

	const int x0 = prim->v[0].x;
	const int y0 = prim->v[0].y;
	const int x1 = prim->v[1].x;
	const int y1 = prim->v[1].y;
	const int x2 = prim->v[2].x;
	const int y2 = prim->v[2].y;

	struct payload p;

	p.a01 = (y1 - y0);
	p.b01 = (x0 - x1);
	p.c01 = (x1 * y0 - y1 * x0);

	p.a12 = (y2 - y1);
	p.b12 = (x1 - x2);
	p.c12 = (x2 * y1 - y2 * x1);

	p.a20 = (y0 - y2);
	p.b20 = (x2 - x0);
	p.c20 = (x0 * y2 - y0 * x2);

	p.area = p.a01 * x2 + p.b01 * y2 + p.c01;

	if (p.area <= 0)
		return;
	p.inv_area = 1.0f / p.area;

	for (uint32_t i = 0; i < gt.sbe.num_attributes; i++) {
		const struct value a0 = prim->vue[0][i + 2];
		const struct value a1 = prim->vue[1][i + 2];
		const struct value a2 = prim->vue[2][i + 2];

		p.attribute_deltas[i * 2] = (struct reg) {
			.f = {
				a1.vec4.x - a0.vec4.x,
				a2.vec4.x - a0.vec4.x,
				0,
				a0.vec4.x,
				a1.vec4.y - a0.vec4.y,
				a2.vec4.y - a0.vec4.y,
				0,
				a0.vec4.y,
			}
		};
		p.attribute_deltas[i * 2 + 1] = (struct reg) {
			.f = {
				a1.vec4.z - a0.vec4.z,
				a2.vec4.z - a0.vec4.z,
				0,
				a0.vec4.z,
				a1.vec4.w - a0.vec4.w,
				a2.vec4.w - a0.vec4.w,
				0,
				a0.vec4.w,
			}
		};
	}

	const int tile_max_x = 128 / 4 - 1;
	const int tile_max_y = 31;
	const int group_max_x = 3;
	const int group_max_y = 1;

	int w2_max_x = p.a01 > 0 ? 1 : 0;
	int w2_max_y = p.b01 > 0 ? 1 : 0;

	/* delta from w2 in top-left corner to maximum w2 in tile */
	p.max_w2_delta = p.a01 * w2_max_x * tile_max_x + p.b01 * w2_max_y * tile_max_y;
	p.max_group_w2_delta =
		p.a01 * w2_max_x * group_max_x + p.b01 * w2_max_y * group_max_y;

	int w0_max_x = p.a12 > 0 ? 1: 0;
	int w0_max_y = p.b12 > 0 ? 1 : 0;

	/* delta from w2 in top-left corner to maximum w2 in tile */
	p.max_w0_delta = p.a12 * w0_max_x * tile_max_x + p.b12 * w0_max_y * tile_max_y;
	p.max_group_w0_delta =
		p.a12 * w0_max_x * group_max_x + p.b12 * w0_max_y * group_max_y;

	int w1_max_x = p.a20 > 0 ? 1 : 0;
	int w1_max_y = p.b20 > 0 ? 1 : 0;

	/* delta from w2 in top-left corner to maximum w2 in tile */
	p.max_w1_delta = p.a20 * w1_max_x * tile_max_x + p.b20 * w1_max_y * tile_max_y;
	p.max_group_w1_delta =
		p.a20 * w1_max_x * group_max_x + p.b20 * w1_max_y * group_max_y;

	for (int i = 0; i < 8; i++) {
		int sx = (i & 1) + ((i & ~3) >> 1);
		int sy = (i & 2) >> 1;
		p.w2_offsets[i] = p.a01 * sx + p.b01 * sy;
		p.w0_offsets[i] = p.a12 * sx + p.b12 * sy;
		p.w1_offsets[i] = p.a20 * sx + p.b20 * sy;
	}

	int row_w2 = p.c01;
	int row_w0 = p.c12;
	int row_w1 = p.c20;
	for (int y = 0; y < rt.height; y += tile_height) {
		int w2 = row_w2;
		int w0 = row_w0;
		int w1 = row_w1;

		for (int x = 0; x < rt.width; x += tile_width) {
			int max_w2 = w2 + p.max_w2_delta;
			int max_w0 = w0 + p.max_w0_delta;
			int max_w1 = w1 + p.max_w1_delta;

			if ((max_w2 | max_w0 | max_w1) >= 0) {
				void *tile = rt.pixels + y * rt.stride + x * 4;
				rasterize_tile(&p, x, y, w2, w0, w1, tile, rt.stride);
			}

			w2 += tile_width * p.a01;
			w0 += tile_width * p.a12;
			w1 += tile_width * p.a20;
		}

		row_w2 += tile_height * p.b01;
		row_w0 += tile_height * p.b12;
		row_w1 += tile_height * p.b20;
	}
}

void
wm_flush(void)
{
	struct rt rt;

	if (framebuffer_filename &&
	    get_render_target(gt.ps.binding_table_address, 0, &rt))
		write_png(framebuffer_filename,
			  rt.width, rt.height, rt.stride, rt.pixels);
}
