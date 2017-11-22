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
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>

#include "ksim.h"
#include "kir.h"

struct edge {
	int32_t a, b, c, bias;
};

struct dispatch {
	struct reg w, z;
	struct reg int_w2, int_w1;
	struct reg w2, w1;
	struct reg w2_pc, w1_pc;
	int x, y;
};

struct ps_primitive {
	float w_deltas[4];
	int32_t area;
	struct edge e01, e12, e20;
	struct reg attribute_deltas[64];

	/* Tile iterator step values */
	__m256i w2_offsets, w0_offsets, w1_offsets;
	__m256i w2_step, w0_step, w1_step;
	__m256i w2_row_step, w0_row_step, w1_row_step;
};

struct ps_thread {
	struct thread t;
	struct reg grf0;
	struct dispatch queue[2];
	int queue_length;

	float inv_area;
	float w_deltas[4];
	void *depth;
	int32_t e01_bias;
	int32_t e20_bias;
	struct reg attribute_deltas[64];

	uint32_t invocation_count;
};

static void
emit_barycentric_conversion(struct kir_program *prog)
{
	kir_program_comment(prog, "compute barycentric coordinates");
	struct kir_reg inv_area =
		kir_program_load_uniform(prog, offsetof(struct ps_thread, inv_area));
	struct kir_reg e01_bias =
		kir_program_load_uniform(prog, offsetof(struct ps_thread, e01_bias));
	struct kir_reg e20_bias =
		kir_program_load_uniform(prog, offsetof(struct ps_thread, e20_bias));
	struct kir_reg w2 =
		kir_program_load_v8(prog, offsetof(struct ps_thread, queue[0].int_w2));
	struct kir_reg w1 =
		kir_program_load_v8(prog, offsetof(struct ps_thread, queue[0].int_w1));

	w2 = kir_program_alu(prog, kir_addd, w2, e01_bias);
	w1 = kir_program_alu(prog, kir_addd, w1, e20_bias);
	w2 = kir_program_alu(prog, kir_d2ps, w2);
	w1 = kir_program_alu(prog, kir_d2ps, w1);
	w2 = kir_program_alu(prog, kir_mulf, w2, inv_area);
	w1 = kir_program_alu(prog, kir_mulf, w1, inv_area);

	kir_program_store_v8(prog, offsetof(struct ps_thread, queue[0].w1), w1);
	kir_program_store_v8(prog, offsetof(struct ps_thread, queue[0].w1_pc), w1);
	kir_program_store_v8(prog, offsetof(struct ps_thread, queue[0].w2), w2);
	kir_program_store_v8(prog, offsetof(struct ps_thread, queue[0].w2_pc), w2);
}

static void
emit_depth_test(struct kir_program *prog)
{
	struct kir_reg base, depth;

	kir_program_comment(prog, "compute depth");
	struct kir_reg b =
		kir_program_load_uniform(prog, offsetof(struct ps_thread, w_deltas[1]));
	struct kir_reg c =
		kir_program_load_uniform(prog, offsetof(struct ps_thread, w_deltas[3]));
	kir_program_load_v8(prog, offsetof(struct ps_thread, queue[0].w2));
	struct kir_reg d = kir_program_alu(prog, kir_maddf, b, prog->dst, c);

	struct kir_reg a =
		kir_program_load_uniform(prog, offsetof(struct ps_thread, w_deltas[0]));
	kir_program_load_v8(prog, offsetof(struct ps_thread, queue[0].w1));
	struct kir_reg w =
		kir_program_alu(prog, kir_maddf, a, prog->dst, d);

	kir_program_store_v8(prog, offsetof(struct ps_thread, queue[0].w), w);

	struct kir_reg z = kir_program_alu(prog, kir_rcp, w);
	kir_program_store_v8(prog, offsetof(struct ps_thread, queue[0].z), z);

	if (!gt.depth.test_enable && !gt.depth.write_enable)
		return;

	kir_program_comment(prog, "load depth");
	base = kir_program_set_load_base_indirect(prog, offsetof(struct ps_thread, depth));
	switch (gt.depth.format) {
	case D32_FLOAT:
		depth = kir_program_load(prog, base, 0);
		break;
	case D24_UNORM_X8_UINT:
		depth = kir_program_load(prog, base, 0);
		depth = kir_program_alu(prog, kir_d2ps, depth);
		kir_program_immf(prog, 1.0f / 16777215.0f);
		depth = kir_program_alu(prog, kir_mulf, depth, prog->dst);
		break;
	case D16_UNORM:
		stub("D16_UNORM");
		break;
	default:
		ksim_unreachable("invalid depth format");
	}

	/* Swizzle two middle pixel pairs so that dword 0-3 and 4-7
	 * match the shader dispatch subspan ordering. */
	// d_f.ireg = _mm256_permute4x64_epi64(d_f.ireg, SWIZZLE(0, 2, 1, 3));

	struct kir_reg computed_depth = w;
	struct kir_reg mask =
		kir_program_load_v8(prog, offsetof(struct ps_thread, t.mask[0].q[0]));

	if (gt.depth.test_enable) {
		kir_program_comment(prog, "depth test");

		static const uint32_t gen_function_to_avx2[] = {
			[COMPAREFUNCTION_ALWAYS]	= _CMP_TRUE_US,
			[COMPAREFUNCTION_NEVER]		= _CMP_FALSE_OS,
			[COMPAREFUNCTION_LESS]		= _CMP_LT_OS,
			[COMPAREFUNCTION_EQUAL]		= _CMP_EQ_OS,
			[COMPAREFUNCTION_LEQUAL]	= _CMP_LE_OS,
			[COMPAREFUNCTION_GREATER]	= _CMP_GT_OS,
			[COMPAREFUNCTION_NOTEQUAL]	= _CMP_NEQ_OS,
			[COMPAREFUNCTION_GEQUAL]	= _CMP_GE_OS,
		};

		kir_program_alu(prog, kir_cmpf, computed_depth, depth,
				gen_function_to_avx2[gt.depth.test_function]);
		mask = kir_program_alu(prog, kir_and, mask, prog->dst);
		kir_program_store_v8(prog, offsetof(struct ps_thread, t.mask[0].q[0]), mask);
	}

	if (gt.depth.write_enable) {
		kir_program_comment(prog, "write depth");

#if 0
		struct reg w;
		w.ireg = _mm256_permute4x64_epi64(d->w.ireg, SWIZZLE(0, 2, 1, 3));
		__m256i m = _mm256_permute4x64_epi64(d->mask.ireg,
						     SWIZZLE(0, 2, 1, 3));
#endif
		struct kir_reg r;

		switch (gt.depth.format) {
		case D32_FLOAT:
			kir_program_mask_store(prog, base, 0, computed_depth, mask);
			break;
		case D24_UNORM_X8_UINT:
			r = computed_depth;
			kir_program_immf(prog, 16777215.0f);
			r = kir_program_alu(prog, kir_mulf, r, prog->dst);
			kir_program_immf(prog, 0.5f);
			kir_program_alu(prog, kir_addf, r, prog->dst);
			kir_program_alu(prog, kir_ps2d, prog->dst);
			kir_program_mask_store(prog, base, 0, prog->dst, mask);
			break;
		case D16_UNORM:
			stub("D16_UNORM");
			break;
		default:
			ksim_unreachable("invalid depth format");
		}

	}

	if (gt.depth.test_enable) {
		struct kir_insn *insn = kir_program_add_insn(prog, kir_eot_if_dead);
		insn->eot.src = mask;
	}
}

static void
dispatch_ps(struct ps_thread *t)
{
	struct dispatch *d = &t->queue[0];
	int count = t->queue_length;
	/* Not sure what we should make this. */
	struct reg *grf = &t->t.grf[0];

	uint32_t mask = _mm256_movemask_ps((__m256) t->t.mask[0].q[0]);
	grf[1] = (struct reg) {
		.ud = {
			/* R1.0-1: MBZ */
			0,
			0,
			/* R1.2: x, y for subspan 0  */
			(d[0].y << 16) | d[0].x,
			/* R1.3: x, y for subspan 1  */
			(d[0].y << 16) | (d[0].x + 2),
			/* R1.4: x, y for subspan 2 (SIMD16) */
			(d[1].y << 16) | d[1].x,
			/* R1.5: x, y for subspan 3 (SIMD16) */
			(d[1].y << 16) | (d[1].x + 2),
			/* R1.6: MBZ */
			0 | 0,
			/* R1.7: Pixel sample mask and copy */
			mask | (mask << 16)

		}
	};

	t->invocation_count++;

	if (count == 1 && gt.ps.enable_simd8) {
		gt.ps.avx_shader_simd8(&t->t);
	} else if (gt.ps.enable_simd16) {
		gt.ps.avx_shader_simd16(&t->t);
	}
}

const int tile_width = 128 / 4;
const int tile_height = 32;

struct tile_iterator {
	int x, y, x0, y0;
	__m256i w2, w0, w1;
};

static void
clear_depth_tile(uint32_t x, uint32_t y)
{
	uint32_t tile_stride = DIV_ROUND_UP(gt.depth.width, 32);
	uint8_t *hiz_tile = gt.depth.hiz_buffer + x / 32 + tile_stride * (y / 32);

	if (*hiz_tile)
		return;
	*hiz_tile = 1;

	struct reg clear_value;
	uint32_t cpp = depth_format_size(gt.depth.format);
	void *depth = ymajor_offset(gt.depth.buffer, x, y, gt.depth.stride, cpp);

	switch (gt.depth.format) {
	case D32_FLOAT:
		clear_value.reg = _mm256_set1_ps(gt.depth.clear_value);
		break;
	case D24_UNORM_X8_UINT:
		clear_value.ireg = _mm256_set1_epi32(gt.depth.clear_value * 16777215.0f);
		break;
	case D16_UNORM:
		clear_value.ireg = _mm256_set1_epi16(gt.depth.clear_value * 65535.0f);
		break;
	default:
		ksim_unreachable("invalid depth format");
	}

	for (uint32_t i = 0; i < 4096; i += 128) {
		_mm256_store_si256((depth + i +   0), clear_value.ireg);
		_mm256_store_si256((depth + i +  32), clear_value.ireg);
		_mm256_store_si256((depth + i +  64), clear_value.ireg);
		_mm256_store_si256((depth + i +  96), clear_value.ireg);
	}
}

struct bbox_iter {
	uint32_t x, y;
	struct rectangle rect;
	int32_t w2, w0, w1;
	int32_t w2_step, w0_step, w1_step;
	int32_t w2_row_step, w0_row_step, w1_row_step;
};

static void
tile_iterator_init(struct tile_iterator *iter,
		   struct ps_primitive *p, const struct bbox_iter *bbox_iter)
{
	iter->x = 0;
	iter->y = 0;
	iter->x0 = bbox_iter->x;
	iter->y0 = bbox_iter->y;

	if (gt.depth.write_enable || gt.depth.test_enable)
		if (gt.depth.hiz_enable)
			clear_depth_tile(iter->x0, iter->y0);

	iter->w2 = _mm256_add_epi32(_mm256_set1_epi32(bbox_iter->w2),
				    p->w2_offsets);

	iter->w0 = _mm256_add_epi32(_mm256_set1_epi32(bbox_iter->w0),
				    p->w0_offsets);

	iter->w1 = _mm256_add_epi32(_mm256_set1_epi32(bbox_iter->w1),
				    p->w1_offsets);
}

static bool
tile_iterator_done(struct tile_iterator *iter)
{
	return iter->y == tile_height;
}

static void
tile_iterator_next(struct tile_iterator *iter, struct ps_primitive *p)
{
	iter->x += 4;
	if (iter->x == tile_width) {
		iter->x = 0;
		iter->y += 2;

		iter->w2 = _mm256_add_epi32(iter->w2, p->w2_row_step);
		iter->w0 = _mm256_add_epi32(iter->w0, p->w0_row_step);
		iter->w1 = _mm256_add_epi32(iter->w1, p->w1_row_step);
	} else {
		iter->w2 = _mm256_add_epi32(iter->w2, p->w2_step);
		iter->w0 = _mm256_add_epi32(iter->w0, p->w0_step);
		iter->w1 = _mm256_add_epi32(iter->w1, p->w1_step);
	}

}

static void
fill_dispatch(struct ps_thread *pt,
	      struct tile_iterator *iter, struct reg mask)
{
	uint32_t q = pt->queue_length;
	struct dispatch *d = &pt->queue[q];

	if (_mm256_movemask_ps(mask.reg) == 0)
		return;

	/* Some pixels are covered and we have to calculate
	 * barycentric coordinates. We add back the tie-breaker
	 * adjustment so as to not distort the barycentric
	 * coordinates.*/
	d->int_w2.ireg = iter->w2;
	d->int_w1.ireg = iter->w1;
	
	pt->t.mask[0].q[q] = mask.ireg;
	d->x = iter->x0 + iter->x;
	d->y = iter->y0 + iter->y;

	if (gt.depth.write_enable || gt.depth.test_enable) {
		uint32_t cpp = depth_format_size(gt.depth.format);
		pt->depth = ymajor_offset(gt.depth.buffer, d->x, d->y, gt.depth.stride, cpp);
	}

	pt->queue_length++;
	if (gt.ps.enable_simd8 || pt->queue_length == 2) {
		dispatch_ps(pt);
		pt->queue_length = 0;
	}
}

static void
init_ps_thread(struct ps_thread *pt, struct ps_primitive *p)
{
	pt->queue_length = 0;
	pt->invocation_count = 0;
	pt->inv_area = 1.0f / p->area;
	memcpy(pt->w_deltas, p->w_deltas, sizeof(pt->w_deltas));
	pt->e01_bias = p->e01.bias;
	pt->e20_bias = p->e20.bias;

	for (uint32_t i = 0; i < gt.sbe.num_attributes * 2; i++)
		pt->attribute_deltas[i] = p->attribute_deltas[i];

	uint32_t fftid = 0;
	pt->grf0 = (struct reg) {
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
}

static void
finish_ps_thread(struct ps_thread *pt)
{
	if (pt->queue_length > 0)
		dispatch_ps(pt);
	if (gt.ps.statistics)
		gt.ps_invocation_count += pt->invocation_count;
}

static void
rasterize_rectlist_tile(struct ps_primitive *p, struct bbox_iter *bbox_iter)
{
	struct tile_iterator iter;
	struct ps_thread pt;

	init_ps_thread(&pt, p);

	/* To determine coverage, we compute the edge function for all
	 * edges in the rectangle. We only have two of the four edges,
	 * but we can compute the edge function from the opposite edge
	 * by subtracting from the area. We also subtract 1 to either
	 * cancel out the bias on the original edge, or to add it to
	 * the opposite edge if the original doesn't have bias. */
	__m256i c = _mm256_set1_epi32(p->area - 1);

	for (tile_iterator_init(&iter, p, bbox_iter);
	     !tile_iterator_done(&iter);
	     tile_iterator_next(&iter, p)) {
		__m256i w2, w3;

		w2 = _mm256_sub_epi32(c, iter.w2);
		w3 = _mm256_sub_epi32(c, iter.w0);

		struct reg mask;
		mask.ireg = _mm256_and_si256(_mm256_and_si256(iter.w2, iter.w0),
					     _mm256_and_si256(w2, w3));

		fill_dispatch(&pt, &iter, mask);
	}

	finish_ps_thread(&pt);
}

static void
rasterize_triangle_tile(struct ps_primitive *p, const struct bbox_iter *bbox_iter)
{
	struct tile_iterator iter;
	struct ps_thread pt;

	init_ps_thread(&pt, p);

	for (tile_iterator_init(&iter, p, bbox_iter);
	     !tile_iterator_done(&iter);
	     tile_iterator_next(&iter, p)) {
		struct reg mask;
		mask.ireg =
			_mm256_and_si256(_mm256_and_si256(iter.w1,
							  iter.w0), iter.w2);

		fill_dispatch(&pt, &iter, mask);
	}

	finish_ps_thread(&pt);
}

struct point {
	int32_t x, y;
};

static inline struct point
snap_point(float x, float y)
{
	return (struct point) {
		(int32_t) (x * 256.0f),
		(int32_t) (y * 256.0f)
	};
}

static inline void
init_edge(struct edge *e, struct point p0, struct point p1)
{
	e->a = (p0.y - p1.y);
	e->b = (p1.x - p0.x);
	e->c = ((int64_t) p1.y * p0.x - (int64_t) p1.x * p0.y) >> 8;
	e->bias = e->a < 0 || (e->a == 0 && e->b < 0);
}

static inline void
invert_edge(struct edge *e)
{
	e->a = -e->a;
	e->b = -e->b;
	e->c = -e->c;
	e->bias = 1 - e->bias;
}

static inline int
eval_edge(struct edge *e, struct point p)
{
	return (((int64_t) e->a * p.x + (int64_t) e->b * p.y) >> 8) + e->c - e->bias;
}

static void
bbox_iter_init(struct bbox_iter *iter, struct ps_primitive *p, struct rectangle *rect)
{
	iter->x = rect->x0;
	iter->y = rect->y0;
	iter->rect = *rect;

	struct point min = snap_point(rect->x0, rect->y0);
	min.x += 128;
	min.y += 128;
	iter->w2 = eval_edge(&p->e01, min);
	iter->w0 = eval_edge(&p->e12, min);
	iter->w1 = eval_edge(&p->e20, min);

	iter->w2_step = tile_width * p->e01.a;
	iter->w0_step = tile_width * p->e12.a;
	iter->w1_step = tile_width * p->e20.a;

	int32_t w = rect->x1 - rect->x0 - tile_width;
	iter->w2_row_step = tile_height * p->e01.b - w * p->e01.a;
	iter->w0_row_step = tile_height * p->e12.b - w * p->e12.a;
	iter->w1_row_step = tile_height * p->e20.b - w * p->e20.a;
}

static bool
bbox_iter_done(struct bbox_iter *iter)
{
	return iter->y == iter->rect.y1;
}

static void
bbox_iter_next(struct bbox_iter *iter)
{
	iter->x += tile_width;
	if (iter->x == iter->rect.x1) {
		iter->x = iter->rect.x0;
		iter->y += tile_height;
		iter->w2 += iter->w2_row_step;
		iter->w0 += iter->w0_row_step;
		iter->w1 += iter->w1_row_step;
	} else {
		iter->w2 += iter->w2_step;
		iter->w0 += iter->w0_step;
		iter->w1 += iter->w1_step;
	}
}

void
rasterize_rectlist(struct ps_primitive *p, struct rectangle *rect)
{
	struct bbox_iter iter;

	for (bbox_iter_init(&iter, p, rect);
	     !bbox_iter_done(&iter); bbox_iter_next(&iter))
		rasterize_rectlist_tile(p, &iter);
}

static int32_t
edge_delta_to_tile_min(struct edge *e)
{
	const int32_t sign_x = (uint32_t) e->a >> 31;
	const int32_t sign_y = (uint32_t) e->b >> 31;

	const int tile_max_x = tile_width - 1;
	const int tile_max_y = tile_height - 1;

	/* This is the delta from w in top-left corner to minimum w in tile. */

	return e->a * sign_x * tile_max_x + e->b * sign_y * tile_max_y;
}

void
rasterize_triangle(struct ps_primitive *p, struct rectangle *rect)
{
	int32_t min_w2_delta = edge_delta_to_tile_min(&p->e01);
	int32_t min_w0_delta = edge_delta_to_tile_min(&p->e12);
	int32_t min_w1_delta = edge_delta_to_tile_min(&p->e20);

	struct bbox_iter iter;
	for (bbox_iter_init(&iter, p, rect);
	     !bbox_iter_done(&iter); bbox_iter_next(&iter)) {
		int32_t min_w2 = iter.w2 + min_w2_delta;
		int32_t min_w0 = iter.w0 + min_w0_delta;
		int32_t min_w1 = iter.w1 + min_w1_delta;

		if ((min_w2 & min_w0 & min_w1) < 0)
			rasterize_triangle_tile(p, &iter);
	}
}

static void
compute_bounding_box(struct rectangle *r, const struct vec4 *v, int count)
{
	r->x0 = INT_MAX;
	r->y0 = INT_MAX;
	r->x1 = INT_MIN;
	r->y1 = INT_MIN;

	for (int i = 0; i < count; i++) {
		int32_t x, y;

		x = floor(v[i].x);
		if (x < r->x0)
			r->x0 = x;
		y = floor(v[i].y);
		if (y < r->y0)
			r->y0 = y;

		x = ceil(v[i].x);
		if (r->x1 < x)
			r->x1 = x;
		y = ceil(v[i].y);
		if (r->y1 < y)
			r->y1 = y;
	}
}

static void
intersect_rectangle(struct rectangle *r, const struct rectangle *other)
{
	if (r->x0 < other->x0)
		r->x0 = other->x0;
	if (r->y0 < other->y0)
		r->y0 = other->y0;
	if (r->x1 > other->x1)
		r->x1 = other->x1;
	if (r->y1 > other->y1)
		r->y1 = other->y1;
}

static void
rewrite_to_rectlist(struct value **vue, struct vec4 *v)
{
	float length, dx, dy, px, py;

	vue[2] = vue[1];
	v[0] = vue[0][1].vec4;
	v[1] = vue[1][1].vec4;
	v[2] = vue[2][1].vec4;

	dx = v[1].x - v[0].x;
	dy = v[1].y - v[0].y;
	length = gt.sf.line_width / 2.0f / hypot(dx, dy);
	dx *= length;
	dy *= length;
	px = -dy;
	py = dx;
	v[0].x = v[0].x - dx - px;
	v[0].y = v[0].y - dy - py;
	v[1].x = v[1].x + dx - px;
	v[1].y = v[1].y + dy - py;
	v[2].x = v[2].x + dx + px;
	v[2].y = v[2].y + dy + py;
}

void
rasterize_primitive(struct value **vue, enum GEN9_3D_Prim_Topo_Type topology)
{
	struct ps_primitive p;
	struct vec4 v[3];

	switch (topology) {
	case _3DPRIM_LINELOOP:
	case _3DPRIM_LINELIST:
	case _3DPRIM_LINESTRIP:
		rewrite_to_rectlist(vue, v);
		break;
	default:
		v[0] = vue[0][1].vec4;
		v[1] = vue[1][1].vec4;
		v[2] = vue[2][1].vec4;
		break;
	}

	struct point p0 = snap_point(v[0].x, v[0].y);
	struct point p1 = snap_point(v[1].x, v[1].y);
	struct point p2 = snap_point(v[2].x, v[2].y);

	init_edge(&p.e01, p0, p1);
	init_edge(&p.e12, p1, p2);
	init_edge(&p.e20, p2, p0);
	p.area = eval_edge(&p.e01, p2);

	if ((gt.wm.front_winding == CounterClockwise &&
	     gt.wm.cull_mode == CULLMODE_FRONT) ||
	    (gt.wm.front_winding == Clockwise &&
	     gt.wm.cull_mode == CULLMODE_BACK) ||
	    (gt.wm.cull_mode == CULLMODE_NONE && p.area > 0)) {
		invert_edge(&p.e01);
		invert_edge(&p.e12);
		invert_edge(&p.e20);
		p.area = -p.area;
	}

	if (p.area >= 0)
		return;

	switch (topology) {
	case _3DPRIM_LINELOOP:
	case _3DPRIM_LINELIST:
	case _3DPRIM_LINESTRIP:
		break;
	default:
		if (gt.wm.front_face_fill_mode != FILL_MODE_WIREFRAME)
			break;

		/* Hacky wireframe implementation: turn each triangle
		 * edge into a line and call back into
		 * rasterize_primitive(). */
		struct value *wf_vue[3];
		wf_vue[0] = vue[0];
		wf_vue[1] = vue[1];
		rasterize_primitive(wf_vue, _3DPRIM_LINELIST);
		wf_vue[0] = vue[1];
		wf_vue[1] = vue[2];
		rasterize_primitive(wf_vue, _3DPRIM_LINELIST);
		wf_vue[0] = vue[2];
		wf_vue[1] = vue[0];
		rasterize_primitive(wf_vue, _3DPRIM_LINELIST);
		return;
	}

	float w[3] = {
		1.0f / v[0].z,
		1.0f / v[1].z,
		1.0f / v[2].z
	};

	p.w_deltas[0] = w[1] - w[0];
	p.w_deltas[1] = w[2] - w[0];
	p.w_deltas[2] = 0.0f;
	p.w_deltas[3] = w[0];

	for (uint32_t i = 0; i < gt.sbe.num_attributes; i++) {
		uint32_t read_index;
		if (gt.sbe.swiz_enable)
			read_index = gt.sbe.read_offset * 2 + gt.sbe.swiz[i];
		else
			read_index = gt.sbe.read_offset * 2 + i;

		const struct value a0 = vue[0][read_index];
		const struct value a1 = vue[1][read_index];
		const struct value a2 = vue[2][read_index];

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

	struct rectangle rect;
	compute_bounding_box(&rect, v, 3);
	intersect_rectangle(&rect, &gt.drawing_rectangle.rect);

	if (gt.wm.scissor_rectangle_enable)
		intersect_rectangle(&rect, &gt.wm.scissor_rect);

	rect.x0 = rect.x0 & ~(tile_width - 1);
	rect.y0 = rect.y0 & ~(tile_height - 1);
	rect.x1 = (rect.x1 + tile_width - 1) & ~(tile_width - 1);
	rect.y1 = (rect.y1 + tile_height - 1) & ~(tile_height - 1);

	if (rect.x1 <= rect.x0 || rect.y1 < rect.y0)
		return;

	const uint32_t dx = 4;
	const uint32_t dy = 2;
	static const struct reg sx = { .d = {  0, 1, 0, 1, 2, 3, 2, 3 } };
	static const struct reg sy = { .d = {  0, 0, 1, 1, 0, 0, 1, 1 } };

	p.w2_offsets =
		_mm256_mullo_epi32(_mm256_set1_epi32(p.e01.a), sx.ireg) +
		_mm256_mullo_epi32(_mm256_set1_epi32(p.e01.b), sy.ireg);
	p.w0_offsets =
		_mm256_mullo_epi32(_mm256_set1_epi32(p.e12.a), sx.ireg) +
		_mm256_mullo_epi32(_mm256_set1_epi32(p.e12.b), sy.ireg);
	p.w1_offsets =
		_mm256_mullo_epi32(_mm256_set1_epi32(p.e20.a), sx.ireg) +
		_mm256_mullo_epi32(_mm256_set1_epi32(p.e20.b), sy.ireg);

	p.w2_step = _mm256_set1_epi32(p.e01.a * dx);
	p.w0_step = _mm256_set1_epi32(p.e12.a * dx);
	p.w1_step = _mm256_set1_epi32(p.e20.a * dx);

	p.w2_row_step = _mm256_set1_epi32(p.e01.b * dy - p.e01.a * (tile_width - dx));
	p.w0_row_step = _mm256_set1_epi32(p.e12.b * dy - p.e12.a * (tile_width - dx));
	p.w1_row_step = _mm256_set1_epi32(p.e20.b * dy - p.e20.a * (tile_width - dx));

	switch (topology) {
	case _3DPRIM_RECTLIST:
	case _3DPRIM_LINELOOP:
	case _3DPRIM_LINELIST:
	case _3DPRIM_LINESTRIP:
		rasterize_rectlist(&p, &rect);
		break;
	default:
		rasterize_triangle(&p, &rect);
	}
}

void
wm_flush(void)
{
	if (framebuffer_filename) {
		struct surface s;
		get_surface(gt.ps.binding_table_address, 0, &s);
		dump_surface(framebuffer_filename, &s);
	}
}

void
depth_clear(void)
{
	uint64_t range;
	void *depth;
	struct reg clear_value;
	int i;

	if (gt.depth.hiz_enable) {
		uint32_t tile_stride = DIV_ROUND_UP(gt.depth.width, 32);
		uint32_t tile_height = DIV_ROUND_UP(gt.depth.height, 32);
		uint32_t size = tile_stride * tile_height;

		void *hiz = map_gtt_offset(gt.depth.hiz_address, &range);
		memset(hiz, 0, size);
		return;
	}

	switch (gt.depth.format) {
	case D32_FLOAT:
		clear_value.reg = _mm256_set1_ps(gt.depth.clear_value);
		break;
	case D24_UNORM_X8_UINT:
		clear_value.ireg = _mm256_set1_epi32(gt.depth.clear_value * 16777215.0f);
		break;
	case D16_UNORM:
		clear_value.ireg = _mm256_set1_epi16(gt.depth.clear_value * 65535.0f);
		break;
	default:
		ksim_unreachable("invalid depth format");
	}

	depth = map_gtt_offset(gt.depth.address, &range);
	int height = (gt.depth.height + 31) & ~31;

	for (i = 0; i < gt.depth.stride * height; i += 32)
		_mm256_store_si256((depth + i), clear_value.ireg);
}

#define NO_KERNEL 1

static void
emit_load_attributes_deltas(struct kir_program *prog, int g)
{
	kir_program_comment(prog, "load attribute deltas");
	for (uint32_t i = 0; i < gt.sbe.num_attributes * 2; i++) {
		kir_program_load_v8(prog, offsetof(struct ps_thread, attribute_deltas[i]));
		kir_program_store_v8(prog, offsetof(struct thread, grf[g++]), prog->dst);
	}
}

static void
emit_load_payload(struct kir_program *prog, int width)
{
	int g = 2;

	kir_program_load_v8(prog, offsetof(struct ps_thread, grf0));
	kir_program_store_v8(prog, offsetof(struct thread, grf[0]), prog->dst);

	if (gt.wm.barycentric_mode)
		kir_program_comment(prog, "load payload: barycentric coordinates");
	for (uint32_t i = 0; i < 6; i++) {
		if (gt.wm.barycentric_mode & (1 << i)) {
			kir_program_load_v8(prog, offsetof(struct ps_thread, queue[0].w1_pc));
			kir_program_store_v8(prog, offsetof(struct thread, grf[g++]), prog->dst);
			kir_program_load_v8(prog, offsetof(struct ps_thread, queue[0].w2_pc));
			kir_program_store_v8(prog, offsetof(struct thread, grf[g++]), prog->dst);

			if (width == 16) {
				kir_program_load_v8(prog, offsetof(struct ps_thread, queue[1].w1_pc));
				kir_program_store_v8(prog, offsetof(struct thread, grf[g++]), prog->dst);
				kir_program_load_v8(prog, offsetof(struct ps_thread, queue[1].w2_pc));
				kir_program_store_v8(prog, offsetof(struct thread, grf[g++]), prog->dst);
			}
		}
	}

	if (gt.ps.uses_source_depth) {
		kir_program_comment(prog, "load payload: source depth");
		kir_program_load_v8(prog, offsetof(struct ps_thread, queue[0].z));
		kir_program_store_v8(prog, offsetof(struct thread, grf[g++]), prog->dst);
	}

	if (gt.ps.uses_source_w) {
		kir_program_comment(prog, "load payload: source w");
		kir_program_load_v8(prog, offsetof(struct ps_thread, queue[0].w));
		kir_program_store_v8(prog, offsetof(struct thread, grf[g++]), prog->dst);
	}

	if (gt.ps.position_offset_xy == POSOFFSET_CENTROID) {
		kir_program_comment(prog, "load payload: POSOFFSET_CENTROID stub");
		g++;
	} else if (gt.ps.position_offset_xy == POSOFFSET_SAMPLE) {
		kir_program_comment(prog, "load payload: POSOFFSET_SAMPLE stub");
		g++;
	}

	if (gt.ps.input_coverage_mask_state != ICMS_NONE) {
		kir_program_comment(prog, "load payload: coverage mask stub");
		g++;
	}
}

static shader_t
compile_ps_for_width(uint64_t kernel_offset, int width)
{
	struct kir_program prog;

	kir_program_init(&prog, gt.ps.binding_table_address,
			 gt.ps.sampler_state_address);

	emit_barycentric_conversion(&prog);

	emit_depth_test(&prog);

	if (gt.ps.enable) {
		emit_load_payload(&prog, width);

		int g;
		if (gt.ps.push_constant_enable)
			g = emit_load_constants(&prog, &gt.ps.curbe, gt.ps.grf_start0);
		else
			g = gt.ps.grf_start0;

		if (gt.ps.attribute_enable)
			emit_load_attributes_deltas(&prog, g);

		kir_program_comment(&prog, "eu ps");
		kir_program_emit_shader(&prog, kernel_offset);
	}

	kir_program_add_insn(&prog, kir_eot);

	return kir_program_finish(&prog);
}

void
compile_ps(void)
{
	uint64_t ksp_simd8 = NO_KERNEL, ksp_simd16 = NO_KERNEL, ksp_simd32 = NO_KERNEL;

	if (!gt.ps.enable)
		return;

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
			if (gt.ps.enable_simd32) {
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
			compile_ps_for_width(ksp_simd8, 8);
	}
	if (ksp_simd16 != NO_KERNEL) {
		ksim_trace(TRACE_EU | TRACE_AVX, "jit simd16 ps\n");
		gt.ps.avx_shader_simd16 =
			compile_ps_for_width(ksp_simd16, 16);
	}
	if (ksp_simd32 != NO_KERNEL) {
		ksim_trace(TRACE_EU | TRACE_AVX, "jit simd32 ps\n");
		gt.ps.avx_shader_simd32 =
			compile_ps_for_width(ksp_simd32, 32);
	}
}
