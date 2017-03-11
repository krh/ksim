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
	int32_t min_x, min_y;
};

struct dispatch {
	struct reg w, z;
	struct reg w2, w1;
	struct reg w2_pc, w1_pc;
	struct reg mask;
	int x, y;
};

struct ps_thread {
	struct thread t;
	struct reg attribute_deltas[64];
	struct dispatch queue[2];

	void *depth;

	float w_deltas[4];
	int queue_length;

	int x0, y0;
	int32_t start_w2, start_w0, start_w1;
	int32_t area;
	float inv_area;
	struct edge e01, e12, e20;

	struct rectangle rect;
	int32_t row_w2, row_w0, row_w1;

	__m256i w2_step, w0_step, w1_step;
	__m256i w2_row_step, w0_row_step, w1_row_step;

	float inv_z1, inv_z2;
};

static void
emit_depth_test(struct kir_program *prog)
{
	struct kir_reg base, depth;

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
	default:
		ksim_unreachable("invalid depth format");
	}

	/* Swizzle two middle pixel pairs so that dword 0-3 and 4-7
	 * match the shader dispatch subspan ordering. */
	// d_f.ireg = _mm256_permute4x64_epi64(d_f.ireg, SWIZZLE(0, 2, 1, 3));

	struct kir_reg computed_depth =
		kir_program_load_v8(prog, offsetof(struct ps_thread, queue[0].w));
	struct kir_reg mask =
		kir_program_load_v8(prog, offsetof(struct ps_thread, t.mask_q1));

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

		kir_program_alu(prog, kir_cmp, computed_depth, depth,
				gen_function_to_avx2[gt.depth.test_function]);
		mask = kir_program_alu(prog, kir_and, mask, prog->dst);
		kir_program_store_v8(prog, offsetof(struct ps_thread, t.mask_q1), mask);
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
	uint32_t fftid = 0;
	struct reg *grf = &t->t.grf[0];

	t->t.mask_q1 = d[0].mask.ireg;
	if (count == 2)
		t->t.mask_q2 = d[1].mask.ireg;
	else
		t->t.mask_q2 = _mm256_set1_epi32(0);

	/* Fixed function header */
	grf[0] = (struct reg) {
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

	uint32_t mask = _mm256_movemask_ps((__m256) t->t.mask_q1);
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

	if (gt.ps.statistics)
		gt.ps_invocation_count++;

	if (count == 1 && gt.ps.enable_simd8) {
		gt.ps.avx_shader_simd8(&t->t);
	} else {
		ksim_assert(gt.ps.enable_simd16);
		gt.ps.avx_shader_simd16(&t->t);
	}
}

const int tile_width = 512 / 4;
const int tile_height = 8;

struct tile_iterator {
	int x, y;
	__m256i w2, w0, w1;
};

static void
tile_iterator_init(struct tile_iterator *iter, struct ps_thread *pt)
{
	__m256i w2_offsets, w0_offsets, w1_offsets;
	static const struct reg sx = { .d = {  0, 1, 0, 1, 2, 3, 2, 3 } };
	static const struct reg sy = { .d = {  0, 0, 1, 1, 0, 0, 1, 1 } };

	iter->x = 0;
	iter->y = 0;

	w2_offsets =
		_mm256_mullo_epi32(_mm256_set1_epi32(pt->e01.a), sx.ireg) +
		_mm256_mullo_epi32(_mm256_set1_epi32(pt->e01.b), sy.ireg);
	w0_offsets =
		_mm256_mullo_epi32(_mm256_set1_epi32(pt->e12.a), sx.ireg) +
		_mm256_mullo_epi32(_mm256_set1_epi32(pt->e12.b), sy.ireg);
	w1_offsets =
		_mm256_mullo_epi32(_mm256_set1_epi32(pt->e20.a), sx.ireg) +
		_mm256_mullo_epi32(_mm256_set1_epi32(pt->e20.b), sy.ireg);

	iter->w2 = _mm256_add_epi32(_mm256_set1_epi32(pt->start_w2),
				    w2_offsets);

	iter->w0 = _mm256_add_epi32(_mm256_set1_epi32(pt->start_w0),
				    w0_offsets);

	iter->w1 = _mm256_add_epi32(_mm256_set1_epi32(pt->start_w1),
				    w1_offsets);
}

static bool
tile_iterator_done(struct tile_iterator *iter)
{
	return iter->y == tile_height;
}

static void
tile_iterator_next(struct tile_iterator *iter, struct ps_thread *pt)
{
	iter->x += 4;
	if (iter->x == tile_width) {
		iter->x = 0;
		iter->y += 2;

		iter->w2 = _mm256_add_epi32(iter->w2, pt->w2_row_step);
		iter->w0 = _mm256_add_epi32(iter->w0, pt->w0_row_step);
		iter->w1 = _mm256_add_epi32(iter->w1, pt->w1_row_step);
	} else {
		iter->w2 = _mm256_add_epi32(iter->w2, pt->w2_step);
		iter->w0 = _mm256_add_epi32(iter->w0, pt->w0_step);
		iter->w1 = _mm256_add_epi32(iter->w1, pt->w1_step);
	}

}

static void
fill_dispatch(struct ps_thread *pt,
	      struct tile_iterator *iter, struct reg mask)
{
	struct dispatch *d = &pt->queue[pt->queue_length];

	if (_mm256_movemask_ps(mask.reg) == 0)
		return;

	/* Some pixels are covered and we have to calculate
	 * barycentric coordinates. We add back the tie-breaker
	 * adjustment so as to not distort the barycentric
	 * coordinates.*/
	d->w2.reg =
		_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(iter->w2, _mm256_set1_epi32(pt->e01.bias))),
			      _mm256_set1_ps(pt->inv_area));
	d->w1.reg =
		_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(iter->w1, _mm256_set1_epi32(pt->e20.bias))),
			      _mm256_set1_ps(pt->inv_area));

	d->w.reg = _mm256_fmadd_ps(_mm256_set1_ps(pt->w_deltas[0]), d->w1.reg,
				   _mm256_fmadd_ps(_mm256_set1_ps(pt->w_deltas[1]), d->w2.reg,
						   _mm256_set1_ps(pt->w_deltas[3])));

	d->z.reg = _mm256_rcp_ps(d->w.reg);
#if 0
	d->w1_pc.reg = _mm256_mul_ps(_mm256_mul_ps(d->z.reg, d->w1.reg),
				     _mm256_set1_ps(pt->inv_z1));
	d->w2_pc.reg = _mm256_mul_ps(_mm256_mul_ps(d->z.reg, d->w2.reg),
				     _mm256_set1_ps(pt->inv_z2));
#else
	d->w1_pc.reg = d->w1.reg;
	d->w2_pc.reg = d->w2.reg;
#endif
	
	d->mask = mask;
	d->x = pt->x0 + iter->x;
	d->y = pt->y0 + iter->y;

	if (gt.depth.write_enable || gt.depth.test_enable) {
		uint64_t range;
		void *buffer = map_gtt_offset(gt.depth.address, &range);
		uint32_t cpp = depth_format_size(gt.depth.format);
		pt->depth = ymajor_offset(buffer, d->x, d->y, gt.depth.stride, cpp);
	}

	pt->queue_length++;
	if (gt.ps.enable_simd8 || pt->queue_length == 2) {
		dispatch_ps(pt);
		pt->queue_length = 0;
	}
}

static void
rasterize_rectlist_tile(struct ps_thread *pt)
{
	struct tile_iterator iter;

	for (tile_iterator_init(&iter, pt);
	     !tile_iterator_done(&iter);
	     tile_iterator_next(&iter, pt)) {
		__m256i w2, w3, c;

		/* To determine coverage, we compute the edge function
		 * for all edges in the rectangle. We only have two of
		 * the four edges, but we can compute the edge
		 * function from the opposite edge by subtracting from
		 * the area. We also subtract 1 to either cancel out
		 * the bias on the original edge, or to add it to the
		 * opposite edge if the original doesn't have bias. */
		c = _mm256_set1_epi32(pt->area - 1);
		w2 = _mm256_sub_epi32(c, iter.w2);
		w3 = _mm256_sub_epi32(c, iter.w0);

		struct reg mask;
		mask.ireg = _mm256_and_si256(_mm256_and_si256(iter.w2, iter.w0),
					     _mm256_and_si256(w2, w3));

		fill_dispatch(pt, &iter, mask);
	}

	if (pt->queue_length > 0) {
		dispatch_ps(pt);
		pt->queue_length = 0;
	}
}

static void
rasterize_triangle_tile(struct ps_thread *pt)
{
	struct tile_iterator iter;

	for (tile_iterator_init(&iter, pt);
	     !tile_iterator_done(&iter);
	     tile_iterator_next(&iter, pt)) {
		struct reg mask;
		mask.ireg =
			_mm256_and_si256(_mm256_and_si256(iter.w1,
							  iter.w0), iter.w2);

		fill_dispatch(pt, &iter, mask);
	}

	if (pt->queue_length > 0) {
		dispatch_ps(pt);
		pt->queue_length = 0;
	}
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
	e->min_x = e->a > 0 ? 0 : 1;
	e->min_y = e->b > 0 ? 0 : 1;
}

static inline void
invert_edge(struct edge *e)
{
	e->a = -e->a;
	e->b = -e->b;
	e->c = -e->c;
	e->min_x = 1 - e->min_x;
	e->min_y = 1 - e->min_y;
	e->bias = 1 - e->bias;
}

static inline int
eval_edge(struct edge *e, struct point p)
{
	return (((int64_t) e->a * p.x + (int64_t) e->b * p.y) >> 8) + e->c - e->bias;
}

static void
bbox_iter_init(struct ps_thread *pt)
{
	pt->x0 = pt->rect.x0;
	pt->y0 = pt->rect.y0;

	pt->start_w2 = pt->row_w2;
	pt->start_w0 = pt->row_w0;
	pt->start_w1 = pt->row_w1;
}

static bool
bbox_iter_done(struct ps_thread *pt)
{
	return pt->y0 == pt->rect.y1;
}

static void
bbox_iter_next(struct ps_thread *pt)
{
	pt->x0 += tile_width;
	if (pt->x0 == pt->rect.x1) {
		pt->x0 = pt->rect.x0;
		pt->y0 += tile_height;
		pt->row_w2 += tile_height * pt->e01.b;
		pt->row_w0 += tile_height * pt->e12.b;
		pt->row_w1 += tile_height * pt->e20.b;
		pt->start_w2 = pt->row_w2;
		pt->start_w0 = pt->row_w0;
		pt->start_w1 = pt->row_w1;
	} else {
		pt->start_w2 += tile_width * pt->e01.a;
		pt->start_w0 += tile_width * pt->e12.a;
		pt->start_w1 += tile_width * pt->e20.a;
	}
}

void
rasterize_rectlist(struct ps_thread *pt)
{
	for (bbox_iter_init(pt); !bbox_iter_done(pt); bbox_iter_next(pt))
		rasterize_rectlist_tile(pt);
}


void
rasterize_triangle(struct ps_thread *pt)
{
	int min_w0_delta, min_w1_delta, min_w2_delta;

	const int tile_max_x = tile_width - 1;
	const int tile_max_y = tile_height - 1;

	/* delta from w in top-left corner to minimum w in tile */
	min_w2_delta = pt->e01.a * pt->e01.min_x * tile_max_x + pt->e01.b * pt->e01.min_y * tile_max_y;
	min_w0_delta = pt->e12.a * pt->e12.min_x * tile_max_x + pt->e12.b * pt->e12.min_y * tile_max_y;
	min_w1_delta = pt->e20.a * pt->e20.min_x * tile_max_x + pt->e20.b * pt->e20.min_y * tile_max_y;

	for (bbox_iter_init(pt); !bbox_iter_done(pt); bbox_iter_next(pt)) {
		int32_t min_w2 = pt->start_w2 + min_w2_delta;
		int32_t min_w0 = pt->start_w0 + min_w0_delta;
		int32_t min_w1 = pt->start_w1 + min_w1_delta;

		if ((min_w2 & min_w0 & min_w1) < 0)
			rasterize_triangle_tile(pt);
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
	struct ps_thread pt;
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

	init_edge(&pt.e01, p0, p1);
	init_edge(&pt.e12, p1, p2);
	init_edge(&pt.e20, p2, p0);
	pt.area = eval_edge(&pt.e01, p2);

	if ((gt.wm.front_winding == CounterClockwise &&
	     gt.wm.cull_mode == CULLMODE_FRONT) ||
	    (gt.wm.front_winding == Clockwise &&
	     gt.wm.cull_mode == CULLMODE_BACK) ||
	    (gt.wm.cull_mode == CULLMODE_NONE && pt.area > 0)) {
		invert_edge(&pt.e01);
		invert_edge(&pt.e12);
		invert_edge(&pt.e20);
		pt.area = -pt.area;
	}

	if (pt.area >= 0)
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

	pt.inv_area = 1.0f / pt.area;

	float w[3] = {
		1.0f / v[0].z,
		1.0f / v[1].z,
		1.0f / v[2].z
	};
	pt.inv_z1 = 1.0 / v[1].z;
	pt.inv_z2 = 1.0 / v[2].z;

	pt.w_deltas[0] = w[1] - w[0];
	pt.w_deltas[1] = w[2] - w[0];
	pt.w_deltas[2] = 0.0f;
	pt.w_deltas[3] = w[0];

	for (uint32_t i = 0; i < gt.sbe.num_attributes; i++) {
		const struct value a0 = vue[0][i + 2];
		const struct value a1 = vue[1][i + 2];
		const struct value a2 = vue[2][i + 2];

		pt.attribute_deltas[i * 2] = (struct reg) {
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
		pt.attribute_deltas[i * 2 + 1] = (struct reg) {
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

	compute_bounding_box(&pt.rect, v, 3);
	intersect_rectangle(&pt.rect, &gt.drawing_rectangle.rect);

	if (gt.wm.scissor_rectangle_enable)
		intersect_rectangle(&pt.rect, &gt.wm.scissor_rect);

	pt.rect.x0 = pt.rect.x0 & ~(tile_width - 1);
	pt.rect.y0 = pt.rect.y0 & ~(tile_height - 1);
	pt.rect.x1 = (pt.rect.x1 + tile_width - 1) & ~(tile_width - 1);
	pt.rect.y1 = (pt.rect.y1 + tile_height - 1) & ~(tile_height - 1);

	if (pt.rect.x1 <= pt.rect.x0 || pt.rect.y1 < pt.rect.y0)
		return;

	struct point min = snap_point(pt.rect.x0, pt.rect.y0);
	min.x += 128;
	min.y += 128;
	pt.row_w2 = eval_edge(&pt.e01, min);
	pt.row_w0 = eval_edge(&pt.e12, min);
	pt.row_w1 = eval_edge(&pt.e20, min);

	pt.queue_length = 0;

	const uint32_t dx = 4;
	const uint32_t dy = 2;

	pt.w2_step = _mm256_set1_epi32(pt.e01.a * dx);
	pt.w0_step = _mm256_set1_epi32(pt.e12.a * dx);
	pt.w1_step = _mm256_set1_epi32(pt.e20.a * dx);

	pt.w2_row_step = _mm256_set1_epi32(pt.e01.b * dy - pt.e01.a * (tile_width - dx));
	pt.w0_row_step = _mm256_set1_epi32(pt.e12.b * dy - pt.e12.a * (tile_width - dx));
	pt.w1_row_step = _mm256_set1_epi32(pt.e20.b * dy - pt.e20.a * (tile_width - dx));

	switch (topology) {
	case _3DPRIM_RECTLIST:
	case _3DPRIM_LINELOOP:
	case _3DPRIM_LINELIST:
	case _3DPRIM_LINESTRIP:
		rasterize_rectlist(&pt);
		break;
	default:
		rasterize_triangle(&pt);
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

	switch (gt.depth.format) {
	case D32_FLOAT:
		clear_value.reg = _mm256_set1_ps(gt.depth.clear_value);
		break;
	case D24_UNORM_X8_UINT:
		clear_value.ireg = _mm256_set1_epi32(gt.depth.clear_value * 16777215.0f);
		break;
	case D16_UNORM:
		stub("D16_UNORM clear");
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
