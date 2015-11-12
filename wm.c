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
#include "write-png.h"

struct payload {
	int x0, y0;
	int start_w2, start_w0, start_w1;
	float inv_area;
	struct reg w2, w0, w1;
	int a01, b01, c01, adjust01;
	int a12, b12, c12, adjust12;
	int a20, b20, c20, adjust20;

	struct {
		struct reg offsets;
		void *buffer;
	} depth;

	float w_deltas[4];
	struct reg attribute_deltas[64];
	struct thread t;
};

struct queue {
	pthread_mutex_t m;
	pthread_cond_t ready_cond, idle_cond;
	struct payload entries[16];
	uint32_t idle_mask;
	uint32_t ready_mask;
};

static void
queue_init(struct queue *q)
{
	pthread_mutex_init(&q->m, NULL);
	pthread_cond_init(&q->ready_cond, NULL);
	pthread_cond_init(&q->idle_cond, NULL);
	q->idle_mask = ((1 << 16) - 1);
}

static void
queue_put(struct queue *q, struct payload *p)
{
	pthread_mutex_lock(&q->m);
	int size;

	while (q->idle_mask == 0)
		pthread_cond_wait(&q->idle_cond, &q->m);
	int entry = __builtin_ffs(q->idle_mask) - 1;

	ksim_trace(TRACE_QUEUE, "queue put %d\n", entry);

	q->idle_mask &= ~(1 << entry);
	q->ready_mask |= (1 << entry);

	size = offsetof(struct payload, attribute_deltas[gt.sbe.num_attributes * 2]);
	memcpy(&q->entries[entry], p, size);

	pthread_cond_signal(&q->ready_cond);

	pthread_mutex_unlock(&q->m);
}

static struct payload *
queue_get(struct queue *q)
{
	pthread_mutex_lock(&q->m);

	while (q->ready_mask == 0) {
		ksim_trace(TRACE_QUEUE,  "queue empty\n");
		pthread_cond_wait(&q->ready_cond, &q->m);
	}

	int entry = __builtin_ffs(q->ready_mask) - 1;

	ksim_trace(TRACE_QUEUE, "queue get %d\n", entry);

	q->ready_mask &= ~(1 << entry);
	struct payload *p = &q->entries[entry];

	pthread_mutex_unlock(&q->m);

	return p;
}

static void
queue_release(struct queue *q, struct payload *p)
{
	pthread_mutex_lock(&q->m);
	int entry = p - q->entries;

	ksim_trace(TRACE_QUEUE, "queue release %d\n", entry);

	q->idle_mask |= (1 << entry);
	pthread_cond_signal(&q->idle_cond);
	pthread_mutex_unlock(&q->m);
}

static void
queue_stall(struct queue *q)
{
	pthread_mutex_lock(&q->m);

	ksim_trace(TRACE_QUEUE, "queue stall ----------------------\n");

	while (q->idle_mask != ((1 << 16) - 1))
		pthread_cond_wait(&q->idle_cond, &q->m);

	pthread_mutex_unlock(&q->m);
}

/* Decode this at jit time and put in constant pool. */

bool
get_surface(uint32_t binding_table_offset, int i, struct surface *s)
{
	uint64_t offset, range;
	const uint32_t *binding_table;
	const uint32_t *state;

	binding_table = map_gtt_offset(binding_table_offset +
				       gt.surface_state_base_address, &range);
	if (range < 4)
		return false;

	state = map_gtt_offset(binding_table[i] +
			       gt.surface_state_base_address, &range);
	if (range < 16 * 4)
		return false;

	s->width = field(state[2], 0, 13) + 1;
	s->height = field(state[2], 16, 29) + 1;
	s->stride = field(state[3], 0, 17) + 1;
	s->format = field(state[0], 18, 26);
	s->cpp = format_size(s->format);

	offset = get_u64(&state[8]);
	s->pixels = map_gtt_offset(offset, &range);
	if (range < s->height * s->stride)
		return false;

	return true;
}

void
sfid_render_cache_rt_write_simd8(struct thread *t,
				 const struct sfid_render_cache_args *args)
{
	int x = t->grf[1].ud[2] & 0xffff;
	int y = t->grf[1].ud[2] >> 16;
	__m256i r, g, b, a, shift;
	struct reg argb;
	__m256 scale;
	scale = _mm256_set1_ps(255.0f);
	struct reg *src = &t->grf[args->src];

	if (x >= args->rt.width || y >= args->rt.height)
		return;

	r = _mm256_cvtps_epi32(_mm256_mul_ps(src[0].reg, scale));
	g = _mm256_cvtps_epi32(_mm256_mul_ps(src[1].reg, scale));
	b = _mm256_cvtps_epi32(_mm256_mul_ps(src[2].reg, scale));
	a = _mm256_cvtps_epi32(_mm256_mul_ps(src[3].reg, scale));

	shift = _mm256_set1_epi32(8);
	argb.ireg = _mm256_sllv_epi32(a, shift);
	argb.ireg = _mm256_or_si256(argb.ireg, r);
	argb.ireg = _mm256_sllv_epi32(argb.ireg, shift);
	argb.ireg = _mm256_or_si256(argb.ireg, g);
	argb.ireg = _mm256_sllv_epi32(argb.ireg, shift);
	argb.ireg = _mm256_or_si256(argb.ireg, b);

#define SWIZZLE(x, y, z, w) \
	( ((x) << 0) | ((y) << 2) | ((z) << 4) | ((w) << 6) )

	/* Swizzle two middle pixel pairs so that dword 0-3 and 4-7
	 * form linear owords of pixels. */
	argb.ireg = _mm256_permute4x64_epi64(argb.ireg, SWIZZLE(0, 2, 1, 3));
	__m256i mask = _mm256_permute4x64_epi64(t->mask_full, SWIZZLE(0, 2, 1, 3));

	const int tile_x = x * args->rt.cpp / 512;
	const int tile_y = y / 8;
	const int tile_stride = args->rt.stride / 512;
	void *tile_base =
		args->rt.pixels + (tile_x + tile_y * tile_stride) * 4096;

	const int ix = x & (512 / args->rt.cpp - 1);
	const int iy = y & 7;
	void *base = tile_base + ix * args->rt.cpp + iy * 512;

	_mm_maskstore_epi32(base,
			    _mm256_extractf128_si256(mask, 0),
			    _mm256_extractf128_si256(argb.ireg, 0));
	_mm_maskstore_epi32(base + 512,
			    _mm256_extractf128_si256(mask, 1),
			    _mm256_extractf128_si256(argb.ireg, 1));
}

static uint32_t
depth_test(struct payload *p, uint32_t mask, int x, int y)
{
	uint32_t cpp = depth_format_size(gt.depth.format);

	if (x >= gt.depth.width || y >= gt.depth.height + 1)
		return 0;

	/* early depth test */
	struct reg w, w_unorm;
	w.reg = _mm256_fmadd_ps(_mm256_set1_ps(p->w_deltas[0]), p->w1.reg,
				_mm256_fmadd_ps(_mm256_set1_ps(p->w_deltas[1]), p->w2.reg,
						_mm256_set1_ps(p->w_deltas[3])));

	struct reg d24x8, cmp, d_f;

	/* Y-tiled depth buffer */
	const int tile_x = x * cpp / 128;
	const int tile_y = y / 32;
	const int tile_stride = gt.depth.stride / 128;
	void *tile_base =
		p->depth.buffer + (tile_x + tile_y * tile_stride) * 4096;

	const int ix = x & (128 / cpp - 1);
	const int iy = y & 31;
	void *base = tile_base + ix * cpp * 32 + iy * 16;

	switch (gt.depth.format) {
	case D32_FLOAT:
		d_f.reg = _mm256_load_ps(base);
		break;
	case D24_UNORM_X8_UINT:
		d24x8.ireg = _mm256_load_si256(base);
		d_f.reg = _mm256_mul_ps(_mm256_cvtepi32_ps(d24x8.ireg),
					_mm256_set1_ps(1.0f / 16777216.0f));
		break;
	case D16_UNORM:
		stub("D16_UNORM");
	}

	/* Swizzle two middle pixel pairs so that dword 0-3 and 4-7
	 * form linear owords of pixels. */
	d_f.ireg = _mm256_permute4x64_epi64(d_f.ireg, SWIZZLE(0, 2, 1, 3));

	if (gt.depth.test_enable) {
		cmp.reg = _mm256_cmp_ps(d_f.reg, w.reg, _CMP_LT_OS);
		cmp.ireg = _mm256_and_si256(cmp.ireg, p->t.mask_full);
		p->t.mask_full = cmp.ireg;
		mask = _mm256_movemask_ps(cmp.reg);
	}

	if (gt.depth.write_enable) {
		w_unorm.ireg = _mm256_cvtps_epi32(_mm256_mul_ps(w.reg, _mm256_set1_ps((1 << 24) - 1)));
		w_unorm.ireg = _mm256_permute4x64_epi64(w_unorm.ireg,
							SWIZZLE(0, 2, 1, 3));
		__m256i mask = _mm256_permute4x64_epi64(p->t.mask_full,
							SWIZZLE(0, 2, 1, 3));
		_mm256_maskstore_epi32(base, mask, w_unorm.ireg);
	}

	return mask;
}

static void
dispatch_ps(struct payload *p, uint32_t mask, int x, int y)
{
	uint32_t g;

	assert(gt.ps.enable_simd8);

	/* Not sure what we should make this. */
	uint32_t fftid = 0;

	p->t.mask = mask;
	/* Fixed function header */
	p->t.grf[0] = (struct reg) {
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

	p->t.grf[1] = (struct reg) {
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
		p->t.grf[g].reg = p->w1.reg;
		p->t.grf[g + 1].reg = p->w2.reg;
		g += 2;
		/* if (simd16) ... */
	}

	if (gt.wm.barycentric_mode & BIM_PERSPECTIVE_CENTROID) {
		p->t.grf[g].reg = p->w1.reg;
		p->t.grf[g + 1].reg = p->w2.reg;
		g += 2;
		/* if (simd16) ... */
	}

	if (gt.wm.barycentric_mode & BIM_PERSPECTIVE_SAMPLE) {
		p->t.grf[g].reg = p->w1.reg;
		p->t.grf[g + 1].reg = p->w2.reg;
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
		g = load_constants(&p->t, &gt.ps.curbe, gt.ps.grf_start0);
	else
		g = gt.ps.grf_start0;

	if (gt.ps.attribute_enable) {
		memcpy(&p->t.grf[g], p->attribute_deltas,
		       gt.sbe.num_attributes * 2 * sizeof(p->t.grf[0]));
	}

	if (gt.ps.statistics)
		gt.ps_invocation_count++;

	void (*f)(struct thread *t) = (void *) gt.ps.avx_shader->code;
	f(&p->t);
}

const int tile_width = 512 / 4;
const int tile_height = 8;

static void
rasterize_tile(struct payload *p)
{
	struct reg row_w2, w2;
	struct reg row_w0, w0;
	struct reg row_w1, w1;

	struct reg w2_offsets, w0_offsets, w1_offsets;
	static const struct reg sx = { .d = {  0, 1, 0, 1, 2, 3, 2, 3 } };
	static const struct reg sy = { .d = {  0, 0, 1, 1, 0, 0, 1, 1 } };

	w2_offsets.ireg =
		_mm256_mullo_epi32(_mm256_set1_epi32(p->a01), sx.ireg) +
		_mm256_mullo_epi32(_mm256_set1_epi32(p->b01), sy.ireg);
	w0_offsets.ireg =
		_mm256_mullo_epi32(_mm256_set1_epi32(p->a12), sx.ireg) +
		_mm256_mullo_epi32(_mm256_set1_epi32(p->b12), sy.ireg);
	w1_offsets.ireg =
		_mm256_mullo_epi32(_mm256_set1_epi32(p->a20), sx.ireg) +
		_mm256_mullo_epi32(_mm256_set1_epi32(p->b20), sy.ireg);

	row_w2.ireg = _mm256_add_epi32(_mm256_set1_epi32(p->start_w2),
				       w2_offsets.ireg);

	row_w0.ireg = _mm256_add_epi32(_mm256_set1_epi32(p->start_w0),
				       w0_offsets.ireg);

	row_w1.ireg = _mm256_add_epi32(_mm256_set1_epi32(p->start_w1),
				       w1_offsets.ireg);


	for (int y = 0; y < tile_height; y += 2) {
		w2.ireg = row_w2.ireg;
		w0.ireg = row_w0.ireg;
		w1.ireg = row_w1.ireg;

		for (int x = 0; x < tile_width; x += 4) {
			struct reg det;
			det.ireg =
				_mm256_and_si256(_mm256_and_si256(w1.ireg,
								  w0.ireg), w2.ireg);

			/* Determine coverage: this is an e < 0 test,
			 * where we've subtracted 1 from top-left
			 * edges to include pixels on those edges. */
			uint32_t mask = _mm256_movemask_ps(det.reg);
			if (mask == 0)
				goto next;

			/* Some pixels are covered and we have to
			 * calculate barycentric coordinates. We add
			 * back the tie-breaker adjustment so as to
			 * not distort the barycentric coordinates.*/
			p->w2.reg =
				_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(w2.ireg, _mm256_set1_epi32(p->adjust01))),
					      _mm256_set1_ps(p->inv_area));
			p->w0.reg =
				_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(w0.ireg, _mm256_set1_epi32(p->adjust12))),
					      _mm256_set1_ps(p->inv_area));
			p->w1.reg =
				_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(w1.ireg, _mm256_set1_epi32(p->adjust20))),
					      _mm256_set1_ps(p->inv_area));

			p->t.mask_full = det.ireg;
			if (gt.depth.test_enable || gt.depth.write_enable)
				mask = depth_test(p, mask, p->x0 + x, p->y0 + y);

			if (mask)
				dispatch_ps(p, mask, p->x0 + x, p->y0 + y);

		next:
			w2.ireg = _mm256_add_epi32(w2.ireg, _mm256_set1_epi32(p->a01 * 4));
			w0.ireg = _mm256_add_epi32(w0.ireg, _mm256_set1_epi32(p->a12 * 4));
			w1.ireg = _mm256_add_epi32(w1.ireg, _mm256_set1_epi32(p->a20 * 4));
		}
		row_w2.ireg = _mm256_add_epi32(row_w2.ireg, _mm256_set1_epi32(p->b01 * 2));
		row_w0.ireg = _mm256_add_epi32(row_w0.ireg, _mm256_set1_epi32(p->b12 * 2));
		row_w1.ireg = _mm256_add_epi32(row_w1.ireg, _mm256_set1_epi32(p->b20 * 2));
	}
}

static bool queue_initialized;
static struct queue queue = { .idle_mask = 0xffff };
static pthread_t threads[4];

static void *
do_work(void *arg)
{
	while (true) {
		struct payload *p = queue_get(&queue);
		rasterize_tile(p);
		queue_release(&queue, p);
	}

	return NULL;
}

void
rasterize_primitive(struct primitive *prim)
{
	const int x0 = prim->v[0].x;
	const int y0 = prim->v[0].y;
	const int x1 = prim->v[1].x;
	const int y1 = prim->v[1].y;
	const int x2 = prim->v[2].x;
	const int y2 = prim->v[2].y;

	if (use_threads && !queue_initialized) {
		queue_init(&queue);
		queue_initialized = true;
		for (int i = 0; i < 4; i++) {
			cpu_set_t set;
			CPU_ZERO(&set);
			CPU_SET(i, &set);
			pthread_create(&threads[i], NULL, do_work, NULL);
			pthread_setaffinity_np(threads[i], sizeof(set), &set);
		}
	}

	struct payload p;

	if ((gt.wm.front_winding == CounterClockwise &&
	     gt.wm.cull_mode == CULLMODE_BACK) ||
	    (gt.wm.front_winding == Clockwise &&
	     gt.wm.cull_mode == CULLMODE_FRONT)) {
		p.a01 = (y0 - y1);
		p.b01 = (x1 - x0);
		p.c01 = (y1 * x0 - x1 * y0);

		p.a12 = (y1 - y2);
		p.b12 = (x2 - x1);
		p.c12 = (y2 * x1 - x2 * y1);

		p.a20 = (y2 - y0);
		p.b20 = (x0 - x2);
		p.c20 = (y0 * x2 - x0 * y2);
	} else {
		p.a01 = (y1 - y0);
		p.b01 = (x0 - x1);
		p.c01 = (x1 * y0 - y1 * x0);

		p.a12 = (y2 - y1);
		p.b12 = (x1 - x2);
		p.c12 = (x2 * y1 - y2 * x1);

		p.a20 = (y0 - y2);
		p.b20 = (x2 - x0);
		p.c20 = (x0 * y2 - y0 * x2);
	}

	int area = p.a01 * x2 + p.b01 * y2 + p.c01;

	if ((gt.wm.cull_mode == CULLMODE_NONE && area > 0)) {
		p.a01 = -p.a01;
		p.b01 = -p.b01;
		p.c01 = -p.c01;

		p.a12 = -p.a12;
		p.b12 = -p.b12;
		p.c12 = -p.c12;

		p.a20 = -p.a20;
		p.b20 = -p.b20;
		p.c20 = -p.c20;
		area = -area;
	}

	if (area >= 0)
		return;
	p.inv_area = 1.0f / area;

	/* Tie breaker adjustments for pixels on edges. */
	p.adjust01 = p.a01 < 0 || (p.a01 == 0 && p.b01 <= 0);
	p.adjust12 = p.a12 < 0 || (p.a12 == 0 && p.b12 <= 0);
	p.adjust20 = p.a20 < 0 || (p.a20 == 0 && p.b20 <= 0);

	float w[3] = {
		1.0f / prim->v[0].z,
		1.0f / prim->v[1].z,
		1.0f / prim->v[2].z
	};
	p.w_deltas[0] = w[1] - w[0];
	p.w_deltas[1] = w[2] - w[0];
	p.w_deltas[2] = 0.0f;
	p.w_deltas[3] = w[0];

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

	static const struct reg sx = { .d = {  0, 1, 0, 1, 2, 3, 2, 3 } };
	static const struct reg sy = { .d = {  0, 0, 1, 1, 0, 0, 1, 1 } };

	if (gt.depth.write_enable || gt.depth.test_enable) {
		uint64_t range;
		uint32_t cpp = depth_format_size(gt.depth.format);

		p.depth.offsets.ireg =
			_mm256_add_epi32(_mm256_mullo_epi32(sx.ireg, _mm256_set1_epi32(cpp)),
					 _mm256_mullo_epi32(sy.ireg, _mm256_set1_epi32(gt.depth.stride)));
		p.depth.buffer = map_gtt_offset(gt.depth.address, &range);
	}

	const int tile_max_x = tile_width - 1;
	const int tile_max_y = tile_height - 1;

	int w2_min_x = p.a01 > 0 ? 0 : 1;
	int w2_min_y = p.b01 > 0 ? 0 : 1;
	int min_w0_delta, min_w1_delta, min_w2_delta;

	/* delta from w2 in top-left corner to minimum w2 in tile */
	min_w2_delta = p.a01 * w2_min_x * tile_max_x + p.b01 * w2_min_y * tile_max_y;

	int w0_min_x = p.a12 > 0 ? 0 : 1;
	int w0_min_y = p.b12 > 0 ? 0 : 1;

	/* delta from w0 in top-left corner to minimum w0 in tile */
	min_w0_delta = p.a12 * w0_min_x * tile_max_x + p.b12 * w0_min_y * tile_max_y;

	int w1_min_x = p.a20 > 0 ? 0 : 1;
	int w1_min_y = p.b20 > 0 ? 0 : 1;

	/* delta from w1 in top-left corner to minumum w1 in tile */
	min_w1_delta = p.a20 * w1_min_x * tile_max_x + p.b20 * w1_min_y * tile_max_y;

	int min_x, min_y, max_x, max_y;

	min_x = INT_MAX;
	min_y = INT_MAX;
	max_x = INT_MIN;
	max_y = INT_MIN;
	for (int i = 0; i < 3; i++) {
		int x, y;

		x = floor(prim->v[i].x);
		if (x < min_x)
			min_x = x;
		y = floor(prim->v[i].y);
		if (y < min_y)
			min_y = y;

		x = ceil(prim->v[i].x);
		if (max_x < x)
			max_x = x;
		y = ceil(prim->v[i].y);
		if (max_y < y)
			max_y = y;
	}

	if (min_x < gt.drawing_rectangle.min_x)
		min_x = gt.drawing_rectangle.min_x;
	if (min_y < gt.drawing_rectangle.min_y)
		min_y = gt.drawing_rectangle.min_y;
	if (max_x > gt.drawing_rectangle.max_x)
		max_x = gt.drawing_rectangle.max_x;
	if (max_y > gt.drawing_rectangle.max_y)
		max_y = gt.drawing_rectangle.max_y;

	min_x = min_x & ~(tile_width - 1);
	min_y = min_y & ~(tile_height - 1);
	max_x = (max_x + tile_width - 1) & ~(tile_width - 1);
	max_y = (max_y + tile_height - 1) & ~(tile_height - 1);

	int row_w2 = p.a01 * min_x + p.b01 * min_y + p.c01 - p.adjust01;
	int row_w0 = p.a12 * min_x + p.b12 * min_y + p.c12 - p.adjust12;
	int row_w1 = p.a20 * min_x + p.b20 * min_y + p.c20 - p.adjust20;
	for (p.y0 = min_y; p.y0 < max_y; p.y0 += tile_height) {
		p.start_w2 = row_w2;
		p.start_w0 = row_w0;
		p.start_w1 = row_w1;

		for (p.x0 = min_x; p.x0 < max_x; p.x0 += tile_width) {
			int min_w2 = p.start_w2 + min_w2_delta;
			int min_w0 = p.start_w0 + min_w0_delta;
			int min_w1 = p.start_w1 + min_w1_delta;

			if ((min_w2 & min_w0 & min_w1) < 0) {
				if (use_threads)
					queue_put(&queue, &p);
				else
					rasterize_tile(&p);
			}

			p.start_w2 += tile_width * p.a01;
			p.start_w0 += tile_width * p.a12;
			p.start_w1 += tile_width * p.a20;
		}

		row_w2 += tile_height * p.b01;
		row_w0 += tile_height * p.b12;
		row_w1 += tile_height * p.b20;
	}
}

void
wm_stall(void)
{
	queue_stall(&queue);
}

void
wm_flush(void)
{
	struct surface rt;

	wm_stall();

	if (framebuffer_filename &&
	    get_surface(gt.ps.binding_table_address, 0, &rt))
		write_png(framebuffer_filename,
			  rt.width, rt.height, rt.stride, rt.pixels);
}

void
wm_clear(void)
{
	struct surface rt;
	void *depth;
	uint64_t range;

	if (!gt.ps.resolve &&
	    get_surface(gt.ps.binding_table_address, 0, &rt)) {
		memset(rt.pixels, 0, rt.height * rt.stride);
		if (gt.depth.write_enable) {
			depth = map_gtt_offset(gt.depth.address, &range);
			memset(depth, 0, gt.depth.stride * gt.depth.height);
		}
	}
}

void
hiz_clear(void)
{
	uint64_t range;
	void *depth;

	if (!gt.depth.write_enable)
		return;

	depth = map_gtt_offset(gt.depth.address, &range);
	memset(depth, 0, gt.depth.stride * gt.depth.height);
}
