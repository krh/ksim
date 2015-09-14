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
	struct { float a, b, c; } w, red, green, blue;
	int area;
	float inv_area;
	int w2[8], w0[8], w1[8];
	int w2_offsets[8], w0_offsets[8];
	uint32_t pixels[8];
	int a01, b01, c01;
	int a12, b12, c12;
	int a20, b20, c20;

	int max_w0_delta, max_w1_delta, max_w2_delta;
	int max_group_w0_delta, max_group_w1_delta, max_group_w2_delta;

	int offsets[8];
};

static void
dispatch_ps(struct payload *p)
{
	struct thread t;
	uint32_t g;

	assert(gt.ps.simd8);

	/* Not sure what we should make this. */
	uint32_t fftid = 0;

	t.mask = 0xff;
	/* Fixed function header */
	t.grf[0] = (struct reg) {
		.ud = {
			/* R0.0 - R0.2: MBZ */
			0,
			0,
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

	g = load_constants(&t, &gt.ps.curbe, gt.ps.urb_start_grf);

	/* SIMD8 PS payload */

	if (gt.ps.statistics)
		gt.ps_invocation_count++;

	run_thread(&t, gt.ps.ksp, TRACE_PS);
}

static void
fragment_shader(struct payload *p)
{
	for (int i = 0; i < 8; i++) {
		float w0 = (float) p->w0[i] * p->inv_area;
		float w2 = (float) p->w2[i] * p->inv_area;
		int red = p->red.b * w0 + p->red.a * w2 + p->red.c;
		int green = p->green.b * w0 + p->green.a * w2 + p->green.c;
		int blue = p->blue.b * w0 + p->blue.a * w2 + p->blue.c;

		p->pixels[i] = (red << 16) | (green << 8) | (blue);
	}
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

			fragment_shader(p);

			void *base = tile + (y * stride) + x * cpp;
			for (int i = 0; i < 8; i++) {
				uint32_t *px = base + p->offsets[i];
				if ((p->w1[i] | p->w0[i] | p->w2[i]) >= 0)
					*px = p->pixels[i];
			}

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

struct rt {
	const uint32_t *state;
	void *pixels;
	int width;
	int height;
	int stride;
	int cpp;
};

static uint32_t *
get_render_target(int i, struct rt *rt)
{
	uint64_t offset, range;
	const uint32_t *binding_table;

	binding_table = map_gtt_offset(gt.ps.binding_table_address +
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
rasterize_primitive(struct primitive *prim)
{
	struct rt rt;

	if (!get_render_target(0, &rt)) {
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

	p.a01 = (y0 - y1);
	p.b01 = (x1 - x0);
	p.c01 = (x0 * y1 - y0 * x1);

	p.a12 = (y1 - y2);
	p.b12 = (x2 - x1);
	p.c12 = (x1 * y2 - y1 * x2);

	p.a20 = (y2 - y0);
	p.b20 = (x0 - x2);
	p.c20 = (x2 * y0 - y2 * x0);

	p.area = p.a01 * x2 + p.b01 * y2 + p.c01;
	p.inv_area = 1.0f / p.area;

	float w0 = 1.0f;
	float red0 = 255.0f;
	float blue0 = 0.0;
	float green0 = 0.0f;

	float w1 = 1.0f;
	float red1 = 0.0f;
	float blue1 = 255.0f;
	float green1 = 0.0f;

	float w2 = 1.0f / 2.0f;
	float red2 = 0.0f;
	float blue2 = 0.0f;
	float green2 = 255.0f;

	p.w.a = w0 - w1;
	p.w.b = w2 - w1;
	p.w.c = w1;

	p.red.a = red0 - red1;
	p.red.b = red2 - red1;
	p.red.c = red1;

	p.green.a = green0 - green1;
	p.green.b = green2 - green1;
	p.green.c = green1;

	p.blue.a = blue0 - blue1;
	p.blue.b = blue2 - blue1;
	p.blue.c = blue1;

	const int max_x = 128 / 4 - 1;
	const int max_y = 31;
	const int group_max_x = 3;
	const int group_max_y = 1;

	int w2_max_x = p.a01 > 0 ? 1 : 0;
	int w2_max_y = p.b01 > 0 ? 1 : 0;

	/* delta from w2 in top-left corner to maximum w2 in tile */
	p.max_w2_delta = p.a01 * w2_max_x * max_x + p.b01 * w2_max_y * max_y;
	p.max_group_w2_delta =
		p.a01 * w2_max_x * group_max_x + p.b01 * w2_max_y * group_max_y;

	int w0_max_x = p.a12 > 0 ? 1: 0;
	int w0_max_y = p.b12 > 0 ? 1 : 0;

	/* delta from w2 in top-left corner to maximum w2 in tile */
	p.max_w0_delta = p.a12 * w0_max_x * max_x + p.b12 * w0_max_y * max_y;
	p.max_group_w0_delta =
		p.a12 * w0_max_x * group_max_x + p.b12 * w0_max_y * group_max_y;

	int w1_max_x = p.a20 > 0 ? 1 : 0;
	int w1_max_y = p.b20 > 0 ? 1 : 0;

	/* delta from w2 in top-left corner to maximum w2 in tile */
	p.max_w1_delta = p.a20 * w1_max_x * max_x + p.b20 * w1_max_y * max_y;
	p.max_group_w1_delta =
		p.a20 * w1_max_x * group_max_x + p.b20 * w1_max_y * group_max_y;

	for (int i = 0; i < 8; i++) {
		int sx = (i & 1) + ((i & ~3) >> 1);
		int sy = (i & 2) >> 1;
		p.w2_offsets[i] = p.a01 * sx + p.b01 * sy;
		p.w0_offsets[i] = p.a12 * sx + p.b12 * sy;
		p.offsets[i] = sy * rt.stride + sx * rt.cpp;
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

	if (framebuffer_filename && get_render_target(0, &rt))
		write_png(framebuffer_filename,
			  rt.width, rt.height, rt.stride, rt.pixels);
}
