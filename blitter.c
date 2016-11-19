/*
 * Copyright © 2016 Kristian H. Kristensen
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

#include <string.h>

#include "eu.h"

void blitter_copy(struct blit *b)
{
	if (b->raster_op != 0xcc) {
		stub("raster op 0x%02x\n", b->raster_op);
		return;
	}

	uint64_t range;
	void *dst = map_gtt_offset(b->dst_offset, &range);
	void *src = map_gtt_offset(b->src_offset, &range);

	int32_t stride = b->src_pitch * 4;
	int32_t height = b->dst_y1 - b->dst_y0;

	ksim_assert(b->dst_x0 == 0 && b->dst_y0 == 0);
	ksim_assert(b->src_x == 0 && b->src_y == 0);
	ksim_assert(b->src_pitch == b->dst_pitch);
	ksim_assert(b->src_tile_mode == b->dst_tile_mode);

	memcpy(dst, src, stride * height);
}
