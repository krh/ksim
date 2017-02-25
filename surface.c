/*
 * Copyright © 2017 Kristian H. Kristensen
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

#include <libpng16/png.h>
#include "ksim.h"

bool
get_surface(uint32_t binding_table_offset, int i, struct surface *s)
{
	uint64_t range;
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

	struct GEN9_RENDER_SURFACE_STATE v;
	GEN9_RENDER_SURFACE_STATE_unpack(state, &v);

	s->type = v.SurfaceType;
	s->width = v.Width + 1;
	s->height = v.Height + 1;
	s->stride = v.SurfacePitch + 1;
	s->format = v.SurfaceFormat;
	s->cpp = format_size(s->format);
	s->tile_mode = v.TileMode;
	s->qpitch = v.SurfaceQPitch << 2;
	s->minimum_array_element = v.MinimumArrayElement;
	s->pixels = map_gtt_offset(v.SurfaceBaseAddress, &range);

	const uint32_t block_size = format_block_size(s->format);
	const uint32_t height_in_blocks = DIV_ROUND_UP(s->height, block_size);

	if (range < height_in_blocks * s->stride) {
		ksim_warn("surface state out-of-range for bo\n");
		return false;
	}

	return true;
}

static char *
detile_xmajor(struct surface *s, __m256i alpha)
{
	int height = align_u64(s->height, 8);
	void *pixels;
	int tile_stride = s->stride / 512;
	int ret;

	ret = posix_memalign(&pixels, 32, s->stride * height);
	ksim_assert(ret == 0);

	ksim_assert((s->stride & 511) == 0);

	for (int y = 0; y < height; y++) {
		int tile_y = y / 8;
		int iy = y & 7;
		void *src = s->pixels + tile_y * tile_stride * 4096 + iy * 512;
		void *dst = pixels + y * s->stride;

		for (int x = 0; x < tile_stride; x++) {
			for (int c = 0; c < 512; c += 32) {
				__m256i m = _mm256_load_si256(src + x * 4096 + c);
				m = _mm256_or_si256(m, alpha);
				_mm256_store_si256(dst + x * 512 + c, m);
			}
		}
	}

	return pixels;
}

static char *
detile_ymajor(struct surface *s, __m256i alpha)
{
	int height = align_u64(s->height, 8);
	void *pixels;
	int tile_stride = s->stride / 128;
	const int column_stride = 32 * 16;
	const int columns = s->stride / 16;
	int ret;

	ret = posix_memalign(&pixels, 32, s->stride * height);
	ksim_assert(ret == 0);

	ksim_assert((s->stride & 127) == 0);

	for (int y = 0; y < height; y += 2) {
		int tile_y = y / 32;
		int iy = y & 31;
		void *src = s->pixels + tile_y * tile_stride * 4096 + iy * 16;
		void *dst = pixels + y * s->stride;

		for (int x = 0; x < columns ; x++) {
			__m256i m = _mm256_load_si256(src + x * column_stride);
			m = _mm256_or_si256(m, alpha);
			_mm_store_si128(dst + x * 16, _mm256_extractf128_si256(m, 0));
			_mm_store_si128(dst + x * 16 + s->stride, _mm256_extractf128_si256(m, 1));
		}
	}

	return pixels;
}

void
dump_surface(const char *filename, struct surface *s)
{
	char *linear;
	__m256i alpha;

	int png_format;
	switch (s->format) {
	case SF_R8G8B8X8_UNORM:
	case SF_R8G8B8A8_UNORM:
	case SF_R8G8B8X8_UNORM_SRGB:
	case SF_R8G8B8A8_UNORM_SRGB:
		png_format = PNG_FORMAT_RGBA;
		break;
	case SF_B8G8R8A8_UNORM:
	case SF_B8G8R8X8_UNORM:
	case SF_B8G8R8A8_UNORM_SRGB:
	case SF_B8G8R8X8_UNORM_SRGB:
		png_format = PNG_FORMAT_BGRA;
		break;
	default:
		stub("image format");
		return;
	}

	switch (s->format) {
	case SF_R8G8B8X8_UNORM:
	case SF_B8G8R8X8_UNORM:
	case SF_R8G8B8X8_UNORM_SRGB:
	case SF_B8G8R8X8_UNORM_SRGB:
		alpha = _mm256_set1_epi32(0xff000000);
		break;
	default:
		alpha = _mm256_set1_epi32(0);
		break;
	}

	switch (s->tile_mode) {
	case LINEAR:
		linear = s->pixels;
		break;
	case XMAJOR:
		linear = detile_xmajor(s, alpha);
		break;
	case YMAJOR:
		linear = detile_ymajor(s, alpha);
		break;
	default:
		linear = s->pixels;
		stub("detile wmajor");
		break;
	}

	FILE *f = fopen(filename, "wb");
	ksim_assert(f != NULL);

	png_image pi = {
		.version = PNG_IMAGE_VERSION,
		.width = s->width,
		.height = s->height,
		.format = png_format
	};

	ksim_assert(png_image_write_to_stdio(&pi, f, 0, linear, s->stride, NULL));

	fclose(f);

	if (linear != s->pixels)
		free(linear);
}
