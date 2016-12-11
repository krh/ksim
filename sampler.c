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

#include "eu.h"
#include "avx-builder.h"

enum simd_mode {
	SIMD_MODE_SIMD8D_SIMD4x2,
	SIMD_MODE_SIMD8,
	SIMD_MODE_SIMD16,
	SIMD_MODE_SIMD32,
};

struct message_descriptor {
	uint32_t	binding_table_index;
	uint32_t	sampler_index;
	uint32_t	message_type;
	enum simd_mode	simd_mode;
	bool		header_present;
	uint32_t	response_length;
	uint32_t	message_length;
	uint32_t	return_format;
	bool		eot;
};

static inline struct message_descriptor
unpack_message_descriptor(uint32_t function_control)
{
	/* Vol 2d, "Message Descriptor - Sampling Engine" (p328) */
	return (struct message_descriptor) {
		.binding_table_index	= field(function_control,   0,    7),
		.sampler_index		= field(function_control,   8,   11),
		.message_type		= field(function_control,  12,   16),
		.simd_mode		= field(function_control,  17,   18),
		.header_present		= field(function_control,  19,   19),
		.response_length	= field(function_control,  20,   24),
		.message_length		= field(function_control,  25,   28),
		.return_format		= field(function_control,  30,   30),
		.eot			= field(function_control,  31,   31),
	};
}

enum simd_mode_extension {
	SIMD_MODE_EXTENSION_SIMD8D,
	SIMD_MODE_EXTENSION_SIMD4x2
};

struct message_header {
	uint32_t			r_offset;
	uint32_t			v_offset;
	uint32_t			u_offset;
	uint32_t			red_channel_mask;
	uint32_t			green_channel_mask;
	uint32_t			blue_channel_mask;
	uint32_t			alpha_channel_mask;
	uint32_t			gather4_source_channel_select;
	uint32_t			simd3264_output_format_control;
	enum simd_mode_extension	simd_mode_extension;
	uint32_t			pixel_null_mask_enable;
	uint32_t			render_target_index;
	uint32_t			sampler_state_pointer;
	uint32_t			destination_x_address;
	uint32_t			destination_y_address;
	uint32_t			output_format;
};

/* Vol 7, p 362 */
enum sample_message_type {
	SAMPLE_MESSAGE_SAMPLE		= 0b00000,
	SAMPLE_MESSAGE_SAMPLE_B		= 0b00001,
	SAMPLE_MESSAGE_SAMPLE_L		= 0b00010,
	SAMPLE_MESSAGE_SAMPLE_C		= 0b00011,
	SAMPLE_MESSAGE_SAMPLE_D		= 0b00100,
	SAMPLE_MESSAGE_SAMPLE_B_C	= 0b00101,
	SAMPLE_MESSAGE_SAMPLE_L_C	= 0b00110,
	SAMPLE_MESSAGE_LD		= 0b00111,
	SAMPLE_MESSAGE_GATHER4		= 0b01000,
	SAMPLE_MESSAGE_LOD		= 0b01001,
	SAMPLE_MESSAGE_RESINFO		= 0b01010,
	SAMPLE_MESSAGE_SAMPLEINFO	= 0b01011,
	SAMPLE_MESSAGE_GATHER4_C	= 0b10000,
	SAMPLE_MESSAGE_GATHER4_PO	= 0b10001,
	SAMPLE_MESSAGE_GATHER4_PO_C	= 0b10010,
	SAMPLE_MESSAGE_D_C		= 0b10100,
	SAMPLE_MESSAGE_MIN		= 0b10110,
	SAMPLE_MESSAGE_MAX		= 0b10111,
	SAMPLE_MESSAGE_LZ		= 0b11000,
	SAMPLE_MESSAGE_C_LZ		= 0b11001,
	SAMPLE_MESSAGE_LD_LZ		= 0b11010, /* Not in docs */
	SAMPLE_MESSAGE_LD2DMS_W		= 0b11100,
	SAMPLE_MESSAGE_LD_MCS		= 0b11101,
	SAMPLE_MESSAGE_LD2DMS		= 0b11110,
};

static inline struct message_header
unpack_message_header(const struct reg h)
{
	return (struct message_header) {
		.r_offset			= field(h.ud[2],  0,  3),
		.v_offset			= field(h.ud[2],  4,  7),
		.u_offset			= field(h.ud[2],  8, 11),
		.red_channel_mask		= field(h.ud[2], 12, 12),
		.green_channel_mask		= field(h.ud[2], 13, 13),
		.blue_channel_mask		= field(h.ud[2], 14, 14),
		.alpha_channel_mask		= field(h.ud[2], 15, 15),
		.gather4_source_channel_select	= field(h.ud[2], 16, 17),
		.simd3264_output_format_control	= field(h.ud[2], 18, 19),
		.simd_mode_extension		= field(h.ud[2], 22, 22),
		.pixel_null_mask_enable		= field(h.ud[2], 23, 23),
		.render_target_index		= field(h.ud[2], 24, 31),
		.sampler_state_pointer		= field(h.ud[3],  0, 31),
		.destination_x_address		= field(h.ud[4],  0, 15),
		.destination_y_address		= field(h.ud[4], 16, 31),
		.output_format			= field(h.ud[5],  0,  4),
	};
}

struct sfid_sampler_args {
	int src;
	int dst;
	int header;
	int rlen;
	struct surface tex;
};

static void
sfid_sampler_ld_simd4x2_linear(struct thread *t, const struct sfid_sampler_args *args)
{
	const struct message_header h = unpack_message_header(t->grf[args->header]);
	struct reg u, sample;
	uint32_t *p;
	/* Payload struct MAP32B_TS_SIMD4X2 */

	ksim_assert(h.simd_mode_extension == SIMD_MODE_EXTENSION_SIMD4x2);

	u.ireg = t->grf[args->src].ireg;
	switch (args->tex.format) {
	case SF_R32G32B32A32_FLOAT:
	case SF_R32G32B32A32_SINT:
	case SF_R32G32B32A32_UINT:
		p = args->tex.pixels + u.ud[0] * args->tex.stride;
		sample.ud[0] = p[0];
		sample.ud[1] = p[1];
		sample.ud[2] = p[2];
		sample.ud[3] = p[3];
		t->grf[args->dst] = sample;
		break;
	default:
		stub("unhandled ld format");
		break;
	}
}

static void
load_format_simd8(void *p, uint32_t format, __m256i offsets, __m256i emask, struct reg *dst)
{
	const __m256i zero = _mm256_set1_epi32(0.0f);

	/* Grr, _mm256_mask_i32gather_epi32() is a macro that doesn't
	 * properly protect its arguments and casts the base pointer
	 * arg to an int pointer. We have to put () around where we
	 * add the offset. */

	switch (format) {
	case SF_R32G32B32A32_FLOAT:
	case SF_R32G32B32A32_SINT:
	case SF_R32G32B32A32_UINT:
		dst[0].ireg = _mm256_mask_i32gather_epi32(zero, (p + 0), offsets, emask, 1);
		dst[1].ireg = _mm256_mask_i32gather_epi32(zero, (p + 4), offsets, emask, 1);
		dst[2].ireg = _mm256_mask_i32gather_epi32(zero, (p + 8), offsets, emask, 1);
		dst[3].ireg = _mm256_mask_i32gather_epi32(zero, (p + 12), offsets, emask, 1);
		break;
	case SF_R16G16B16A16_UINT: {
		const __m256i mask = _mm256_set1_epi32(0xffff);
		struct reg rg, ba;

		rg.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[0].ireg = _mm256_and_si256(rg.ireg, mask);
		dst[1].ireg = _mm256_srli_epi32(rg.ireg, 16);
		ba.ireg = _mm256_mask_i32gather_epi32(zero, (p + 4), offsets, emask, 1);
		dst[2].ireg = _mm256_and_si256(ba.ireg, mask);
		dst[3].ireg = _mm256_srli_epi32(ba.ireg, 16);
		break;
	}

	case SF_R16G16B16A16_UNORM: {
		const __m256i mask = _mm256_set1_epi32(0xffff);
		const __m256 scale = _mm256_set1_ps(1.0f / 65535.0f);
		struct reg rg, ba;

		rg.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rg.ireg, mask)), scale);
		dst[1].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(rg.ireg, 16)), scale);
		ba.ireg = _mm256_mask_i32gather_epi32(zero, (p + 4), offsets, emask, 1);
		dst[2].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(ba.ireg, mask)), scale);
		dst[3].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(ba.ireg, 16)), scale);
		break;
	}

	case SF_R8G8B8X8_UNORM: {
		const __m256i mask = _mm256_set1_epi32(0xff);
		const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
		struct reg rgbx;

		rgbx.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgbx.ireg, mask)), scale);
		rgbx.ireg = _mm256_srli_epi32(rgbx.ireg, 8);
		dst[1].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgbx.ireg, mask)), scale);
		rgbx.ireg = _mm256_srli_epi32(rgbx.ireg, 8);
		dst[2].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgbx.ireg, mask)), scale);
		dst[3].reg = _mm256_set1_ps(1.0f);
		break;
	}

	case SF_R8G8B8A8_UNORM: {
		const __m256i mask = _mm256_set1_epi32(0xff);
		const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
		struct reg rgba;

		rgba.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgba.ireg, mask)), scale);
		rgba.ireg = _mm256_srli_epi32(rgba.ireg, 8);
		dst[1].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgba.ireg, mask)), scale);
		rgba.ireg = _mm256_srli_epi32(rgba.ireg, 8);
		dst[2].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgba.ireg, mask)), scale);
		rgba.ireg = _mm256_srli_epi32(rgba.ireg, 8);
		dst[3].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgba.ireg, mask)), scale);
		break;
	}

	case SF_B8G8R8X8_UNORM: {
		const __m256i mask = _mm256_set1_epi32(0xff);
		const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
		struct reg bgrx;

		bgrx.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[2].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(bgrx.ireg, mask)), scale);
		bgrx.ireg = _mm256_srli_epi32(bgrx.ireg, 8);
		dst[1].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(bgrx.ireg, mask)), scale);
		bgrx.ireg = _mm256_srli_epi32(bgrx.ireg, 8);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(bgrx.ireg, mask)), scale);
		dst[3].reg = _mm256_set1_ps(1.0f);
		break;
	}

	case SF_R8G8B8A8_UINT: {
		const __m256i mask = _mm256_set1_epi32(0xff);
		struct reg rgba;

		rgba.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[0].ireg = _mm256_and_si256(rgba.ireg, mask);
		rgba.ireg = _mm256_srli_epi32(rgba.ireg, 8);
		dst[1].ireg = _mm256_and_si256(rgba.ireg, mask);
		rgba.ireg = _mm256_srli_epi32(rgba.ireg, 8);
		dst[2].ireg = _mm256_and_si256(rgba.ireg, mask);
		rgba.ireg = _mm256_srli_epi32(rgba.ireg, 8);
		dst[3].ireg = _mm256_and_si256(rgba.ireg, mask);
		break;
	}

	case SF_R8_UNORM: {
		const __m256i mask = _mm256_set1_epi32(0xff);
		const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
		struct reg rgba;

		rgba.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgba.ireg, mask)), scale);
		break;
	}

	case SF_L8_UNORM: {
		const __m256i mask = _mm256_set1_epi32(0xff);
		const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
		struct reg rgba;

		rgba.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgba.ireg, mask)), scale);
		dst[1].reg = dst[0].reg;
		dst[2].reg = dst[0].reg;
		dst[3].reg = _mm256_set1_ps(1.0f);
		break;
	}

	case SF_R8_UINT: {
		const __m256i mask = _mm256_set1_epi32(0xff);
		struct reg rgba;

		rgba.ireg = _mm256_mask_i32gather_epi32(zero, (p + 0), offsets, emask, 1);
		dst[0].ireg = _mm256_and_si256(rgba.ireg, mask);
		break;
	}

	case SF_R24_UNORM_X8_TYPELESS: {
		const __m256i mask = _mm256_set1_epi32(0xffffff);
		const __m256 scale = _mm256_set1_ps(1.0f / 16777215.0f);
		struct reg r;

		r.ireg = _mm256_mask_i32gather_epi32(zero, (p + 0), offsets, emask, 1);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(r.ireg, mask)), scale);
		break;
	}

	default:
		dst[0].reg = _mm256_set1_ps(1.0f);
		dst[1].reg = _mm256_set1_ps(0.0f);
		dst[2].reg = _mm256_set1_ps(0.0f);
		dst[3].reg = _mm256_set1_ps(1.0f);
		stub("sampler ld format %d", format);
		break;
	}
}

static void
sfid_sampler_ld_simd8_linear(struct thread *t, const struct sfid_sampler_args *args)
{
	struct reg u, v;

	u.ireg = t->grf[args->src].ireg;
	v.ireg = t->grf[args->src + 1].ireg;
	void *p = args->tex.pixels;

	struct reg offsets;
	offsets.ireg =
		_mm256_add_epi32(_mm256_mullo_epi32(u.ireg, _mm256_set1_epi32(args->tex.cpp)),
				 _mm256_mullo_epi32(v.ireg, _mm256_set1_epi32(args->tex.stride)));

	load_format_simd8(p, args->tex.format,
			  offsets.ireg, t->mask_q1, &t->grf[args->dst]);
}

static void
sfid_sampler_ld_simd16_linear(struct thread *t, const struct sfid_sampler_args *args)
{
	struct reg u, v;
	void *p = args->tex.pixels;
	struct reg offsets;

	u.ireg = t->grf[args->src].ireg;
	v.ireg = t->grf[args->src + 1].ireg;
	offsets.ireg =
		_mm256_add_epi32(_mm256_mullo_epi32(u.ireg, _mm256_set1_epi32(args->tex.cpp)),
				 _mm256_mullo_epi32(v.ireg, _mm256_set1_epi32(args->tex.stride)));

	load_format_simd8(p, args->tex.format,
			  offsets.ireg, t->mask_q1, &t->grf[args->dst]);

	u.ireg = t->grf[args->src + 2].ireg;
	v.ireg = t->grf[args->src + 3].ireg;
	offsets.ireg =
		_mm256_add_epi32(_mm256_mullo_epi32(u.ireg, _mm256_set1_epi32(args->tex.cpp)),
				 _mm256_mullo_epi32(v.ireg, _mm256_set1_epi32(args->tex.stride)));

	uint32_t dst1 = args->dst + format_channels(args->tex.format);
	load_format_simd8(p, args->tex.format,
			  offsets.ireg, t->mask_q2, &t->grf[dst1]);
}

struct sample_position {
	struct reg u;
	struct reg v;
	struct reg r;
};

static void
transform_sample_position(const struct sfid_sampler_args *args, struct reg *src,
			  struct sample_position *coords)
{
	struct reg u, v;

	if (args->tex.type == SURFTYPE_CUBE) {
		/* Compare x and z first so we end up with x or z as u. */
		__m256i abs_mask = _mm256_set1_epi32(0x7fffffff);
		__m256 x = (__m256) _mm256_and_si256(src[0].ireg, abs_mask);
		__m256 z = (__m256)_mm256_and_si256(src[2].ireg, abs_mask);
		__m256 xz_mask = _mm256_cmp_ps(x, z, _CMP_GT_OQ);
		__m256 abs_xz_major = _mm256_blendv_ps(z, x, xz_mask);
		__m256 xz_major = _mm256_blendv_ps(src[2].reg, src[0].reg, xz_mask);
		__m256 us = _mm256_blendv_ps(src[0].reg, src[2].reg, xz_mask);
		__m256i face = _mm256_blendv_epi8(_mm256_set1_epi32(4),
						  _mm256_set1_epi32(0),
						  (__m256i) xz_mask);

		__m256 y = (__m256) _mm256_and_si256(src[1].ireg, abs_mask);
		__m256 y_mask = _mm256_cmp_ps(y, abs_xz_major, _CMP_GT_OQ);
		__m256 major = _mm256_blendv_ps(xz_major, src[1].reg, y_mask);
		__m256 vs = _mm256_blendv_ps(src[1].reg, xz_major, y_mask);
		face = _mm256_blendv_epi8(face, _mm256_set1_epi32(2),
					  (__m256i) y_mask);

		const __m256 mhalf = _mm256_set1_ps(-0.5f);
		const __m256 half = _mm256_set1_ps(0.5f);
		us = _mm256_add_ps(_mm256_mul_ps(us, half), half);
		vs = _mm256_add_ps(_mm256_mul_ps(vs, mhalf), half);

		/* For cases where x > z and y > both, we end up with
		 * u = z and v = x. That's the wrong ordering, so swap
		 * those. */
		__m256i swap_xz_mask = _mm256_and_si256((__m256i) xz_mask,
							(__m256i) y_mask);
		u.reg = _mm256_blendv_ps(us, vs, (__m256) swap_xz_mask);
		v.reg = _mm256_blendv_ps(vs, us, (__m256) swap_xz_mask);

		/* FIXME: Missing negation on u for +x and -z cases,
		 * on v for +y case. */

		/* Add sign bit to determine positive or negative face */
		coords->r.ireg =
			_mm256_add_epi32(face, _mm256_srli_epi32((__m256i) major, 31));

	} else {
		/* Wrap */
		u.reg = _mm256_floor_ps(src[0].reg);
		u.reg = _mm256_sub_ps(src[0].reg, u.reg);

		v.reg = _mm256_floor_ps(src[1].reg);
		v.reg = _mm256_sub_ps(src[1].reg, v.reg);

#if 0
		/* Clamp */
		u.reg = _mm256_min_ps(src[0].reg, _mm256_set1_ps(1.0f));
		u.reg = _mm256_max_ps(u.reg, _mm256_setzero_ps());

		v.reg = _mm256_min_ps(src[1].reg, _mm256_set1_ps(1.0f));
		v.reg = _mm256_max_ps(v.reg, _mm256_setzero_ps());
#endif
	}

	u.reg = _mm256_mul_ps(u.reg, _mm256_set1_ps(args->tex.width));
	v.reg = _mm256_mul_ps(v.reg, _mm256_set1_ps(args->tex.height));

	u.reg = _mm256_add_ps(u.reg, _mm256_set1_ps(0.25f));
	v.reg = _mm256_add_ps(v.reg, _mm256_set1_ps(0.25f));

	coords->u.ireg = _mm256_cvttps_epi32(u.reg);
	coords->v.ireg = _mm256_cvttps_epi32(v.reg);
}

static void
sfid_sampler_sample_simd8_linear(struct thread *t, const struct sfid_sampler_args *args)
{
	struct sample_position pos;

	transform_sample_position(args, &t->grf[args->src], &pos);

	struct reg offsets;
	offsets.ireg =
		_mm256_add_epi32(_mm256_mullo_epi32(pos.u.ireg, _mm256_set1_epi32(args->tex.cpp)),
				 _mm256_mullo_epi32(pos.v.ireg, _mm256_set1_epi32(args->tex.stride)));

	load_format_simd8(args->tex.pixels, args->tex.format,
			  offsets.ireg, t->mask_q1, &t->grf[args->dst]);
}

static void
sfid_sampler_sample_simd8_ymajor(struct thread *t, const struct sfid_sampler_args *args)
{
	struct sample_position pos;

	transform_sample_position(args, &t->grf[args->src], &pos);

	ksim_assert(is_power_of_two(args->tex.cpp));
	const int log2_cpp = __builtin_ffs(args->tex.cpp) - 1;
	__m256i u_bytes = _mm256_slli_epi32(pos.u.ireg, log2_cpp);

	__m256i tile_y = _mm256_srli_epi32(pos.v.ireg, 5);
	__m256i stride_in_tiles = _mm256_set1_epi32(4096 * args->tex.stride / 128);
	__m256i tile_base = _mm256_mullo_epi32(tile_y, stride_in_tiles);

	if (args->tex.type == SURFTYPE_CUBE) {
		__m256i slice_stride =
			_mm256_set1_epi32(args->tex.stride * args->tex.qpitch);
		__m256i slice_base =
			_mm256_mullo_epi32(pos.r.ireg, slice_stride);

		tile_base = _mm256_add_epi32(tile_base, slice_base);
	}

	__m256i oword_offset = _mm256_and_si256(u_bytes, _mm256_set1_epi32(0xf));
	__m256i column_offset = _mm256_slli_epi32(_mm256_srli_epi32(u_bytes, 4), 9);
	__m256i row = _mm256_and_si256(pos.v.ireg, _mm256_set1_epi32(0x1f));
	__m256i row_offset = _mm256_slli_epi32(row, 4);

	__m256i offset = _mm256_add_epi32(_mm256_add_epi32(tile_base, row_offset),
					  _mm256_add_epi32(oword_offset, column_offset));

	load_format_simd8(args->tex.pixels, args->tex.format,
			  offset, t->mask_q1, &t->grf[args->dst]);
}

static void
sfid_sampler_sample_simd8_xmajor(struct thread *t, const struct sfid_sampler_args *args)
{
	struct sample_position pos;

	transform_sample_position(args, &t->grf[args->src], &pos);

	ksim_assert(is_power_of_two(args->tex.cpp));
	const int log2_cpp = __builtin_ffs(args->tex.cpp) - 1;
	__m256i u_bytes = _mm256_slli_epi32(pos.u.ireg, log2_cpp);

	__m256i tile_y = _mm256_srli_epi32(pos.v.ireg, 3);
	__m256i stride_in_tiles = _mm256_set1_epi32(4096 * args->tex.stride / 512);
	__m256i tile_base = _mm256_mullo_epi32(tile_y, stride_in_tiles);

	__m256i intra_column_offset = _mm256_and_si256(u_bytes, _mm256_set1_epi32(511));
	__m256i column_offset = _mm256_slli_epi32(_mm256_srli_epi32(u_bytes, 9), 12);
	__m256i row = _mm256_and_si256(pos.v.ireg, _mm256_set1_epi32(0x7));
	__m256i row_offset = _mm256_slli_epi32(row, 9);

	__m256i offset = _mm256_add_epi32(_mm256_add_epi32(tile_base, row_offset),
					  _mm256_add_epi32(intra_column_offset, column_offset));

	load_format_simd8(args->tex.pixels, args->tex.format,
			  offset, t->mask_q1, &t->grf[args->dst]);
}

static void
sfid_sampler_noop_stub(struct thread *t, const struct sfid_sampler_args *args)
{
	struct reg *dst = &t->grf[args->dst];

	dst[0].reg = _mm256_set1_ps(1.0f);
	dst[1].reg = _mm256_set1_ps(0.0f);
	dst[2].reg = _mm256_set1_ps(0.0f);
	dst[3].reg = _mm256_set1_ps(1.0f);
}

void
builder_emit_sfid_sampler(struct builder *bld, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	const int exec_size = 1 << unpack_inst_common(inst).exec_size;
	struct sfid_sampler_args *args;
	void *func;
	int num;

	const struct message_descriptor d =
		unpack_message_descriptor(send.function_control);

	args = builder_get_const_data(bld, sizeof *args, 8);

	args->dst = unpack_inst_2src_dst(inst).num;
	num = unpack_inst_2src_src0(inst).num;
	if (d.header_present)
		args->header = num++;
	else
		args->header = -1;
	args->src = num;

	bool tex_valid = get_surface(bld->binding_table_address,
				     d.binding_table_index, &args->tex);
	ksim_assert(tex_valid);

	builder_emit_load_rsi_rip_relative(bld, builder_offset(bld, args));

	switch (d.message_type) {
	case SAMPLE_MESSAGE_LD:
	case SAMPLE_MESSAGE_LD_LZ:
		if (d.simd_mode == SIMD_MODE_SIMD8D_SIMD4x2 &&
		    args->tex.tile_mode == LINEAR) {
			/* We only handle 4x2, which on SKL requires
			 * the simd mode extension bit in the header
			 * to be set. Assert we have a header. */
			ksim_assert(d.header_present);
			ksim_assert(exec_size == 4);
			func = sfid_sampler_ld_simd4x2_linear;
		} else if (d.simd_mode == SIMD_MODE_SIMD8 &&
			   args->tex.tile_mode == LINEAR) {
			func = sfid_sampler_ld_simd8_linear;
		} else if (d.simd_mode == SIMD_MODE_SIMD16 &&
			   args->tex.tile_mode == LINEAR) {
			func = sfid_sampler_ld_simd16_linear;
		} else {
			stub("ld simd mode %d", d.simd_mode);
			func = sfid_sampler_noop_stub;
		}
		break;
	default:
		if (args->tex.tile_mode == LINEAR) {
			func = sfid_sampler_sample_simd8_linear;
		} else if (args->tex.tile_mode == YMAJOR) {
			func = sfid_sampler_sample_simd8_ymajor;
		} else if (args->tex.tile_mode == XMAJOR) {
			func = sfid_sampler_sample_simd8_xmajor;
		} else {
			stub("sampler tile mode %d", args->tex.tile_mode);
			func = sfid_sampler_noop_stub;
		}
		break;
	}

	builder_emit_call(bld, func);

	args->rlen = send.rlen;
	if (args->rlen == 0) {
		const uint32_t bti = 0; /* Should be M0.2 from header */
		const uint32_t opcode = 12;
		const uint32_t type = 4;
		void *func;
		/* dst is the null reg for rlen 0 messages, and so
		 * we'll end up overwriting grf0 - grf3.  We need the
		 * fragment x and y from grf1. so move it up. */
		args->dst = 2;

		func = builder_emit_sfid_render_cache_helper(bld, opcode,
							     type, args->dst, bti);
		builder_emit_jmp_relative(bld, (uint8_t *) func - bld->p);
	}

	builder_invalidate_all(bld);
}
