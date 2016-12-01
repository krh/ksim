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
#include "avx-builder.h"

struct sfid_render_cache_args {
	int src;
	struct surface rt;
};

static inline void
blend_unorm8_argb(struct reg *src, __m256i dst_argb)
{
	if (gt.blend.enable) {
		const __m256i mask = _mm256_set1_epi32(0xff);
		const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
		struct reg dst[4];

		/* Convert to float */
		dst[2].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(dst_argb, mask)), scale);
		dst_argb = _mm256_srli_epi32(dst_argb, 8);
		dst[1].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(dst_argb, mask)), scale);
		dst_argb = _mm256_srli_epi32(dst_argb, 8);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(dst_argb, mask)), scale);
		dst_argb = _mm256_srli_epi32(dst_argb, 8);
		dst[3].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(dst_argb, mask)), scale);

		/* Blend, assuming src BLENDFACTOR_SRC_ALPHA, dst
		 * BLENDFACTOR_INV_SRC_ALPHA, and BLENDFUNCTION_ADD. */
		const __m256 inv_alpha = _mm256_sub_ps(_mm256_set1_ps(1.0f), src[3].reg);
		src[0].reg = _mm256_add_ps(_mm256_mul_ps(src[3].reg, src[0].reg),
					   _mm256_mul_ps(inv_alpha, dst[0].reg));
		src[1].reg = _mm256_add_ps(_mm256_mul_ps(src[3].reg, src[1].reg),
					   _mm256_mul_ps(inv_alpha, dst[1].reg));
		src[2].reg = _mm256_add_ps(_mm256_mul_ps(src[3].reg, src[2].reg),
					   _mm256_mul_ps(inv_alpha, dst[2].reg));
		src[3].reg = _mm256_add_ps(_mm256_mul_ps(src[3].reg, src[3].reg),
					   _mm256_mul_ps(inv_alpha, dst[3].reg));
	}
}

__m256 _ZGVdN8vv_powf(__m256 x, __m256 y);

static inline void
gamma_correct(enum GEN9_SURFACE_FORMAT format, struct reg *c)
{
	if (srgb_format(format)) {
		const __m256 inv_gamma = _mm256_set1_ps(1.0f / 2.4f);
		c[0].reg = _ZGVdN8vv_powf(c[0].reg, inv_gamma);
		c[1].reg = _ZGVdN8vv_powf(c[1].reg, inv_gamma);
		c[2].reg = _ZGVdN8vv_powf(c[2].reg, inv_gamma);
	}
}

static void
sfid_render_cache_rt_write_rep16_bgra_unorm8_xmajor(struct thread *t,
						    const struct sfid_render_cache_args *args)
{
	const __m128 scale = _mm_set1_ps(255.0f);
	const __m128 half =  _mm_set1_ps(0.5f);
	struct reg src[1];

	memcpy(src, &t->grf[args->src], sizeof(src));

	if (srgb_format(args->rt.format)) {
		const __m256 inv_gamma = _mm256_set1_ps(1.0f / 2.4f);
		src[0].reg = _ZGVdN8vv_powf(src[0].reg, inv_gamma);
		/* Don't gamma correct alpha */
		src[0].f[3] = t->grf[args->src].f[3];
	}

	__m128 bgra = _mm_shuffle_ps(_mm256_castps256_ps128(src[0].reg),
				     _mm256_castps256_ps128(src[0].reg),
				     SWIZZLE(2, 1, 0, 3));

	bgra = _mm_mul_ps(bgra, scale);
	bgra = _mm_add_ps(bgra, half);

	__m128i bgra_i = _mm_cvtps_epi32(bgra);
	bgra_i = _mm_packus_epi32(bgra_i, bgra_i);
	bgra_i = _mm_packus_epi16(bgra_i, bgra_i);

	/* Swizzle two middle mask pairs so that dword 0-3 and 4-7
	 * form linear owords of pixels. */
	__m256i mask = _mm256_permute4x64_epi64(t->mask_q1, SWIZZLE(0, 2, 1, 3));

	const int x0 = t->grf[1].uw[4];
	const int y0 = t->grf[1].uw[5];
	const int cpp = 4;
	void *base0 = xmajor_offset(args->rt.pixels, x0,  y0, args->rt.stride, cpp);

	_mm_maskstore_epi32(base0, _mm256_extractf128_si256(mask, 0), bgra_i);
	_mm_maskstore_epi32(base0 + 512, _mm256_extractf128_si256(mask, 1), bgra_i);

	const int x1 = t->grf[1].uw[8];
	const int y1 = t->grf[1].uw[9];
	void *base1 = xmajor_offset(args->rt.pixels, x1,  y1, args->rt.stride, 4);

	__m256i mask1 = _mm256_permute4x64_epi64(t->mask_q2, SWIZZLE(0, 2, 1, 3));
	_mm_maskstore_epi32(base1, _mm256_extractf128_si256(mask1, 0), bgra_i);
	_mm_maskstore_epi32(base1 + 512, _mm256_extractf128_si256(mask1, 1), bgra_i);
}

static void
sfid_render_cache_rt_write_rep16_rgba_unorm8_ymajor(struct thread *t,
						    const struct sfid_render_cache_args *args)
{
	const __m128 scale = _mm_set1_ps(255.0f);
	const __m128 half =  _mm_set1_ps(0.5f);
	struct reg src[1];

	memcpy(src, &t->grf[args->src], sizeof(src));

	if (srgb_format(args->rt.format)) {
		const __m256 inv_gamma = _mm256_set1_ps(1.0f / 2.4f);
		src[0].reg = _ZGVdN8vv_powf(src[0].reg, inv_gamma);
		/* Don't gamma correct alpha */
		src[0].f[3] = t->grf[args->src].f[3];
	}

	__m128 rgba = _mm256_castps256_ps128(src[0].reg);

	rgba = _mm_mul_ps(rgba, scale);
	rgba = _mm_add_ps(rgba, half);

	__m128i rgba_i = _mm_cvtps_epi32(rgba);
	rgba_i = _mm_packus_epi32(rgba_i, rgba_i);
	rgba_i = _mm_packus_epi16(rgba_i, rgba_i);

	/* Swizzle two middle mask pairs so that dword 0-3 and 4-7
	 * form linear owords of pixels. */
	__m256i mask0 = _mm256_permute4x64_epi64(t->mask_q1, SWIZZLE(0, 2, 1, 3));

	const int cpp = 4;
	const int x0 = t->grf[1].uw[4];
	const int y0 = t->grf[1].uw[5];
	void *base0 = ymajor_offset(args->rt.pixels, x0, y0, args->rt.stride, cpp);

	_mm_maskstore_epi32(base0, _mm256_extractf128_si256(mask0, 0), rgba_i);
	_mm_maskstore_epi32(base0 + 16, _mm256_extractf128_si256(mask0, 1), rgba_i);

	const int x1 = t->grf[1].uw[8];
	const int y1 = t->grf[1].uw[9];
	void *base1 = ymajor_offset(args->rt.pixels, x1, y1, args->rt.stride, cpp);
	__m256i mask1 = _mm256_permute4x64_epi64(t->mask_q2, SWIZZLE(0, 2, 1, 3));

	_mm_maskstore_epi32(base1, _mm256_extractf128_si256(mask1, 0), rgba_i);
	_mm_maskstore_epi32(base1 + 16, _mm256_extractf128_si256(mask1, 1), rgba_i);
}

static inline __m256i
to_unorm(__m256 reg, float scale_f)
{
	const __m256 scale = _mm256_set1_ps(scale_f);
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 zero = _mm256_set1_ps(0.0f);
	const __m256 half =  _mm256_set1_ps(0.5f);

	const __m256 clamped = _mm256_max_ps(_mm256_min_ps(reg, one), zero);

	return _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(clamped, scale), half));
}

static void
sfid_render_cache_rt_write_simd8_bgra_unorm8_xmajor(struct thread *t,
						    const struct sfid_render_cache_args *args)
{
	const int x = t->grf[1].uw[4];
	const int y = t->grf[1].uw[5];
	__m256i argb;
	const float scale = 255.0f;
	struct reg src[4];

	memcpy(src, &t->grf[args->src], sizeof(src));

	const int cpp = 4;
	void *base = xmajor_offset(args->rt.pixels, x, y, args->rt.stride, cpp);

	if (gt.blend.enable) {
		/* Load unorm8 */
		__m128i lo = _mm_load_si128(base);
		__m128i hi = _mm_load_si128(base + 512);
		__m256i dst_argb = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
		dst_argb = _mm256_permute4x64_epi64(dst_argb, SWIZZLE(0, 2, 1, 3));

		blend_unorm8_argb(src, dst_argb);
	}

	gamma_correct(args->rt.format, src);

	const __m256i r = to_unorm(src[0].reg, scale);
	const __m256i g = to_unorm(src[1].reg, scale);
	const __m256i b = to_unorm(src[2].reg, scale);
	const __m256i a = to_unorm(src[3].reg, scale);

	argb = _mm256_slli_epi32(a, 8);
	argb = _mm256_or_si256(argb, r);
	argb = _mm256_slli_epi32(argb, 8);
	argb = _mm256_or_si256(argb, g);
	argb = _mm256_slli_epi32(argb, 8);
	argb = _mm256_or_si256(argb, b);

	/* Swizzle two middle pixel pairs so that dword 0-3 and 4-7
	 * form linear owords of pixels. */
	argb = _mm256_permute4x64_epi64(argb, SWIZZLE(0, 2, 1, 3));
	__m256i mask = _mm256_permute4x64_epi64(t->mask_q1, SWIZZLE(0, 2, 1, 3));

	_mm_maskstore_epi32(base,
			    _mm256_extractf128_si256(mask, 0),
			    _mm256_extractf128_si256(argb, 0));
	_mm_maskstore_epi32(base + 512,
			    _mm256_extractf128_si256(mask, 1),
			    _mm256_extractf128_si256(argb, 1));
}

static void
write_uint8_linear(struct thread *t,
		   const struct sfid_render_cache_args *args,
		   __m256i r, __m256i g, __m256i b, __m256i a)
{
	const int x = t->grf[1].uw[4];
	const int y = t->grf[1].uw[5];
	__m256i rgba;

	rgba = _mm256_slli_epi32(a, 8);
	rgba = _mm256_or_si256(rgba, b);
	rgba = _mm256_slli_epi32(rgba, 8);
	rgba = _mm256_or_si256(rgba, g);
	rgba = _mm256_slli_epi32(rgba, 8);
	rgba = _mm256_or_si256(rgba, r);

#define SWIZZLE(x, y, z, w) \
	( ((x) << 0) | ((y) << 2) | ((z) << 4) | ((w) << 6) )

	/* Swizzle two middle pixel pairs so that dword 0-3 and 4-7
	 * form linear owords of pixels. */
	rgba = _mm256_permute4x64_epi64(rgba, SWIZZLE(0, 2, 1, 3));
	__m256i mask = _mm256_permute4x64_epi64(t->mask_q1, SWIZZLE(0, 2, 1, 3));

	void *base = args->rt.pixels + x * args->rt.cpp + y * args->rt.stride;

	_mm_maskstore_epi32(base,
			    _mm256_extractf128_si256(mask, 0),
			    _mm256_extractf128_si256(rgba, 0));
	_mm_maskstore_epi32(base + args->rt.stride,
			    _mm256_extractf128_si256(mask, 1),
			    _mm256_extractf128_si256(rgba, 1));
}

static void
sfid_render_cache_rt_write_simd8_rgba_unorm8_linear(struct thread *t,
						    const struct sfid_render_cache_args *args)
{
	const float scale = 255.0f;
	const struct reg *src = &t->grf[args->src];

	const __m256i r = to_unorm(src[0].reg, scale);
	const __m256i g = to_unorm(src[1].reg, scale);
	const __m256i b = to_unorm(src[2].reg, scale);
	const __m256i a = to_unorm(src[3].reg, scale);

	write_uint8_linear(t, args, r, g, b, a);
}

static void
sfid_render_cache_rt_write_simd8_rgba_uint8_linear(struct thread *t,
						   const struct sfid_render_cache_args *args)
{
	__m256i r, g, b, a;
	struct reg *src = &t->grf[args->src];

	r = src[0].ireg;
	g = src[1].ireg;
	b = src[2].ireg;
	a = src[3].ireg;

	write_uint8_linear(t, args, r, g, b, a);
}

static void
sfid_render_cache_rt_write_simd16(struct thread *t,
				  const struct sfid_render_cache_args *args)
{
	stub("sfid_render_cache_rt_write_simd16");
}

static void
sfid_render_cache_rt_write_simd8_rgba_uint32_linear(struct thread *t,
						    const struct sfid_render_cache_args *args)
{
	const int x = t->grf[1].uw[4];
	const int y = t->grf[1].uw[5];
	const struct reg *src = &t->grf[args->src];

	__m128i *base0 = args->rt.pixels + x * args->rt.cpp + y * args->rt.stride;
	__m128i *base1 = (void *) base0 + args->rt.stride;

	__m256i rg0145 = _mm256_unpacklo_epi32(src[0].ireg, src[1].ireg);
	__m256i rg2367 = _mm256_unpackhi_epi32(src[0].ireg, src[1].ireg);
	__m256i ba0145 = _mm256_unpacklo_epi32(src[2].ireg, src[3].ireg);
	__m256i ba2367 = _mm256_unpackhi_epi32(src[2].ireg, src[3].ireg);

	__m256i rgba04 = _mm256_unpacklo_epi64(rg0145, ba0145);
	__m256i rgba15 = _mm256_unpackhi_epi64(rg0145, ba0145);

	__m256i rgba26 = _mm256_unpacklo_epi64(rg2367, ba2367);
	__m256i rgba37 = _mm256_unpackhi_epi64(rg2367, ba2367);

	struct reg mask = { .ireg = t->mask_q1 };

	if (mask.d[0] < 0)
		base0[0] = _mm256_extractf128_si256(rgba04, 0);
	if (mask.d[1] < 0)
		base0[1] = _mm256_extractf128_si256(rgba15, 0);
	if (mask.d[2] < 0)
		base1[0] = _mm256_extractf128_si256(rgba26, 0);
	if (mask.d[3] < 0)
		base1[1] = _mm256_extractf128_si256(rgba37, 0);

	if (mask.d[4] < 0)
		base0[2] = _mm256_extractf128_si256(rgba04, 1);
	if (mask.d[5] < 0)
		base0[3] = _mm256_extractf128_si256(rgba15, 1);
	if (mask.d[6] < 0)
		base1[2] = _mm256_extractf128_si256(rgba26, 1);
	if (mask.d[7] < 0)
		base1[3] = _mm256_extractf128_si256(rgba37, 1);
}

static void
write_uint16_linear(struct thread *t,
		    const struct sfid_render_cache_args *args,
		    __m256i r, __m256i g, __m256i b, __m256i a)
{
	const int x = t->grf[1].uw[4];
	const int y = t->grf[1].uw[5];
	__m256i rg, ba;

	rg = _mm256_slli_epi32(g, 16);
	rg = _mm256_or_si256(rg, r);
	ba = _mm256_slli_epi32(a, 16);
	ba = _mm256_or_si256(ba, b);

	__m256i p0 = _mm256_unpacklo_epi32(rg, ba);
	__m256i m0 = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(t->mask_q1, 0));

	__m256i p1 = _mm256_unpackhi_epi32(rg, ba);
	__m256i m1 = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(t->mask_q1, 1));

	void *base = args->rt.pixels + x * args->rt.cpp + y * args->rt.stride;

	_mm_maskstore_epi64(base,
			    _mm256_extractf128_si256(m0, 0),
			    _mm256_extractf128_si256(p0, 0));
	_mm_maskstore_epi64((base + 16),
			    _mm256_extractf128_si256(m1, 0),
			    _mm256_extractf128_si256(p0, 1));

	_mm_maskstore_epi64((base + args->rt.stride),
			    _mm256_extractf128_si256(m0, 1),
			    _mm256_extractf128_si256(p1, 0));
	_mm_maskstore_epi64((base + args->rt.stride + 16),
			    _mm256_extractf128_si256(m1, 1),
			    _mm256_extractf128_si256(p1, 1));
}

static void
sfid_render_cache_rt_write_simd8_rgba_unorm16_linear(struct thread *t,
						     const struct sfid_render_cache_args *args)
{
	__m256i r, g, b, a;
	const __m256 scale = _mm256_set1_ps(65535.0f);
	const __m256 half =  _mm256_set1_ps(0.5f);
	struct reg *src = &t->grf[args->src];

	r = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[0].reg, scale), half));
	g = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[1].reg, scale), half));
	b = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[2].reg, scale), half));
	a = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[3].reg, scale), half));

	write_uint16_linear(t, args, r, g, b, a);
}

static void
sfid_render_cache_rt_write_simd8_rgba_uint16_linear(struct thread *t,
						    const struct sfid_render_cache_args *args)
{
	__m256i r, g, b, a;
	struct reg *src = &t->grf[args->src];

	r = src[0].ireg;
	g = src[1].ireg;
	b = src[2].ireg;
	a = src[3].ireg;

	write_uint16_linear(t, args, r, g, b, a);
}

static void
sfid_render_cache_rt_write_simd8_r_uint8_ymajor(struct thread *t,
						const struct sfid_render_cache_args *args)
{
	const int x = t->grf[1].uw[4];
	const int y = t->grf[1].uw[5];
	const int cpp = 1;

	void *base = ymajor_offset(args->rt.pixels, x, y, args->rt.stride, cpp);

	struct reg *src = &t->grf[args->src];

	__m256i r32 = _mm256_permute4x64_epi64(src[0].ireg, SWIZZLE(0, 2, 1, 3));

	__m128i lo = _mm256_extractf128_si256(r32, 0);
	__m128i hi = _mm256_extractf128_si256(r32, 1);
	__m128i r16 = _mm_packus_epi32(lo, hi);
	__m128i r8 = _mm_packus_epi16(r16, r16);

	/* FIXME: Needs masking. */
	*(uint32_t *) (base +  0) = _mm_extract_epi32(r8, 0);
	*(uint32_t *) (base + 16) = _mm_extract_epi32(r8, 1);
}

static void
sfid_render_cache_rt_write_simd8_unorm8_ymajor(struct thread *t,
					       const struct sfid_render_cache_args *args)
{
	const int x = t->grf[1].uw[4];
	const int y = t->grf[1].uw[5];
	const int cpp = 1;
	struct reg *src = &t->grf[args->src];
	const __m256 scale = _mm256_set1_ps(255.0f);
	const __m256 half =  _mm256_set1_ps(0.5f);
	__m256i r, g, b, a;
	__m256i rgba;

	switch (args->rt.format) {
	case SF_R8G8B8A8_UNORM:
		r = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[0].reg, scale), half));
		g = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[1].reg, scale), half));
		b = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[2].reg, scale), half));
		a = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[3].reg, scale), half));
		break;
	case SF_B8G8R8A8_UNORM:
		b = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[0].reg, scale), half));
		g = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[1].reg, scale), half));
		r = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[2].reg, scale), half));
		a = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(src[3].reg, scale), half));
		break;
	default:
		stub("unorm8 ymajor format");
		return;
	}

	rgba = _mm256_slli_epi32(r, 8);
	rgba = _mm256_or_si256(rgba, g);
	rgba = _mm256_slli_epi32(rgba, 8);
	rgba = _mm256_or_si256(rgba, b);
	rgba = _mm256_slli_epi32(rgba, 8);
	rgba = _mm256_or_si256(rgba, a);

	/* Swizzle two middle pixel pairs so that dword 0-3 and 4-7
	 * form linear owords of pixels. */
	rgba = _mm256_permute4x64_epi64(rgba, SWIZZLE(0, 2, 1, 3));
	__m256i mask = _mm256_permute4x64_epi64(t->mask_q1, SWIZZLE(0, 2, 1, 3));

	void *base = ymajor_offset(args->rt.pixels, x, y, args->rt.stride, cpp);

	_mm_maskstore_epi32(base,
			    _mm256_extractf128_si256(mask, 0),
			    _mm256_extractf128_si256(rgba, 0));
	_mm_maskstore_epi32(base + 16,
			    _mm256_extractf128_si256(mask, 1),
			    _mm256_extractf128_si256(rgba, 1));
}

void *
builder_emit_sfid_render_cache_helper(struct builder *bld,
				      uint32_t opcode, uint32_t type, uint32_t src, uint32_t surface)
{
	struct sfid_render_cache_args *args;
	bool rt_valid;

	args = builder_get_const_data(bld, sizeof *args, 32);
	args->src = src;

	rt_valid = get_surface(bld->binding_table_address, surface, &args->rt);
	ksim_assert(rt_valid);
	if (!rt_valid)
		return NULL;

	builder_emit_load_rsi_rip_relative(bld, builder_offset(bld, args));

	/* vol 2d, p445 */
	switch (opcode) {
	case 12: /* rt write */
		switch (type) {
		case 0:
			return sfid_render_cache_rt_write_simd16;
		case 1: /* rep16 */
			if (args->rt.format == SF_B8G8R8A8_UNORM &&
			    args->rt.tile_mode == XMAJOR)
				return sfid_render_cache_rt_write_rep16_bgra_unorm8_xmajor;
			else if (args->rt.format == SF_B8G8R8A8_UNORM_SRGB &&
				 args->rt.tile_mode == XMAJOR)
				return sfid_render_cache_rt_write_rep16_bgra_unorm8_xmajor;
			else if (args->rt.format == SF_B8G8R8X8_UNORM &&
				 args->rt.tile_mode == XMAJOR)
				return sfid_render_cache_rt_write_rep16_bgra_unorm8_xmajor;
			else if (args->rt.format == SF_B8G8R8X8_UNORM_SRGB &&
				 args->rt.tile_mode == XMAJOR)
				return sfid_render_cache_rt_write_rep16_bgra_unorm8_xmajor;
			else if (args->rt.format == SF_R8G8B8A8_UNORM &&
				 args->rt.tile_mode == YMAJOR)
				return sfid_render_cache_rt_write_rep16_rgba_unorm8_ymajor;
			else
				stub("rep16 rt write format/tile_mode: %d %d",
				     args->rt.format, args->rt.tile_mode);

		case 4: /* simd8 */
			if (args->rt.format == SF_R16G16B16A16_UNORM &&
			    args->rt.tile_mode == LINEAR)
				return sfid_render_cache_rt_write_simd8_rgba_unorm16_linear;
			else if (args->rt.format == SF_R8G8B8A8_UNORM &&
				 args->rt.tile_mode == LINEAR)
				return sfid_render_cache_rt_write_simd8_rgba_unorm8_linear;
			else if (args->rt.format == SF_R8G8B8A8_UINT &&
				 args->rt.tile_mode == LINEAR)
				return sfid_render_cache_rt_write_simd8_rgba_uint8_linear;
			else if (args->rt.format == SF_B8G8R8A8_UNORM &&
				 args->rt.tile_mode == XMAJOR)
				return sfid_render_cache_rt_write_simd8_bgra_unorm8_xmajor;
			else if (args->rt.format == SF_B8G8R8X8_UNORM &&
				 args->rt.tile_mode == XMAJOR)
				return sfid_render_cache_rt_write_simd8_bgra_unorm8_xmajor;
			else if (args->rt.format == SF_B8G8R8A8_UNORM_SRGB &&
				 args->rt.tile_mode == XMAJOR)
				return sfid_render_cache_rt_write_simd8_bgra_unorm8_xmajor;
			else if (args->rt.format == SF_R32G32B32A32_UINT &&
				 args->rt.tile_mode == LINEAR)
				return sfid_render_cache_rt_write_simd8_rgba_uint32_linear;
			else if (args->rt.format == SF_R16G16B16A16_UINT &&
				 args->rt.tile_mode == LINEAR)
				return sfid_render_cache_rt_write_simd8_rgba_uint16_linear;
			else if (args->rt.format == SF_R8_UINT &&
				 args->rt.tile_mode == YMAJOR)
				return sfid_render_cache_rt_write_simd8_r_uint8_ymajor;
			else if (args->rt.format == SF_R8G8B8A8_UNORM &&
				 args->rt.tile_mode == YMAJOR)
				return sfid_render_cache_rt_write_simd8_unorm8_ymajor;
			else if (args->rt.format == SF_B8G8R8A8_UNORM &&
				 args->rt.tile_mode == YMAJOR)
				return sfid_render_cache_rt_write_simd8_unorm8_ymajor;
			else
				stub("simd8 rt write format/tile_mode: %d %d",
				     args->rt.format, args->rt.tile_mode);
			break;
		default:
			stub("rt write type %d", type);
			return NULL;
		}
	default:
		stub("render cache message opcode %d", opcode);
		return NULL;
	}
}

void
builder_emit_sfid_render_cache(struct builder *bld, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	uint32_t opcode = field(send.function_control, 14, 17);
	uint32_t type = field(send.function_control, 8, 10);
	uint32_t surface = field(send.function_control, 0, 7);
	uint32_t src = unpack_inst_2src_src0(inst).num;
	void *p;

	p = builder_emit_sfid_render_cache_helper(bld, opcode, type, src, surface);
	if (p == NULL)
		builder_emit_ret(bld);

	/* In case of eot, we end the thread by jumping
	 * (instead of calling) to the sfid implementation.
	 * When the sfid implementation returns it will return
	 * to our caller when it's done (tail-call
	 * optimization).
	 */
	if (send.eot) {
		builder_emit_jmp_relative(bld, (uint8_t *) p - bld->p);
	} else {
		builder_emit_call(bld, p);
	}
}
