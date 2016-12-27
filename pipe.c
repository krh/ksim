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
#include "avx-builder.h"

static void
dispatch_vs(struct vf_buffer *buffer, __m256i mask)
{
	struct reg *grf = &buffer->t.grf[0];
	uint32_t g;

	if (!gt.vs.enable)
		return;

	/* Not sure what we should make this. */
	uint32_t fftid = 0;

	buffer->t.mask = _mm256_movemask_ps((__m256) mask);
	buffer->t.mask_q1 = mask;

	/* Fixed function header */
	grf[0] = (struct reg) {
		.ud = {
			/* R0.0 - R0.2: MBZ */
			0,
			0,
			0,
			/* R0.3: per-thread scratch space, sampler ptr */
			gt.vs.sampler_state_address |
			gt.vs.scratch_size,
			/* R0.4: binding table pointer */
			gt.vs.binding_table_address,
			/* R0.5: fftid, scratch offset */
			gt.vs.scratch_pointer | fftid,
			/* R0.6: thread id */
			gt.vs.tid++ & 0xffffff,
			/* R0.7: Reserved */
			0,
		}
	};

	grf[1].ireg = buffer->vue_handles.ireg;

	g = load_constants(&buffer->t, &gt.vs.curbe, gt.vs.urb_start_grf);

	/* SIMD8 VS payload */
	__m256i *src = &buffer->data[gt.vs.vue_read_offset * 2 * 4].ireg;
	__m256i *dst = &grf[g].ireg;
	for (uint32_t i = 0; i < gt.vs.vue_read_length * 2 * 4; i++)
		dst[i] = src[i];

	if (gt.vs.statistics)
		gt.vs_invocation_count++;

	gt.vs.avx_shader(&buffer->t);
}

static void
validate_vf_state(void)
{
	uint32_t vb_used, b;
	uint64_t range;

	/* Make sure vue is big enough to hold all vertex elements */
	ksim_assert(gt.vf.ve_count * 16 <= gt.vs.urb.size);

	vb_used = 0;
	for (uint32_t i = 0; i < gt.vf.ve_count; i++) {
		ksim_assert((1 << gt.vf.ve[i].vb) & gt.vf.vb_valid);
		ksim_assert(valid_vertex_format(gt.vf.ve[i].format));
		if (gt.vf.ve[i].valid)
			vb_used |= 1 << gt.vf.ve[i].vb;
	}

	/* Check all VEs reference valid VBs. */
	ksim_assert((vb_used & gt.vf.vb_valid) == vb_used);

	for_each_bit(b, vb_used) {
		gt.vf.vb[b].data = map_gtt_offset(gt.vf.vb[b].address, &range);
		ksim_assert(gt.vf.vb[b].size <= range);
	}

	/* Check that SGVs are written within bounds */
	ksim_assert(gt.vf.iid_element * 16 < gt.vs.urb.size);
	ksim_assert(gt.vf.vid_element * 16 < gt.vs.urb.size);
}

static void
dump_sf_clip_viewport(void)
{
	const float *vp = gt.sf.viewport;

	spam("viewport matrix:\n");
	for (uint32_t i = 0; i < 6; i++)
		spam("  %20.4f\n", vp[i]);
	spam("guardband: %f,%f - %f,%f\n",
	     gt.sf.guardband.x0, gt.sf.guardband.y0,
	     gt.sf.guardband.x1, gt.sf.guardband.y1);
	spam("depth viewport: %f-%f\n",
	     gt.cc.viewport[0], gt.cc.viewport[1]);
}

static void
setup_prim(struct value **vue_in, uint32_t parity)
{
	struct value *vue[3];
	uint32_t provoking;

	for (int i = 0; i < 3; i++) {
		if (vue_in[i][0].header.clip_flags)
			return;
	}

	switch (gt.ia.topology) {
	case _3DPRIM_TRILIST:
	case _3DPRIM_TRISTRIP:
		provoking = gt.sf.tri_strip_provoking;
		break;
	case _3DPRIM_TRIFAN:
		provoking = gt.sf.tri_fan_provoking;
		break;
	case _3DPRIM_POLYGON:
	case _3DPRIM_QUADLIST:
	case _3DPRIM_QUADSTRIP:
	default:
		provoking = 0;
		break;
	case _3DPRIM_RECTLIST:
		/* The documentation requires a specific vertex
		 * ordering, but the hw doesn't actually care.  Our
		 * rasterizer does though, so rotate vertices to make
		 * sure the first to edges are axis parallel. */
		if (vue_in[0][1].vec4.x != vue_in[1][1].vec4.x &&
		    vue_in[0][1].vec4.y != vue_in[1][1].vec4.y) {
			ksim_warn("invalid rect list vertex order\n");
			provoking = 1;
		} else if (vue_in[1][1].vec4.x != vue_in[2][1].vec4.x &&
			   vue_in[1][1].vec4.y != vue_in[2][1].vec4.y) {
			ksim_warn("invalid rect list vertex order\n");
			provoking = 2;
		} else {
			provoking = 0;
		}
		break;
	}

	static const int indices[5] = { 0, 1, 2, 0, 1 };
	vue[0] = vue_in[indices[provoking]];
	vue[1] = vue_in[indices[provoking + 1 + parity]];
	vue[2] = vue_in[indices[provoking + 2 - parity]];

	rasterize_primitive(vue);
}

static void
transform_vertices(struct vf_buffer *buffer)
{
	if (!gt.clip.perspective_divide_disable) {
		/* vrcpps doesn't have sufficient precision for
		 * perspective divide. We can use vdivps (latency
		 * 21/throughput 13) or do a Newton-Raphson step on
		 * vrcpps.  This turns into vrcpps, vfnmadd213ps and
		 * vmulps, with latencies 7, 5 and 5, which is
		 * slightly better.
		 */
		__m256 inv_w0 = _mm256_rcp_ps(buffer->w);

		/* NR step: inv_w = inv_w0 * (2 - w * inv_w0) */
		__m256 inv_w = _mm256_mul_ps(inv_w0, _mm256_fnmadd_ps(buffer->w, inv_w0,
								      _mm256_set1_ps(2.0f)));

		buffer->x = _mm256_mul_ps(buffer->x, inv_w);
		buffer->y = _mm256_mul_ps(buffer->y, inv_w);
		buffer->z = _mm256_mul_ps(buffer->z, inv_w);
		buffer->w = inv_w;
 	}

	if (gt.clip.guardband_clip_test_enable ||
	    gt.clip.viewport_clip_test_enable) {
		__m256 x0, y0, x1, y1;
		if (gt.clip.guardband_clip_test_enable) {
			x0 = _mm256_set1_ps(gt.sf.guardband.x0);
			y0 = _mm256_set1_ps(gt.sf.guardband.y0);
			x1 = _mm256_set1_ps(gt.sf.guardband.x1);
			y1 = _mm256_set1_ps(gt.sf.guardband.y1);
		} else {
			x0 = _mm256_set1_ps(-1.0f);
			y0 = _mm256_set1_ps(-1.0f);
			x1 = _mm256_set1_ps(1.0f);
			y1 = _mm256_set1_ps(1.0f);
 		}

		__m256 l = _mm256_cmp_ps(buffer->x, x0, _CMP_LT_OS);
		__m256 r = _mm256_cmp_ps(buffer->x, x1, _CMP_GT_OS);
		__m256 t = _mm256_cmp_ps(buffer->y, y0, _CMP_LT_OS);
		__m256 b = _mm256_cmp_ps(buffer->y, y1, _CMP_GT_OS);

		buffer->clip_flags.ireg = 
			_mm256_or_si256(_mm256_or_si256(_mm256_castps_si256(l),
							_mm256_castps_si256(r)),
					_mm256_or_si256(_mm256_castps_si256(t),
							_mm256_castps_si256(b)));
	}

	if (gt.sf.viewport_transform_enable) {
		const float *vp = gt.sf.viewport;
		__m256 m00 = _mm256_set1_ps(vp[0]);
		__m256 m11 = _mm256_set1_ps(vp[1]);
		__m256 m22 = _mm256_set1_ps(vp[2]);
		__m256 m30 = _mm256_set1_ps(vp[3]);
		__m256 m31 = _mm256_set1_ps(vp[4]);
		__m256 m32 = _mm256_set1_ps(vp[5]);

		buffer->x = _mm256_fmadd_ps(m00, buffer->x, m30);
		buffer->y = _mm256_fmadd_ps(m11, buffer->y, m31);
		buffer->z = _mm256_fmadd_ps(m22, buffer->z, m32);
	}
}

struct ia_state {
	struct value *vue[16];
	uint32_t head, tail;
	int tristrip_parity;
	struct value *trifan_first_vertex;
};

static void
assemble_primitives(struct ia_state *s)
{
	struct value *vue[3];
	uint32_t tail = s->tail;

	switch (gt.ia.topology) {
	case _3DPRIM_TRILIST:
		while (s->head - tail >= 3) {
			vue[0] = s->vue[(tail + 0) & 15];
			vue[1] = s->vue[(tail + 1) & 15];
			vue[2] = s->vue[(tail + 2) & 15];
			setup_prim(vue, 0);
			tail += 3;
			gt.ia_primitives_count++;
		}
		break;

	case _3DPRIM_TRISTRIP:
		while (s->head - tail >= 3) {
			vue[0] = s->vue[(tail + 0) & 15];
			vue[1] = s->vue[(tail + 1) & 15];
			vue[2] = s->vue[(tail + 2) & 15];
			setup_prim(vue, s->tristrip_parity);
			tail += 1;
			s->tristrip_parity = 1 - s->tristrip_parity;
			gt.ia_primitives_count++;
		}
		break;

	case _3DPRIM_POLYGON:
	case _3DPRIM_TRIFAN:
		if (s->trifan_first_vertex == NULL) {
			/* We always have at least one vertex
			 * when we get, so this is safe. */
			ksim_assert(s->head - tail >= 1);
			s->trifan_first_vertex = s->vue[tail & 15];
			/* Bump the queue tail now so we don't free
			 * the vue below */
			s->tail++;
			tail++;
			gt.ia_primitives_count++;
		}

		while (s->head - tail >= 2) {
			vue[0] = s->trifan_first_vertex;
			vue[1] = s->vue[(tail + 0) & 15];
			vue[2] = s->vue[(tail + 1) & 15];
			setup_prim(vue, s->tristrip_parity);
			tail += 1;
			gt.ia_primitives_count++;
		}
		break;
	case _3DPRIM_QUADLIST:
		while (s->head - tail >= 4) {
			vue[0] = s->vue[(tail + 3) & 15];
			vue[1] = s->vue[(tail + 0) & 15];
			vue[2] = s->vue[(tail + 1) & 15];
			setup_prim(vue, 0);
			vue[0] = s->vue[(tail + 3) & 15];
			vue[1] = s->vue[(tail + 1) & 15];
			vue[2] = s->vue[(tail + 2) & 15];
			setup_prim(vue, 0);
			tail += 4;
			gt.ia_primitives_count++;
		}
		break;
	case _3DPRIM_QUADSTRIP:
		while (s->head - tail >= 4) {
			vue[0] = s->vue[(tail + 3) & 15];
			vue[1] = s->vue[(tail + 0) & 15];
			vue[2] = s->vue[(tail + 1) & 15];
			setup_prim(vue, 0);
			vue[0] = s->vue[(tail + 3) & 15];
			vue[1] = s->vue[(tail + 2) & 15];
			vue[2] = s->vue[(tail + 0) & 15];
			setup_prim(vue, 0);
			tail += 2;
			gt.ia_primitives_count++;
		}
		break;

	case _3DPRIM_RECTLIST:
		while (s->head - tail >= 3) {
			vue[0] = s->vue[(tail + 0) & 15];
			vue[1] = s->vue[(tail + 1) & 15];
			vue[2] = s->vue[(tail + 2) & 15];
			setup_prim(vue, 0);
			tail += 3;
		}
		break;

	default:
		stub("topology %d", gt.ia.topology);
		tail = s->head;
		break;
	}

	while (tail - s->tail > 0) {
		struct value *vue = s->vue[s->tail++ & 15];
		free_urb_entry(&gt.vs.urb, vue);
	}
}

void
reset_ia_state(struct ia_state *s)
{
	if (s->trifan_first_vertex) {
		free_urb_entry(&gt.vs.urb, s->trifan_first_vertex);
		s->trifan_first_vertex = NULL;
	}

	while (s->head - s->tail > 0) {
		struct value *vue = s->vue[s->tail++ & 15];
		free_urb_entry(&gt.vs.urb, vue);
	}

	s->head = 0;
	s->tail = 0;
	s->tristrip_parity = 0;
}

static void
fetch_vertices(struct vf_buffer *buffer, uint32_t iid, __m256i vid, __m256i mask)
{
	const __m256i zero = _mm256_setzero_si256();
	uint32_t a = 0;

	const struct reg m = { .ireg = mask };
	for (uint32_t c = 0; c < 8; c++) {
		if (m.d[c] >= 0)
			continue;
		void *entry = alloc_urb_entry(&gt.vs.urb);
		buffer->vue_handles.ud[c] = urb_entry_to_handle(entry);
	}

	__m256i vertex_index;
	if (gt.prim.access_type == RANDOM) {
		uint64_t range;
		void *ib = map_gtt_offset(gt.vf.ib.address, &range);

		vertex_index = _mm256_add_epi32(_mm256_set1_epi32(gt.prim.start_vertex), vid);

		switch (gt.vf.ib.format) {
			/* FIXME: INDEX_BYTE and INDEX_WORD
			 * can read outside the index
			 * buffer. */
		case INDEX_BYTE:
			vertex_index = _mm256_mask_i32gather_epi32(zero, ib, vertex_index, mask, 1);
			vertex_index = _mm256_and_si256(vertex_index, _mm256_set1_epi32(0xff));
			break;
		case INDEX_WORD:
			vertex_index = _mm256_mask_i32gather_epi32(zero, ib, vertex_index, mask, 2);
			vertex_index = _mm256_and_si256(vertex_index, _mm256_set1_epi32(0xffff));
			break;
		case INDEX_DWORD:
			vertex_index = _mm256_mask_i32gather_epi32(zero, ib, vertex_index, mask, 4);
			break;
		}
		vertex_index = _mm256_add_epi32(_mm256_set1_epi32(gt.prim.base_vertex), vertex_index);
	} else {
		vertex_index = _mm256_add_epi32(_mm256_set1_epi32(gt.prim.start_vertex), vid);
	}

	for (uint32_t i = 0; i < gt.vf.ve_count; i++) {
		struct ve *ve = &gt.vf.ve[i];
		struct vb *vb = &gt.vf.vb[ve->vb];
		__m256i index;

		if (!gt.vf.ve[i].valid)
			continue;

		if (gt.vf.ve[i].instancing) {
			index = _mm256_set1_epi32(gt.prim.start_instance + iid / gt.vf.ve[i].step_rate);
		} else {
			index = vertex_index;
		}

		__m256i offset =
			_mm256_add_epi32( _mm256_mullo_epi32(index, _mm256_set1_epi32(vb->pitch)),
					  _mm256_set1_epi32(ve->offset));

		load_format_simd8(vb->data, ve->format, offset, mask, &buffer->data[a]);

		for (uint32_t c = 0; c < 4; c++) {
			switch (ve->cc[c]) {
			case VFCOMP_STORE_0:
				buffer->data[a + c].ireg = _mm256_setzero_si256();
				break;
			case VFCOMP_STORE_1_FP:
				buffer->data[a + c].reg = _mm256_set1_ps(1.0f);
				break;
			case VFCOMP_STORE_1_INT:
				buffer->data[a + c].ireg = _mm256_set1_epi32(1);
				break;
			}
		}

		a += 4;

		/* edgeflag */
	}

	/* 3DSTATE_VF_SGVS */
	if (gt.vf.iid_enable) {
		a = gt.vf.iid_element * 4 + gt.vf.iid_component;
		buffer->data[a].ireg = _mm256_set1_epi32(iid);
	}
	if (gt.vf.vid_enable) {
		a = gt.vf.vid_element * 4 + gt.vf.vid_component;
		buffer->data[a].ireg = vid;
	}

	if (trace_mask & TRACE_VF) {
		struct reg v = { .ireg = vid };
		ksim_trace(TRACE_VF, "Loaded vue for idd=%d, vid=[", iid);
		for (uint32_t c = 0; c < 8; c++)
			ksim_trace(TRACE_VF, " %d", v.ud[c]);
		ksim_trace(TRACE_VF, " ]\n");

		uint32_t count = gt.vf.ve_count;
		if (gt.vf.iid_element + 1 > count)
			count = gt.vf.iid_element + 1;
		if (gt.vf.vid_element + 1 > count)
			count = gt.vf.vid_element + 1;
		for (uint32_t i = 0; i < count * 4; i++) {
			ksim_trace(TRACE_VF, "    ");
			for (uint32_t c = 0; c < 8; c++)
				ksim_trace(TRACE_VF, "  %8.2f", buffer->data[i].f[c]);
			ksim_trace(TRACE_VF, "\n");
		}
	}
}

static void
flush_to_vues(struct vf_buffer *buffer, __m256i mask, struct ia_state *s)
{
	/* Transpose the SIMD8 vf_buffer back into individual VUEs */
	const struct reg m = { .ireg = mask };
	for (uint32_t c = 0; c < 8; c++) {
		if (m.d[c] >= 0)
			continue;

		__m256i *vue = urb_handle_to_entry(buffer->vue_handles.ud[c]);
		__m256i offsets = (__m256i) (__v8si) { 0, 8, 16, 24, 32, 40, 48, 56 };
		for (uint32_t i = 0; i < gt.vs.urb.size / 32; i++)
			vue[i] = _mm256_i32gather_epi32(&buffer->data[i * 8].d[c], offsets, 4);

		s->vue[s->head++ & 15] = (struct value *) vue;
 	}

	ksim_assert(s->head - s->tail < 16);
}

static void
compile_vs(void)
{
	struct builder bld;

	ksim_trace(TRACE_EU | TRACE_AVX, "jit vs\n");

	builder_init(&bld,
		     gt.vs.binding_table_address,
		     gt.vs.sampler_state_address);

	if (gt.vs.enable)
		builder_emit_shader(&bld, gt.vs.ksp);

	builder_emit_ret(&bld);

	gt.vs.avx_shader = builder_finish(&bld);
}

void
dispatch_primitive(void)
{
	validate_vf_state();

	validate_urb_state();

	if (gt.sf.viewport_transform_enable)
		dump_sf_clip_viewport();

	ksim_assert(gt.vs.simd8 || !gt.vs.enable);

	prepare_shaders();

	compile_vs();

	gt.depth.write_enable =
		gt.depth.write_enable0 && gt.depth.write_enable1;

	/* Configure csr to round toward zero to make vcvtps2dq match
	 * the GEN EU behavior when converting from float to int. This
	 * may disagree with the rounding mode programmed in
	 * 3DSTATE_PS etc, which only affects rounding of internal
	 * intermediate float results. */
	const uint32_t csr_default =
		_MM_MASK_INVALID |
		_MM_MASK_DENORM |
		_MM_MASK_DIV_ZERO |
		_MM_MASK_OVERFLOW |
		_MM_MASK_UNDERFLOW |
		_MM_MASK_INEXACT |
		_MM_ROUND_TOWARD_ZERO;

	_mm_setcsr(csr_default);

	struct vf_buffer buffer;
	static const struct reg range = { .d = {  0, 1, 2, 3, 4, 5, 6, 7 } };
	struct ia_state state = { 0, };
	for (uint32_t iid = 0; iid < gt.prim.instance_count; iid++) {
		for (uint32_t i = 0; i < gt.prim.vertex_count; i += 8) {
			__m256i vid = _mm256_add_epi32(range.ireg, _mm256_set1_epi32(i));
			__m256i mask = _mm256_sub_epi32(range.ireg, _mm256_set1_epi32(gt.prim.vertex_count - i));
			fetch_vertices(&buffer, iid, vid, mask);
			dispatch_vs(&buffer, mask);
			transform_vertices(&buffer);
			flush_to_vues(&buffer, mask, &state);
			assemble_primitives(&state);
		}

		reset_ia_state(&state);
	}

	if (gt.vf.statistics)
		gt.ia_vertices_count +=
			gt.prim.vertex_count * gt.prim.instance_count;

	wm_flush();
}
