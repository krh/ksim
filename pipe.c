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
#include "kir.h"

static void
dispatch_vs(struct vf_buffer *buffer, __m256i mask)
{
	struct reg *grf = &buffer->t.grf[0];

	/* Not sure what we should make this. */
	uint32_t fftid = 0;

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

	const struct reg m = { .ireg = mask };
	for (uint32_t c = 0; c < 8; c++) {
		if (m.d[c] >= 0)
			continue;
		void *entry = alloc_urb_entry(&gt.vs.urb);
		buffer->vue_handles.ud[c] = urb_entry_to_handle(entry);
	}

	grf[1].ireg = buffer->vue_handles.ireg;

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
dump_vue(struct vf_buffer *buffer)
{
	struct reg v = buffer->vid;
	ksim_trace(TRACE_VF, "Loaded vue for idd=%d, vid=[", buffer->iid);
	for (uint32_t c = 0; c < 8; c++)
		ksim_trace(TRACE_VF, " %d", v.ud[c]);
	ksim_trace(TRACE_VF, " ], mask=[");
	v.ireg = buffer->t.mask_q1;
	for (uint32_t c = 0; c < 8; c++)
		ksim_trace(TRACE_VF, " %d", v.ud[c]);
	ksim_trace(TRACE_VF, " ]\n");

	for (uint32_t i = 0; i < gt.vs.urb.size / 4; i++) {
		ksim_trace(TRACE_VF, "  0x%04x:  ", (void *) &buffer->data[i] - (void *) buffer);
		for (uint32_t c = 0; c < 8; c++)
			ksim_trace(TRACE_VF, "  %8.2f", buffer->data[i].f[c]);
		ksim_trace(TRACE_VF, "\n");
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

static struct kir_reg
emit_gather(struct kir_program *prog, void *base, struct kir_reg offset,
	    uint32_t scale, uint32_t base_offset)
{
	/* This little gather helper loads the mask, gathers and then
	 * invalidates and releases the mask register. vpgatherdd
	 * writes 0 to the mask register, so we need to reload it for
	 * each gather and make sure we don't reuse the mask
	 * register. */

	struct kir_reg mask = kir_program_load_v8(prog, offsetof(struct vf_buffer, t.mask_q1));

	return kir_program_gather(prog, base, offset, mask, scale, base_offset);
}

static void
emit_load_format_simd8(struct kir_program *prog, enum GEN9_SURFACE_FORMAT format,
		       void *base, struct kir_reg offset, struct kir_reg *dst)
{
	switch (format) {
	case SF_R32_FLOAT:
	case SF_R32_SINT:
	case SF_R32_UINT:
		dst[0] = emit_gather(prog, base, offset, 1, 0);
		dst[1] = kir_program_immd(prog, 0);
		dst[2] = kir_program_immd(prog, 0);
		dst[3] = kir_program_immf(prog, 1.0f);
		break;

	case SF_R32G32_FLOAT:
	case SF_R32G32_SINT:
	case SF_R32G32_UINT:
		dst[0] = emit_gather(prog, base, offset, 1, 0);
		dst[1] = emit_gather(prog, base, offset, 1, 4);
		dst[2] = kir_program_immd(prog, 0);
		dst[3] = kir_program_immf(prog, 1.0f);
		break;

	case SF_R32G32B32_FLOAT:
	case SF_R32G32B32_SINT:
	case SF_R32G32B32_UINT:
		dst[0] = emit_gather(prog, base, offset, 1, 0);
		dst[1] = emit_gather(prog, base, offset, 1, 4);
		dst[2] = emit_gather(prog, base, offset, 1, 8);
		dst[3] = kir_program_immf(prog, 1.0f);
		break;

	case SF_R32G32B32A32_FLOAT:
	case SF_R32G32B32A32_SINT:
	case SF_R32G32B32A32_UINT:
		dst[0] = emit_gather(prog, base,offset, 1, 0);
		dst[1] = emit_gather(prog, base,offset, 1, 4);
		dst[2] = emit_gather(prog, base,offset, 1, 8);
		dst[3] = emit_gather(prog, base,offset, 1, 12);
		break;

	default:
		stub("fetch format");
		ksim_unreachable("fetch_format");
		break;
	}
}

static void
emit_vertex_fetch(struct kir_program *prog)
{
	kir_program_comment(prog, "vertex fetch");

	struct kir_reg vid = kir_program_load_v8(prog, offsetof(struct vf_buffer, vid));
	if (gt.prim.start_vertex > 0) {
		kir_program_load_uniform(prog, offsetof(struct vf_buffer, start_vertex));
		vid = kir_program_alu(prog, kir_addd, vid, prog->dst);
	}

	if (gt.prim.access_type == RANDOM) {
		kir_program_comment(prog, "vertex fetch: index buffer fetch");

		/* FIXME: INDEX_BYTE and INDEX_WORD can read outside
		 * the index buffer. */
		uint64_t range;
		void *index_buffer = map_gtt_offset(gt.vf.ib.address, &range);
		struct kir_reg dst;

		switch (gt.vf.ib.format) {
		case INDEX_BYTE:
			dst = emit_gather(prog, index_buffer, vid, 1, 0);
			dst = kir_program_alu(prog, kir_shli, dst, 24);
			dst = kir_program_alu(prog, kir_shri, dst, 24);
			break;
		case INDEX_WORD:
			dst = emit_gather(prog, index_buffer, vid, 2, 0);
			dst = kir_program_alu(prog, kir_shli, dst, 16);
			dst = kir_program_alu(prog, kir_shri, dst, 16);
			break;
		case INDEX_DWORD:
			dst = emit_gather(prog, index_buffer, vid, 4, 0);
			break;
		}

		if (gt.prim.base_vertex > 0) {
			kir_program_load_uniform(prog, offsetof(struct vf_buffer, base_vertex));
			dst = kir_program_alu(prog, kir_addd, dst, prog->dst);
		}
		vid = dst;
	}

	for (uint32_t i = 0; i < gt.vf.ve_count; i++) {
		struct ve *ve = &gt.vf.ve[i];
		struct vb *vb = &gt.vf.vb[ve->vb];
		struct kir_reg index;

		if (!gt.vf.ve[i].valid)
			continue;

		kir_program_comment(prog, "vertex fetch: ve %d: offset %d, pitch %d, format %d, vb %p",
				    i, ve->offset, vb->pitch, ve->format, vb->data);

		if (gt.vf.ve[i].instancing) {
			if (gt.vf.ve[i].step_rate > 1) {
				/* FIXME: index = _mm256_set1_epi32(gt.prim.start_instance + iid / gt.vf.ve[i].step_rate); */
				stub("instancing step rate > 1");
				index = kir_program_load_uniform(prog, offsetof(struct vf_buffer, iid));
			} else {
				index = kir_program_load_uniform(prog, offsetof(struct vf_buffer, iid));
			}
			if (gt.prim.start_instance > 0) {
				kir_program_load_uniform(prog, offsetof(struct vf_buffer, start_instance));
				index = kir_program_alu(prog, kir_addd, index, prog->dst);
			}
		} else {
			index = vid;
		}

		struct kir_reg offset;
		if (vb->pitch == 0) {
			offset = kir_program_immd(prog, ve->offset);
		} else if (is_power_of_two(vb->pitch)) {
			uint32_t pitch_log2 = __builtin_ffs(vb->pitch) - 1;
			offset = kir_program_alu(prog, kir_shli, index, pitch_log2);
		} else if (vb->pitch % 3 == 0 && is_power_of_two(vb->pitch / 3)) {
			offset = kir_program_alu(prog, kir_shli, index, 1);
			offset = kir_program_alu(prog, kir_addd, offset, index);
			uint32_t pitch_log2 = __builtin_ffs(vb->pitch / 3) - 1;
			offset = kir_program_alu(prog, kir_shli, offset, pitch_log2);
		} else {
			kir_program_immd(prog, vb->pitch);
			offset = kir_program_alu(prog, kir_muld, index, prog->dst);
		}

		if (vb->pitch > 0 && ve->offset > 0) {
			kir_program_immd(prog, ve->offset);
			offset = kir_program_alu(prog, kir_addd, offset, prog->dst);
		}

		struct kir_reg dst[4];
		emit_load_format_simd8(prog, ve->format, vb->data, offset, dst);

		for (uint32_t c = 0; c < 4; c++) {
			struct kir_reg src;
			switch (ve->cc[c]) {
			case VFCOMP_NOSTORE:
				continue;
			case VFCOMP_STORE_SRC:
				src = dst[c];
				break;
			case VFCOMP_STORE_0:
				src = kir_program_immf(prog, 0.0f);
				break;
			case VFCOMP_STORE_1_FP:
				src = kir_program_immf(prog, 1.0f);
				break;
			case VFCOMP_STORE_1_INT:
				src = kir_program_immd(prog, 1);
				break;
			case VFCOMP_STORE_PID:
				ksim_unreachable("VFCOMP_STORE_PID");
				break;
			}
			kir_program_store_v8(prog, offsetof(struct vf_buffer, data[i * 4 + c]), src);
		}
	}

	if (gt.vf.iid_enable || gt.vf.vid_enable) {
		kir_program_comment(prog, "vertex fetch: system generated values");
		if (gt.vf.iid_enable) {
			kir_program_load_uniform(prog, offsetof(struct vf_buffer, iid));
			uint32_t reg = gt.vf.iid_element * 4 + gt.vf.iid_component;
			kir_program_store_v8(prog, offsetof(struct vf_buffer, data[reg]), prog->dst);
		}
		if (gt.vf.vid_enable) {
			kir_program_load_v8(prog, offsetof(struct vf_buffer, vid));
			uint32_t reg = gt.vf.vid_element * 4 + gt.vf.vid_component;
			kir_program_store_v8(prog, offsetof(struct vf_buffer, data[reg]), prog->dst);
		}
	}

	if (trace_mask & TRACE_URB)
		kir_program_call(prog, dump_vue, 0);
}

static void
emit_load_vue(struct kir_program *prog, uint32_t grf)
{
	uint32_t src = offsetof(struct vf_buffer, data[gt.vs.vue_read_offset * 2 * 4]);
	uint32_t dst = offsetof(struct vf_buffer, t.grf[grf]);

	kir_program_comment(prog, "copy vue");
	for (uint32_t i = 0; i < gt.vs.vue_read_length * 2 * 4; i++) {
		kir_program_load_v8(prog, src + i * 32);
		kir_program_store_v8(prog, dst + i * 32, prog->dst);
	}
}

static void
emit_perspective_divide(struct kir_program *prog)
{
	/* vrcpps doesn't have sufficient precision for perspective
	 * divide. We can use vdivps (latency 21/throughput 13) or do
	 * a Newton-Raphson step on vrcpps.  This turns into vrcpps,
	 * vfnmadd213ps and vmulps, with latencies 7, 5 and 5, which
	 * is slightly better.
	 */

	kir_program_comment(prog, "perspective divide");

	struct kir_reg w = kir_program_load_v8(prog, offsetof(struct vf_buffer, w));
	struct kir_reg inv_w0 = kir_program_alu(prog, kir_rcp, w);

	/* NR step: inv_w = inv_w0 * (2 - w * inv_w0) */
	struct kir_reg two = kir_program_immf(prog, 2.0f);

	kir_program_alu(prog, kir_nmaddf, w, inv_w0, two);
	struct kir_reg inv_w = kir_program_alu(prog, kir_mulf, inv_w0, prog->dst);

	const struct kir_reg x = kir_program_load_v8(prog, offsetof(struct vf_buffer, x));
	kir_program_alu(prog, kir_mulf, x, inv_w);
	kir_program_store_v8(prog, offsetof(struct vf_buffer, x), prog->dst);

	const struct kir_reg y = kir_program_load_v8(prog, offsetof(struct vf_buffer, y));
	kir_program_alu(prog, kir_mulf, y, inv_w);
	kir_program_store_v8(prog, offsetof(struct vf_buffer, y), prog->dst);

	const struct kir_reg z = kir_program_load_v8(prog, offsetof(struct vf_buffer, z));
	kir_program_alu(prog, kir_mulf, z, inv_w);
	kir_program_store_v8(prog, offsetof(struct vf_buffer, z), prog->dst);

	kir_program_store_v8(prog, offsetof(struct vf_buffer, w), inv_w);
}

static void
emit_clip_test(struct kir_program *prog)
{
	kir_program_comment(prog, "clip tests");

	struct kir_reg x0 = kir_program_load_uniform(prog, offsetof(struct vf_buffer, clip.x0));
	struct kir_reg x1 = kir_program_load_uniform(prog, offsetof(struct vf_buffer, clip.x1));
	struct kir_reg y0 = kir_program_load_uniform(prog, offsetof(struct vf_buffer, clip.y0));
	struct kir_reg y1 = kir_program_load_uniform(prog, offsetof(struct vf_buffer, clip.y1));
	struct kir_reg x = kir_program_load_v8(prog, offsetof(struct vf_buffer, x));
	struct kir_reg y = kir_program_load_v8(prog, offsetof(struct vf_buffer, y));

	struct kir_reg x0f = kir_program_alu(prog, kir_cmp, x0, x, _CMP_LT_OS);
	struct kir_reg x1f = kir_program_alu(prog, kir_cmp, x1, x, _CMP_GT_OS);
	struct kir_reg y0f = kir_program_alu(prog, kir_cmp, y0, y, _CMP_LT_OS);
	struct kir_reg y1f = kir_program_alu(prog, kir_cmp, y1, y, _CMP_GT_OS);

	struct kir_reg xf = kir_program_alu(prog, kir_or, x0f, x1f);
	struct kir_reg yf = kir_program_alu(prog, kir_or, y0f, y1f);
	struct kir_reg f = kir_program_alu(prog, kir_or, xf, yf);

	kir_program_store_v8(prog, offsetof(struct vf_buffer, clip_flags), f);
}

static void
emit_viewport_transform(struct kir_program *prog)
{
	kir_program_comment(prog, "viewport transform");

	struct kir_reg m00 = kir_program_load_uniform(prog, offsetof(struct vf_buffer, vp.m00));
	struct kir_reg m11 = kir_program_load_uniform(prog, offsetof(struct vf_buffer, vp.m11));
	struct kir_reg m22 = kir_program_load_uniform(prog, offsetof(struct vf_buffer, vp.m22));
	struct kir_reg m30 = kir_program_load_uniform(prog, offsetof(struct vf_buffer, vp.m30));
	struct kir_reg m31 = kir_program_load_uniform(prog, offsetof(struct vf_buffer, vp.m31));
	struct kir_reg m32 = kir_program_load_uniform(prog, offsetof(struct vf_buffer, vp.m32));

	struct kir_reg x = kir_program_load_v8(prog, offsetof(struct vf_buffer, x));
	struct kir_reg y = kir_program_load_v8(prog, offsetof(struct vf_buffer, y));
	struct kir_reg z = kir_program_load_v8(prog, offsetof(struct vf_buffer, z));

	struct kir_reg xs = kir_program_alu(prog, kir_maddf, x, m00, m30);
	struct kir_reg ys = kir_program_alu(prog, kir_maddf, y, m11, m31);
	struct kir_reg zs = kir_program_alu(prog, kir_maddf, z, m22, m32);

	kir_program_store_v8(prog, offsetof(struct vf_buffer, x), xs);
	kir_program_store_v8(prog, offsetof(struct vf_buffer, y), ys);
	kir_program_store_v8(prog, offsetof(struct vf_buffer, z), zs);
}

static void
compile_vs(void)
{
	struct kir_program prog;

	ksim_trace(TRACE_EU | TRACE_AVX, "jit vs\n");

	kir_program_init(&prog,
			 gt.vs.binding_table_address,
			 gt.vs.sampler_state_address);

	uint32_t grf;
	if (gt.vs.enable)
		grf = emit_load_constants(&prog, &gt.vs.curbe,
					  gt.vs.urb_start_grf);

	/* Always need to fetch, even if we don't have a VS. */
	emit_vertex_fetch(&prog);

	if (gt.vs.enable) {
		emit_load_vue(&prog, grf);

		kir_program_comment(&prog, "eu vs");
		kir_program_emit_shader(&prog, gt.vs.ksp);
	}

	if (trace_mask & TRACE_URB)
		kir_program_call(&prog, dump_vue, 0);

	if (!gt.clip.perspective_divide_disable)
		emit_perspective_divide(&prog);

	if (gt.clip.guardband_clip_test_enable ||
	    gt.clip.viewport_clip_test_enable)
		emit_clip_test(&prog);

	if (gt.sf.viewport_transform_enable)
		emit_viewport_transform(&prog);

	if (trace_mask & TRACE_URB)
		kir_program_call(&prog, dump_vue, 0);

	kir_program_add_insn(&prog, kir_eot);

	gt.vs.avx_shader = kir_program_finish(&prog);
}

static void
init_vf_buffer(struct vf_buffer *buffer)
{
	buffer->start_vertex = gt.prim.start_vertex;
	buffer->base_vertex = gt.prim.base_vertex;
	buffer->start_instance = gt.prim.start_instance;

	if (gt.clip.guardband_clip_test_enable) {
		buffer->clip = gt.sf.guardband;
	} else {
		struct rectanglef vp_clip = { -1.0f, -1.0f, 1.0f, 1.0f };
		buffer->clip = vp_clip;
	}

	if (gt.sf.viewport_transform_enable) {
		const float *vp = gt.sf.viewport;

		buffer->vp.m00 = vp[0];
		buffer->vp.m11 = vp[1];
		buffer->vp.m22 = vp[2];
		buffer->vp.m30 = vp[3];
		buffer->vp.m31 = vp[4];
		buffer->vp.m32 = vp[5];
	}

	load_constants(&buffer->t, &gt.vs.curbe);
}

void
dispatch_primitive(void)
{
	validate_vf_state();

	validate_urb_state();

	if (gt.sf.viewport_transform_enable)
		dump_sf_clip_viewport();

	ksim_assert(gt.vs.simd8 || !gt.vs.enable);

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

	reset_shader_pool();

	compile_vs();

	compile_ps();

	struct vf_buffer buffer;

	init_vf_buffer(&buffer);

	static const struct reg range = { .d = {  0, 1, 2, 3, 4, 5, 6, 7 } };
	struct ia_state state = { 0, };
	for (uint32_t iid = 0; iid < gt.prim.instance_count; iid++) {
		for (uint32_t i = 0; i < gt.prim.vertex_count; i += 8) {
			__m256i vid = _mm256_add_epi32(range.ireg, _mm256_set1_epi32(i));
			__m256i mask = _mm256_sub_epi32(range.ireg, _mm256_set1_epi32(gt.prim.vertex_count - i));

			buffer.iid = iid;
			buffer.vid.ireg = vid;

			dispatch_vs(&buffer, mask);
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
