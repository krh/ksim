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

struct vs_thread {
	struct thread t;
	struct reg vid;
	void *index_buffer;
	uint32_t iid;
	uint32_t start_vertex;
	uint32_t base_vertex;
	uint32_t start_instance;
	struct vue_buffer buffer;
};

struct ia_state {
	struct value *vue[16];
	uint32_t head, tail;
	int tristrip_parity;
	struct value *trifan_first_vertex;
};

static void
flush_to_vues(struct vue_buffer *b, uint32_t count, struct ia_state *s)
{
	/* Transpose the SIMD8 vs_thread back into individual VUEs */
	for (uint32_t c = 0; c < count; c++) {
		__m256i *vue = urb_handle_to_entry(b->vue_handles.ud[c]);
		__m256i offsets = (__m256i) (__v8si) { 0, 8, 16, 24, 32, 40, 48, 56 };
		for (uint32_t i = 0; i < gt.vs.urb.size / 32; i++)
			vue[i] = _mm256_i32gather_epi32(&b->data[i * 8].d[c], offsets, 4);


		s->vue[s->head++ & 15] = (struct value *) vue;
 	}

	ksim_assert(s->head - s->tail < 16);
}

static void
dispatch_vs(struct vs_thread *t, uint32_t iid, uint32_t vid, struct ia_state *state)
{
	struct reg *grf = &t->t.grf[0];

	/* Not sure what we should make this. */
	uint32_t fftid = 0;

	uint32_t rest = gt.prim.vertex_count - vid;
	uint32_t count;
	if (rest > 8)
		count = 8;
	else
		count = rest;

	static const struct reg range = { .d = {  0, 1, 2, 3, 4, 5, 6, 7 } };
	t->t.mask_q1 = _mm256_sub_epi32(range.ireg, _mm256_set1_epi32(rest));
	t->iid = iid;
	t->vid.ireg = _mm256_add_epi32(range.ireg, _mm256_set1_epi32(vid));

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

	for (uint32_t c = 0; c < count; c++) {
		void *entry = alloc_urb_entry(&gt.vs.urb);
		t->buffer.vue_handles.ud[c] = urb_entry_to_handle(entry);
	}

	grf[1].ireg = t->buffer.vue_handles.ireg;

	if (gt.vs.statistics)
		gt.vs_invocation_count++;

	gt.vs.avx_shader(&t->t);

	flush_to_vues(&t->buffer, count, state);
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

void
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
assemble_primitives(struct ia_state *s)
{
	struct value *vue[32];
	uint32_t tail = s->tail;
	int count;

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

	case _3DPRIM_LINELIST:
		while (s->head - tail >= 2) {
			vue[0] = s->vue[(tail + 0) & 15];
			vue[1] = s->vue[(tail + 1) & 15];
			setup_prim(vue, 0);
			tail += 2;
		}
		break;

	case _3DPRIM_LINESTRIP:
		while (s->head - tail >= 2) {
			vue[0] = s->vue[(tail + 0) & 15];
			vue[1] = s->vue[(tail + 1) & 15];
			setup_prim(vue, 0);
			tail += 1;
		}
		break;

	case _3DPRIM_LINELOOP:
		if (s->trifan_first_vertex == NULL) {
			/* We always have at least one vertex
			 * when we get, so this is safe. */
			ksim_assert(s->head - tail >= 1);
			s->trifan_first_vertex = s->vue[tail & 15];
			/* Bump the queue tail now so we don't free
			 * the vue below */
			s->tail++;
		}

		while (s->head - tail >= 2) {
			vue[0] = s->vue[(tail + 0) & 15];
			vue[1] = s->vue[(tail + 1) & 15];
			setup_prim(vue, 0);
			tail += 1;
		}
		break;

	case _3DPRIM_PATCHLIST_1 ... _3DPRIM_PATCHLIST_32:
		/* FIXME: Need to bump s->vue queue size to accomodate
		 * big patchlist primitives. */
		count = gt.ia.topology - _3DPRIM_PATCHLIST_1 + 1;
		while (s->head - tail >= count) {
			for (uint32_t i = 0; i < count; i++)
				vue[i] = s->vue[(tail + i) & 15];
			tessellate_patch(vue);
			tail += count;
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
	struct value *vue[3];
	uint32_t tail = s->tail;

	switch (gt.ia.topology) {
	case _3DPRIM_LINELOOP:
		if (s->head - tail == 1) {
			vue[0] = s->vue[(tail + 0) & 15];
			vue[1] = s->trifan_first_vertex;
			setup_prim(vue, 0);
			gt.ia_primitives_count++;
		}
		break;
	default:
		break;
	}

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
dump_vue(struct vs_thread *t)
{
	struct vue_buffer *b = &t->buffer;
	struct reg v = t->vid;

	ksim_trace(TRACE_VF, "Loaded vue for idd=%d, vid=[", t->iid);
	for (uint32_t c = 0; c < 8; c++)
		ksim_trace(TRACE_VF, " %d", v.ud[c]);
	ksim_trace(TRACE_VF, " ], mask=[");
	v.ireg = t->t.mask_q1;
	for (uint32_t c = 0; c < 8; c++)
		ksim_trace(TRACE_VF, " %d", v.ud[c]);
	ksim_trace(TRACE_VF, " ]\n");

	for (uint32_t i = 0; i < gt.vs.urb.size / 4; i++) {
		ksim_trace(TRACE_VF, "  0x%04x:  ", (void *) &b->data[i] - (void *) t);
		for (uint32_t c = 0; c < 8; c++)
			ksim_trace(TRACE_VF, "  %8.2f", b->data[i].f[c]);
		ksim_trace(TRACE_VF, "\n");
	}
}

static struct kir_reg
emit_gather(struct kir_program *prog,
	    struct kir_reg base, struct kir_reg offset,
	    uint32_t scale, uint32_t base_offset)
{
	/* This little gather helper loads the mask, gathers and then
	 * invalidates and releases the mask register. vpgatherdd
	 * writes 0 to the mask register, so we need to reload it for
	 * each gather and make sure we don't reuse the mask
	 * register. */

	struct kir_reg mask = kir_program_load_v8(prog, offsetof(struct vs_thread, t.mask_q1));

	return kir_program_gather(prog, base, offset, mask, scale, base_offset);
}

static void
emit_load_format_simd8(struct kir_program *prog, enum GEN9_SURFACE_FORMAT format,
		       struct kir_reg base, struct kir_reg offset, struct kir_reg *dst)
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
		dst[0] = emit_gather(prog, base, offset, 1, 0);
		dst[1] = emit_gather(prog, base, offset, 1, 4);
		dst[2] = emit_gather(prog, base, offset, 1, 8);
		dst[3] = emit_gather(prog, base, offset, 1, 12);
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

	struct kir_reg vid = kir_program_load_v8(prog, offsetof(struct vs_thread, vid));
	if (gt.prim.start_vertex > 0) {
		kir_program_load_uniform(prog, offsetof(struct vs_thread, start_vertex));
		vid = kir_program_alu(prog, kir_addd, vid, prog->dst);
	}

	if (gt.prim.access_type == RANDOM) {
		kir_program_comment(prog, "vertex fetch: index buffer fetch");

		/* FIXME: INDEX_BYTE and INDEX_WORD can read outside
		 * the index buffer. */
		uint64_t range;
		void *index_buffer = map_gtt_offset(gt.vf.ib.address, &range);
		struct kir_reg dst, base;

		base = kir_program_set_load_base_imm(prog, index_buffer);
		switch (gt.vf.ib.format) {
		case INDEX_BYTE:
			dst = emit_gather(prog, base, vid, 1, 0);
			dst = kir_program_alu(prog, kir_shli, dst, 24);
			dst = kir_program_alu(prog, kir_shri, dst, 24);
			break;
		case INDEX_WORD:
			dst = emit_gather(prog, base, vid, 2, 0);
			dst = kir_program_alu(prog, kir_shli, dst, 16);
			dst = kir_program_alu(prog, kir_shri, dst, 16);
			break;
		case INDEX_DWORD:
			dst = emit_gather(prog, base, vid, 4, 0);
			break;
		}

		if (gt.prim.base_vertex > 0) {
			kir_program_load_uniform(prog, offsetof(struct vs_thread, base_vertex));
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
				index = kir_program_load_uniform(prog, offsetof(struct vs_thread, iid));
			} else {
				index = kir_program_load_uniform(prog, offsetof(struct vs_thread, iid));
			}
			if (gt.prim.start_instance > 0) {
				kir_program_load_uniform(prog, offsetof(struct vs_thread, start_instance));
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
		struct kir_reg base = kir_program_set_load_base_imm(prog, vb->data);
		emit_load_format_simd8(prog, ve->format, base, offset, dst);

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
			kir_program_store_v8(prog, offsetof(struct vs_thread,
							    buffer.data[i * 4 + c]), src);
		}
	}

	if (gt.vf.iid_enable || gt.vf.vid_enable) {
		kir_program_comment(prog, "vertex fetch: system generated values");
		if (gt.vf.iid_enable) {
			kir_program_load_uniform(prog, offsetof(struct vs_thread, iid));
			uint32_t reg = gt.vf.iid_element * 4 + gt.vf.iid_component;
			kir_program_store_v8(prog, offsetof(struct vs_thread,
							    buffer.data[reg]), prog->dst);
		}
		if (gt.vf.vid_enable) {
			kir_program_load_v8(prog, offsetof(struct vs_thread, vid));
			uint32_t reg = gt.vf.vid_element * 4 + gt.vf.vid_component;
			kir_program_store_v8(prog, offsetof(struct vs_thread,
							    buffer.data[reg]), prog->dst);
		}
	}

	if (trace_mask & TRACE_URB)
		kir_program_call(prog, dump_vue, 0);
}

static void
emit_load_vue(struct kir_program *prog, uint32_t grf)
{
	uint32_t src = offsetof(struct vs_thread, buffer.data[gt.vs.vue_read_offset * 2 * 4]);
	uint32_t dst = offsetof(struct vs_thread, t.grf[grf]);

	kir_program_comment(prog, "copy vue");
	for (uint32_t i = 0; i < gt.vs.vue_read_length * 2 * 4; i++) {
		kir_program_load_v8(prog, src + i * 32);
		kir_program_store_v8(prog, dst + i * 32, prog->dst);
	}
}

#define vue_offset(base, field) ((base) + offsetof(struct vue_buffer, field))

static void
emit_perspective_divide(struct kir_program *prog, uint32_t base)
{
	/* vrcpps doesn't have sufficient precision for perspective
	 * divide. We can use vdivps (latency 21/throughput 13) or do
	 * a Newton-Raphson step on vrcpps.  This turns into vrcpps,
	 * vfnmadd213ps and vmulps, with latencies 7, 5 and 5, which
	 * is slightly better.
	 */

	kir_program_comment(prog, "perspective divide");

	struct kir_reg w = kir_program_load_v8(prog, vue_offset(base, w));
	struct kir_reg inv_w0 = kir_program_alu(prog, kir_rcp, w);

	/* NR step: inv_w = inv_w0 * (2 - w * inv_w0) */
	struct kir_reg two = kir_program_immf(prog, 2.0f);

	kir_program_alu(prog, kir_nmaddf, w, inv_w0, two);
	struct kir_reg inv_w = kir_program_alu(prog, kir_mulf, inv_w0, prog->dst);

	const struct kir_reg x = kir_program_load_v8(prog, vue_offset(base, x));
	kir_program_alu(prog, kir_mulf, x, inv_w);
	kir_program_store_v8(prog, vue_offset(base, x), prog->dst);

	const struct kir_reg y = kir_program_load_v8(prog, vue_offset(base, y));
	kir_program_alu(prog, kir_mulf, y, inv_w);
	kir_program_store_v8(prog, vue_offset(base, y), prog->dst);

	const struct kir_reg z = kir_program_load_v8(prog, vue_offset(base, z));
	kir_program_alu(prog, kir_mulf, z, inv_w);
	kir_program_store_v8(prog, vue_offset(base, z), prog->dst);

	kir_program_store_v8(prog, vue_offset(base, w), inv_w);
}

static void
emit_clip_test(struct kir_program *prog, uint32_t base)
{
	kir_program_comment(prog, "clip tests");

	struct kir_reg x0 = kir_program_load_uniform(prog, vue_offset(base, clip.x0));
	struct kir_reg x1 = kir_program_load_uniform(prog, vue_offset(base, clip.x1));
	struct kir_reg y0 = kir_program_load_uniform(prog, vue_offset(base, clip.y0));
	struct kir_reg y1 = kir_program_load_uniform(prog, vue_offset(base, clip.y1));
	struct kir_reg x = kir_program_load_v8(prog, vue_offset(base, x));
	struct kir_reg y = kir_program_load_v8(prog, vue_offset(base, y));

	struct kir_reg x0f = kir_program_alu(prog, kir_cmp, x0, x, _CMP_LT_OS);
	struct kir_reg x1f = kir_program_alu(prog, kir_cmp, x1, x, _CMP_GT_OS);
	struct kir_reg y0f = kir_program_alu(prog, kir_cmp, y0, y, _CMP_LT_OS);
	struct kir_reg y1f = kir_program_alu(prog, kir_cmp, y1, y, _CMP_GT_OS);

	struct kir_reg xf = kir_program_alu(prog, kir_or, x0f, x1f);
	struct kir_reg yf = kir_program_alu(prog, kir_or, y0f, y1f);
	struct kir_reg f = kir_program_alu(prog, kir_or, xf, yf);

	kir_program_store_v8(prog, vue_offset(base, clip_flags), f);
}

static void
emit_viewport_transform(struct kir_program *prog, uint32_t base)
{
	kir_program_comment(prog, "viewport transform");

	struct kir_reg m00 = kir_program_load_uniform(prog, vue_offset(base, vp.m00));
	struct kir_reg m11 = kir_program_load_uniform(prog, vue_offset(base, vp.m11));
	struct kir_reg m22 = kir_program_load_uniform(prog, vue_offset(base, vp.m22));
	struct kir_reg m30 = kir_program_load_uniform(prog, vue_offset(base, vp.m30));
	struct kir_reg m31 = kir_program_load_uniform(prog, vue_offset(base, vp.m31));
	struct kir_reg m32 = kir_program_load_uniform(prog, vue_offset(base, vp.m32));

	struct kir_reg x = kir_program_load_v8(prog, vue_offset(base, x));
	struct kir_reg y = kir_program_load_v8(prog, vue_offset(base, y));
	struct kir_reg z = kir_program_load_v8(prog, vue_offset(base, z));

	struct kir_reg xs = kir_program_alu(prog, kir_maddf, x, m00, m30);
	struct kir_reg ys = kir_program_alu(prog, kir_maddf, y, m11, m31);
	struct kir_reg zs = kir_program_alu(prog, kir_maddf, z, m22, m32);

	kir_program_store_v8(prog, vue_offset(base, x), xs);
	kir_program_store_v8(prog, vue_offset(base, y), ys);
	kir_program_store_v8(prog, vue_offset(base, z), zs);
}

void
emit_vertex_post_processing(struct kir_program *prog, uint32_t base)
{
	if (!gt.clip.perspective_divide_disable)
		emit_perspective_divide(prog, base);

	if (gt.clip.guardband_clip_test_enable ||
	    gt.clip.viewport_clip_test_enable)
		emit_clip_test(prog, base);

	if (gt.sf.viewport_transform_enable)
		emit_viewport_transform(prog, base);
}

static void
compile_vs(void)
{
	struct kir_program prog;

	ksim_trace(TRACE_EU | TRACE_AVX, "jit vs\n");

	kir_program_init(&prog,
			 gt.vs.binding_table_address,
			 gt.vs.sampler_state_address);

	prog.urb_offset = offsetof(struct vs_thread, buffer.data);

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

	if (!gt.gs.enable && !gt.hs.enable)
		emit_vertex_post_processing(&prog,
					    offsetof(struct vs_thread, buffer));

	if (trace_mask & TRACE_URB)
		kir_program_call(&prog, dump_vue, 0);

	kir_program_add_insn(&prog, kir_eot);

	gt.vs.avx_shader = kir_program_finish(&prog);
}

void
init_vue_buffer(struct vue_buffer *b)
{
	if (gt.clip.guardband_clip_test_enable) {
		b->clip = gt.sf.guardband;
	} else {
		struct rectanglef vp_clip = { -1.0f, -1.0f, 1.0f, 1.0f };
		b->clip = vp_clip;
	}

	if (gt.sf.viewport_transform_enable) {
		const float *vp = gt.sf.viewport;

		b->vp.m00 = vp[0];
		b->vp.m11 = vp[1];
		b->vp.m22 = vp[2];
		b->vp.m30 = vp[3];
		b->vp.m31 = vp[4];
		b->vp.m32 = vp[5];
	}
}

static void
init_vs_thread(struct vs_thread *t)
{
	t->start_vertex = gt.prim.start_vertex;
	t->base_vertex = gt.prim.base_vertex;
	t->start_instance = gt.prim.start_instance;

	init_vue_buffer(&t->buffer);

	load_constants(&t->t, &gt.vs.curbe);
}

void
dispatch_primitive(void)
{
	validate_vf_state();

	validate_urb_state();

	if (gt.sf.viewport_transform_enable)
		dump_sf_clip_viewport();

	ksim_assert(gt.vs.simd8 || !gt.vs.enable);

	if (gt.ia.topology < _3DPRIM_PATCHLIST_1)
		ksim_assert(!gt.hs.enable && !gt.ds.enable && !gt.te.enable);
	else
		ksim_assert(gt.hs.enable && gt.ds.enable && gt.te.enable);
	if (gt.hs.enable) {
		ksim_assert(gt.hs.dispatch_mode == DISPATCH_MODE_SINGLE_PATCH);
		ksim_assert(gt.te.domain == TRI);
		ksim_assert(gt.te.topology == OUTPUT_TRI_CW ||
			    gt.te.topology == OUTPUT_TRI_CCW);
		ksim_assert(gt.ds.dispatch_mode == DISPATCH_MODE_SIMD8_SINGLE_PATCH);
	}

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
	compile_hs();
	compile_ds();
	compile_ps();

	struct vs_thread t;

	init_vs_thread(&t);

	struct ia_state state = { 0, };
	for (uint32_t iid = 0; iid < gt.prim.instance_count; iid++) {
		for (uint32_t i = 0; i < gt.prim.vertex_count; i += 8) {
			dispatch_vs(&t, iid, i, &state);
			assemble_primitives(&state);
		}

		reset_ia_state(&state);
	}

	if (gt.vf.statistics)
		gt.ia_vertices_count +=
			gt.prim.vertex_count * gt.prim.instance_count;

	wm_flush();
}
