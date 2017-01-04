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
dump_fetched_vue(struct vf_buffer *buffer)
{
	struct reg v = buffer->vid;
	ksim_trace(TRACE_VF, "Loaded vue for idd=%d, vid=[", buffer->iid);
	for (uint32_t c = 0; c < 8; c++)
		ksim_trace(TRACE_VF, " %d", v.ud[c]);
	ksim_trace(TRACE_VF, " ]\n");

	uint32_t count = gt.vf.ve_count;
	if (gt.vf.iid_element + 1 > count)
		count = gt.vf.iid_element + 1;
	if (gt.vf.vid_element + 1 > count)
		count = gt.vf.vid_element + 1;
	for (uint32_t i = 0; i < count * 4; i++) {
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

static int
load_uniform(struct builder *bld, uint32_t offset)
{
	struct eu_region r = {
		.offset = offset,
		.type_size = 4,
		.exec_size = 1,
		.vstride = 0,
		.width = 1,
		.hstride = 0
	};

	return builder_emit_region_load(bld, &r);
}

static int
load_v8(struct builder *bld, uint32_t offset)
{
	struct eu_region r = {
		.offset = offset,
		.type_size = 4,
		.exec_size = 8,
		.vstride = 8,
		.width = 8,
		.hstride = 1
	};

	return builder_emit_region_load(bld, &r);
}

static void
store_v8(struct builder *bld, uint32_t offset, int reg)
{
	const struct eu_region r = {
		.offset = offset,
		.type_size = 4,
		.exec_size = 8,
		.vstride = 8,
		.width = 8,
		.hstride = 1
	};

	builder_emit_region_store(bld, &r, reg);
}

static int
emit_gather(struct builder *bld, int offset, uint32_t scale, uint32_t base_offset)
{
	/* This little gather helper loads the mask, gathers and then
	 * invalidates and releases the mask register. vpgatherdd
	 * writes 0 to the mask register, so we need to reload it for
	 * each gather and make sure we don't reuse the mask
	 * register. */

	int mask = load_v8(bld, offsetof(struct vf_buffer, t.mask_q1));
	int reg = builder_get_reg(bld);

	builder_emit_vpgatherdd(bld, reg, offset, mask, scale, base_offset);
	bld->regs[mask].contents = 0;
	builder_release_reg(bld, mask);

	return reg;
}

static void
emit_load_format_simd8(struct builder *bld, enum GEN9_SURFACE_FORMAT format,
		       void *base, int offset, int *dst)
{
	void **p = builder_get_const_data(bld, sizeof(*p), sizeof(*p));
	*p = base;
	builder_emit_load_rax_rip_relative(bld, builder_offset(bld, p));

	switch (format) {
	case SF_R32_FLOAT:
	case SF_R32_SINT:
	case SF_R32_UINT:
		dst[0] = emit_gather(bld, offset, 1, 0);
		dst[1] = builder_get_reg_with_uniform(bld, 0);
		dst[2] = builder_get_reg_with_uniform(bld, 0);
		dst[3] = builder_get_reg_with_uniform(bld, float_to_u32(1.0f));
		break;

	case SF_R32G32_FLOAT:
	case SF_R32G32_SINT:
	case SF_R32G32_UINT:
		dst[0] = emit_gather(bld, offset, 1, 0);
		dst[1] = emit_gather(bld, offset, 1, 4);
		dst[2] = builder_get_reg_with_uniform(bld, 0);
		dst[3] = builder_get_reg_with_uniform(bld, float_to_u32(1.0f));
		break;

	case SF_R32G32B32_FLOAT:
	case SF_R32G32B32_SINT:
	case SF_R32G32B32_UINT:
		dst[0] = emit_gather(bld, offset, 1, 0);
		dst[1] = emit_gather(bld, offset, 1, 4);
		dst[2] = emit_gather(bld, offset, 1, 8);
		dst[3] = builder_get_reg_with_uniform(bld, float_to_u32(1.0f));
		break;

	case SF_R32G32B32A32_FLOAT:
	case SF_R32G32B32A32_SINT:
	case SF_R32G32B32A32_UINT:
		dst[0] = emit_gather(bld, offset, 1, 0);
		dst[1] = emit_gather(bld, offset, 1, 4);
		dst[2] = emit_gather(bld, offset, 1, 8);
		dst[3] = emit_gather(bld, offset, 1, 12);
		break;

	default:
		stub("fetch format");
		ksim_unreachable("fetch_format");
		break;
	}
}

static void
builder_emit_vertex_fetch(struct builder *bld)
{
	ksim_trace(TRACE_EU, "[ff vertex fetch]\n");

	int vid = load_v8(bld, offsetof(struct vf_buffer, vid));
	if (gt.prim.start_vertex > 0) {
		int start_vertex = load_uniform(bld, offsetof(struct vf_buffer, start_vertex));
		builder_emit_vpaddd(bld, vid, vid, start_vertex);
	}

	if (gt.prim.access_type == RANDOM) {
		ksim_trace(TRACE_EU, "[ff vertex fetch: index buffer fetch]\n");

		/* FIXME: INDEX_BYTE and INDEX_WORD can read outside
		 * the index buffer. */
		int dst;
		builder_emit_load_rax(bld, offsetof(struct vf_buffer, index_buffer));
		switch (gt.vf.ib.format) {
		case INDEX_BYTE:
			dst = emit_gather(bld, vid, 0, 0);
			builder_emit_vpslld(bld, dst, dst, 24);
			builder_emit_vpsrld(bld, dst, dst, 24);
			break;
		case INDEX_WORD:
			dst = emit_gather(bld, vid, 2, 0);
			builder_emit_vpslld(bld, dst, dst, 16);
			builder_emit_vpsrld(bld, dst, dst, 16);
			break;
		case INDEX_DWORD:
			dst = emit_gather(bld, vid, 4, 0);
			break;
		}

		if (gt.prim.base_vertex > 0) {
			int base_vertex =
				load_uniform(bld, offsetof(struct vf_buffer, base_vertex));
			builder_emit_vpaddd(bld, dst, dst, base_vertex);
		}
		vid = dst;
	}

	builder_trace(bld, trace_file);
	builder_release_regs(bld);

	for (uint32_t i = 0; i < gt.vf.ve_count; i++) {
		struct ve *ve = &gt.vf.ve[i];
		struct vb *vb = &gt.vf.vb[ve->vb];
		int index;

		if (!gt.vf.ve[i].valid)
			continue;

		ksim_trace(TRACE_EU, "[ff vertex fetch: ve %d: offset %d, pitch %d, format %d]\n",
			   i, ve->offset, vb->pitch, ve->format);

		if (gt.vf.ve[i].instancing) {
			if (gt.vf.ve[i].step_rate > 1) {
				/* FIXME: index = _mm256_set1_epi32(gt.prim.start_instance + iid / gt.vf.ve[i].step_rate); */
				stub("instancing step rate > 1");
				index = load_uniform(bld, offsetof(struct vf_buffer, iid));
			} else {
				index = load_uniform(bld, offsetof(struct vf_buffer, iid));
			}
			if (gt.prim.start_instance > 0) {
				int start_instance = load_uniform(bld, offsetof(struct vf_buffer, start_instance));
				builder_emit_vpaddd(bld, index, index, start_instance);
			}
		} else {
			index = builder_use_reg(bld, &bld->regs[vid]);
		}

		int offset = builder_get_reg(bld);
		int element_offset;

		if (vb->pitch == 0) {
			offset = builder_get_reg_with_uniform(bld, ve->offset);
		} else if (is_power_of_two(vb->pitch)) {
			uint32_t pitch_log2 = __builtin_ffs(vb->pitch) - 1;
			builder_emit_vpslld(bld, offset, index, pitch_log2);
		} else if (vb->pitch % 3 == 0 && is_power_of_two(vb->pitch / 3)) {
			builder_emit_vpslld(bld, offset, index, 1);
			builder_emit_vpaddd(bld, offset, offset, index);
			uint32_t pitch_log2 = __builtin_ffs(vb->pitch / 3) - 1;
			builder_emit_vpslld(bld, offset, offset, pitch_log2);
		} else {
			int pitch = builder_get_reg_with_uniform(bld, vb->pitch);
			builder_emit_vpmulld(bld, offset, pitch, index);
		}

		if (vb->pitch > 0 && ve->offset > 0) {
			element_offset = builder_get_reg_with_uniform(bld, ve->offset);
			builder_emit_vpaddd(bld, offset, offset, element_offset);
		}

		int dst[4];
		emit_load_format_simd8(bld, ve->format, vb->data, offset, dst);

		for (uint32_t c = 0; c < 4; c++) {
			int src;
			switch (ve->cc[c]) {
			case VFCOMP_NOSTORE:
				continue;
			case VFCOMP_STORE_SRC:
				src = dst[c];
				break;
			case VFCOMP_STORE_0:
				src = builder_get_reg_with_uniform(bld, 0);
				break;
			case VFCOMP_STORE_1_FP:
				src = builder_get_reg_with_uniform(bld, float_to_u32(1.0f));
				break;
			case VFCOMP_STORE_1_INT:
				src = builder_get_reg_with_uniform(bld, 1);
				break;
			case VFCOMP_STORE_PID:
				ksim_unreachable("VFCOMP_STORE_PID");
				break;
			}
			store_v8(bld, offsetof(struct vf_buffer, data[i * 4 + c]), src);
		}

		builder_trace(bld, trace_file);
		builder_release_regs(bld);
	}

	if (gt.vf.iid_enable || gt.vf.vid_enable) {
		ksim_trace(TRACE_EU, "[ff vertex fetch: system generated values]\n");
		if (gt.vf.iid_enable) {
			int iid = load_uniform(bld, offsetof(struct vf_buffer, iid));
			uint32_t reg = gt.vf.iid_element * 4 + gt.vf.iid_component;
			store_v8(bld, offsetof(struct vf_buffer, data[reg]), iid);
		}
		if (gt.vf.vid_enable) {
			int vid = load_v8(bld, offsetof(struct vf_buffer, vid));
			uint32_t reg = gt.vf.vid_element * 4 + gt.vf.vid_component;
			store_v8(bld, offsetof(struct vf_buffer, data[reg]), vid);
		}

		builder_trace(bld, trace_file);
		builder_release_regs(bld);
	}

	if (trace_mask & TRACE_VF) {
		builder_emit_call(bld, dump_fetched_vue);
		builder_invalidate_all(bld);
	}

}

static uint32_t
builder_emit_load_constants(struct builder *bld, struct vf_buffer *buffer,
			    struct curbe *c, uint32_t start)
{
	uint32_t grf = start;
	struct reg *regs;
	uint64_t base, range;
	uint32_t bc = 0;

	ksim_trace(TRACE_EU, "[ff load constants]\n");
	for (uint32_t b = 0; b < 4; b++) {
		if (b == 0 && gt.curbe_dynamic_state_base)
			base = gt.dynamic_state_base_address;
		else
			base = 0;

		if (c->buffer[b].length > 0) {
			regs = map_gtt_offset(c->buffer[b].address + base, &range);
			ksim_assert(c->buffer[b].length * sizeof(regs[0]) <= range);
		}

		for (uint32_t i = 0; i < c->buffer[b].length; i++) {
			buffer->constants[bc] = regs[i];
			int reg = load_v8(bld, offsetof(struct vf_buffer, constants[bc++]));
			store_v8(bld, offsetof(struct vf_buffer, t.grf[grf++]), reg);
		}
	}

	ksim_assert(bc < ARRAY_LENGTH(buffer->constants));

	builder_trace(bld, trace_file);
	builder_release_regs(bld);

	return grf;
}

static void
builder_emit_load_vue(struct builder *bld, uint32_t grf)
{
	uint32_t src = offsetof(struct vf_buffer, data[gt.vs.vue_read_offset * 2 * 4]);
	uint32_t dst = offsetof(struct vf_buffer, t.grf[grf]);

	ksim_trace(TRACE_EU, "[ff copy vue]\n");
	for (uint32_t i = 0; i < gt.vs.vue_read_length * 2 * 4; i++) {
		int reg = load_v8(bld, src + i * 32);
		store_v8(bld, dst + i * 32, reg);
	}

	builder_trace(bld, trace_file);
	builder_release_regs(bld);
}

static void
compile_vs(struct vf_buffer *buffer)
{
	struct builder bld;

	ksim_trace(TRACE_EU | TRACE_AVX, "jit vs\n");

	builder_init(&bld,
		     gt.vs.binding_table_address,
		     gt.vs.sampler_state_address);

	uint32_t grf;
	if (gt.vs.enable) {
		grf = builder_emit_load_constants(&bld, buffer,
						  &gt.vs.curbe,
						  gt.vs.urb_start_grf);
	}

	/* Always need to fetch, even if we don't have a VS. */
	builder_emit_vertex_fetch(&bld);

	if (gt.vs.enable) {
		builder_emit_load_vue(&bld, grf);
		builder_emit_shader(&bld, gt.vs.ksp);
	}

	if (!gt.clip.perspective_divide_disable) {
		/* vrcpps doesn't have sufficient precision for
		 * perspective divide. We can use vdivps (latency
		 * 21/throughput 13) or do a Newton-Raphson step on
		 * vrcpps.  This turns into vrcpps, vfnmadd213ps and
		 * vmulps, with latencies 7, 5 and 5, which is
		 * slightly better.
		 */

		ksim_trace(TRACE_EU, "[ff perspective divide]\n");

		int w = load_v8(&bld, offsetof(struct vf_buffer, w));
		int inv_w = builder_get_reg(&bld);
		builder_emit_vrcpps(&bld, inv_w, w);

		/* NR step: inv_w = inv_w0 * (2 - w * inv_w0) */
		int two = builder_get_reg_with_uniform(&bld,
						       float_to_u32(2.0f));

		builder_emit_vfnmadd132ps(&bld, w, inv_w, two);
		builder_emit_vmulps(&bld, inv_w, inv_w, w);

		const int x = load_v8(&bld, offsetof(struct vf_buffer, x));
		builder_emit_vmulps(&bld, x, x, inv_w);
		store_v8(&bld, offsetof(struct vf_buffer, x), x);

		const int y = load_v8(&bld, offsetof(struct vf_buffer, y));
		builder_emit_vmulps(&bld, y, y, inv_w);
		store_v8(&bld, offsetof(struct vf_buffer, y), y);

		const int z = load_v8(&bld, offsetof(struct vf_buffer, z));
		builder_emit_vmulps(&bld, z, z, inv_w);
		store_v8(&bld, offsetof(struct vf_buffer, z), z);

		store_v8(&bld, offsetof(struct vf_buffer, w), inv_w);

		builder_trace(&bld, trace_file);
		builder_release_regs(&bld);
	}

	if (gt.clip.guardband_clip_test_enable ||
	    gt.clip.viewport_clip_test_enable) {
		ksim_trace(TRACE_EU, "[ff clip tests]\n");

		int x0 = load_uniform(&bld, offsetof(struct vf_buffer, clip.x0));
		int x1 = load_uniform(&bld, offsetof(struct vf_buffer, clip.x1));
		int y0 = load_uniform(&bld, offsetof(struct vf_buffer, clip.y0));
		int y1 = load_uniform(&bld, offsetof(struct vf_buffer, clip.y1));
		int x = load_v8(&bld, offsetof(struct vf_buffer, x));
		int y = load_v8(&bld, offsetof(struct vf_buffer, y));

		builder_emit_vcmpps(&bld, _CMP_LT_OS, x0, x0, x);
		builder_emit_vcmpps(&bld, _CMP_GT_OS, x1, x1, x);
		builder_emit_vcmpps(&bld, _CMP_LT_OS, y0, y0, y);
		builder_emit_vcmpps(&bld, _CMP_GT_OS, y1, y1, y);

		builder_emit_vpor(&bld, x0, x0, x1);
		builder_emit_vpor(&bld, y0, y0, y1);
		builder_emit_vpor(&bld, x0, x0, y0);

		store_v8(&bld, offsetof(struct vf_buffer, clip_flags), x0);

		builder_trace(&bld, trace_file);
		builder_release_regs(&bld);
	}

	if (gt.sf.viewport_transform_enable) {
		ksim_trace(TRACE_EU, "[ff viewport transform]\n");

		int m00 = load_uniform(&bld, offsetof(struct vf_buffer, vp.m00));
		int m11 = load_uniform(&bld, offsetof(struct vf_buffer, vp.m11));
		int m22 = load_uniform(&bld, offsetof(struct vf_buffer, vp.m22));
		int m30 = load_uniform(&bld, offsetof(struct vf_buffer, vp.m30));
		int m31 = load_uniform(&bld, offsetof(struct vf_buffer, vp.m31));
		int m32 = load_uniform(&bld, offsetof(struct vf_buffer, vp.m32));

		int x = load_v8(&bld, offsetof(struct vf_buffer, x));
		int y = load_v8(&bld, offsetof(struct vf_buffer, y));
		int z = load_v8(&bld, offsetof(struct vf_buffer, z));

		builder_emit_vfmadd132ps(&bld, x, m00, m30);
		builder_emit_vfmadd132ps(&bld, y, m11, m31);
		builder_emit_vfmadd132ps(&bld, z, m22, m32);

		store_v8(&bld, offsetof(struct vf_buffer, x), x);
		store_v8(&bld, offsetof(struct vf_buffer, y), y);
		store_v8(&bld, offsetof(struct vf_buffer, z), z);

		builder_trace(&bld, trace_file);
		builder_release_regs(&bld);
	}

	builder_emit_ret(&bld);
	builder_trace(&bld, trace_file);
	builder_release_regs(&bld);

	gt.vs.avx_shader = builder_finish(&bld);
}

static void
init_vf_buffer(struct vf_buffer *buffer)
{
	uint64_t range;
	buffer->index_buffer = map_gtt_offset(gt.vf.ib.address, &range);

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

	init_vf_buffer(&buffer);

	compile_vs(&buffer);

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
