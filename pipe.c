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

static inline int32_t
fp_as_int32(float f)
{
	return (union { float f; int32_t i; }) { .f = f }.i;
}

static int32_t
store_component(uint32_t cc, int32_t src)
{
	switch (cc) {
	case VFCOMP_NOSTORE:
		return 77; /* shouldn't matter */
	case VFCOMP_STORE_SRC:
		return src;
	case VFCOMP_STORE_0:
		return 0;
	case VFCOMP_STORE_1_FP:
		return fp_as_int32(1.0f);
	case VFCOMP_STORE_1_INT:
		return 1;
	case VFCOMP_STORE_PID:
		return 0; /* what's pid again? */
	default:
		ksim_warn("illegal component control: %d\n", cc);
		return 0;
	}
}

static struct value *
fetch_vertex(uint32_t instance_id, uint32_t vertex_id)
{
	struct value *vue;
	struct value v;

	vue = alloc_urb_entry(&gt.vs.urb);
	for (uint32_t i = 0; i < gt.vf.ve_count; i++) {
		struct ve *ve = &gt.vf.ve[i];
		ksim_assert((1 << ve->vb) & gt.vf.vb_valid);
		struct vb *vb = &gt.vf.vb[ve->vb];

		if (!gt.vf.ve[i].valid)
			continue;

		uint32_t index;
		if (gt.vf.ve[i].instancing) {
			index = gt.prim.start_instance + instance_id / gt.vf.ve[i].step_rate;
		} else if (gt.prim.access_type == RANDOM) {
			uint64_t range;
			void *ib = map_gtt_offset(gt.vf.ib.address, &range);

			index = gt.prim.start_vertex + vertex_id;

			switch (gt.vf.ib.format) {
			case INDEX_BYTE:
				index = ((uint8_t *) ib)[index] + gt.prim.base_vertex;
				break;
			case INDEX_WORD:
				index = ((uint16_t *) ib)[index] + gt.prim.base_vertex;
				break;
			case INDEX_DWORD:
				index = ((uint32_t *) ib)[index] + gt.prim.base_vertex;
				break;
			}
		} else {
			index = gt.prim.start_vertex + vertex_id;
		}

		uint32_t offset = index * vb->pitch + ve->offset;
		if (offset + format_size(ve->format) > vb->size) {
			ksim_trace(TRACE_WARN, "vertex element %d overflows vertex buffer %d\n",
				   i, ve->vb);
			v = vec4(0, 0, 0, 0);
		} else {
			v = fetch_format(vb->data + offset, ve->format);
		}

		for (uint32_t c = 0; c < 4; c++)
			vue[i].v[c] = store_component(ve->cc[c], v.v[c]);

		/* edgeflag */
	}

	/* 3DSTATE_VF_SGVS */
	if (gt.vf.iid_enable && gt.vf.vid_enable)
		ksim_assert(gt.vf.iid_element != gt.vf.vid_element ||
			    gt.vf.iid_component != gt.vf.vid_component);

	if (gt.vf.iid_enable)
		vue[gt.vf.iid_element].v[gt.vf.iid_component] = instance_id;
	if (gt.vf.vid_enable)
		vue[gt.vf.vid_element].v[gt.vf.vid_component] = vertex_id;

	if (trace_mask & TRACE_VF) {
		ksim_trace(TRACE_VF, "Loaded vue for vid=%d, iid=%d:\n",
			   vertex_id, instance_id);
		uint32_t count = gt.vf.ve_count;
		if (gt.vf.iid_element + 1 > count)
			count = gt.vf.iid_element + 1;
		if (gt.vf.vid_element + 1 > count)
			count = gt.vf.vid_element + 1;
		for (uint32_t i = 0; i < count; i++)
			ksim_trace(TRACE_VF, "    %8.2f  %8.2f  %8.2f  %8.2f\n",
				   vue[i].f[0], vue[i].f[1], vue[i].f[2], vue[i].f[3]);
	}

	return vue;
}

static void
dispatch_vs(struct value **vue, uint32_t mask)
{
	struct thread t;
	uint32_t g, c;

	if (!gt.vs.enable)
		return;

	assert(gt.vs.simd8);

	/* Not sure what we should make this. */
	uint32_t fftid = 0;

	t.mask = mask;
	/* Fixed function header */
	t.grf[0] = (struct reg) {
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

	for_each_bit(c, mask)
		t.grf[1].ud[c] = urb_entry_to_handle(vue[c]);

	g = load_constants(&t, &gt.vs.curbe, gt.vs.urb_start_grf);

	/* SIMD8 VS payload */
	for (uint32_t i = 0; i < gt.vs.vue_read_length * 2; i++) {
		for_each_bit(c, mask) {
			for (uint32_t j = 0; j < 4; j++)
				t.grf[g + j].ud[c] = vue[c][gt.vs.vue_read_offset * 2 + i].v[j];
		}
		g += 4;
	}

	if (gt.vs.statistics)
		gt.vs_invocation_count++;

	dispatch_shader(gt.vs.avx_shader, &t);
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

enum vue_flag {
	VUE_FLAG_CLIP = 1
};

static void
setup_prim(struct value **vue_in, uint32_t parity)
{
	struct value *vue[3];
	uint32_t provoking;

	for (int i = 0; i < 3; i++) {
		if (vue_in[i][0].u[0] & VUE_FLAG_CLIP)
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
transform_and_queue_vues(struct value **vue, int count)
{
	const float *vp = gt.sf.viewport;
	float m00, m11, m22, m30, m31, m32;
	struct rectanglef clip;

	if (gt.sf.viewport_transform_enable) {
		m00 = vp[0];
		m11 = vp[1];
		m22 = vp[2];
		m30 = vp[3];
		m31 = vp[4];
		m32 = vp[5];
	}

	if (gt.clip.guardband_clip_test_enable) {
		clip = gt.sf.guardband;
	} else {
		clip.x0 = -1;
		clip.y0 = -1;
		clip.x1 = 1;
		clip.y1 = 1;
	}

	for (int i = 0; i < count; i++) {
		struct vec4 *pos = &vue[i][1].vec4;
		if (!gt.clip.perspective_divide_disable) {
			float inv_w = 1.0f / pos->w;
			pos->x *= inv_w;
			pos->y *= inv_w;
			pos->z *= inv_w;
			pos->w = inv_w;
		}

		if (gt.clip.guardband_clip_test_enable ||
		    gt.clip.viewport_clip_test_enable) {
			const struct vec4 v = vue[i][1].vec4;
			if (v.x < clip.x0 || clip.x1 < v.x ||
			    v.y < clip.y0 || clip.y1 < v.y || v.z > 1.0f) {
				vue[i][0].u[0] = VUE_FLAG_CLIP;
			} else {
				vue[i][0].u[0] = 0;
			}
		}

		if (gt.sf.viewport_transform_enable) {
			pos->x = m00 * pos->x + m30;
			pos->y = m11 * pos->y + m31;
			pos->z = m22 * pos->z + m32;
		}

		gt.ia.queue.vue[gt.ia.queue.head++ & 15] = vue[i];
	}

	ksim_assert(gt.ia.queue.head - gt.ia.queue.tail < 16);
}

static void
assemble_primitives()
{
	struct value *vue[3];
	uint32_t tail = gt.ia.queue.tail;

	switch (gt.ia.topology) {
	case _3DPRIM_TRILIST:
		while (gt.ia.queue.head - tail >= 3) {
			vue[0] = gt.ia.queue.vue[(tail + 0) & 15];
			vue[1] = gt.ia.queue.vue[(tail + 1) & 15];
			vue[2] = gt.ia.queue.vue[(tail + 2) & 15];
			setup_prim(vue, 0);
			tail += 3;
			gt.ia_primitives_count++;
		}
		break;

	case _3DPRIM_TRISTRIP:
		while (gt.ia.queue.head - tail >= 3) {
			vue[0] = gt.ia.queue.vue[(tail + 0) & 15];
			vue[1] = gt.ia.queue.vue[(tail + 1) & 15];
			vue[2] = gt.ia.queue.vue[(tail + 2) & 15];
			setup_prim(vue, gt.ia.tristrip_parity);
			tail += 1;
			gt.ia.tristrip_parity = 1 - gt.ia.tristrip_parity;
			gt.ia_primitives_count++;
		}
		break;

	case _3DPRIM_POLYGON:
	case _3DPRIM_TRIFAN:
		if (gt.ia.trifan_first_vertex == NULL) {
			/* We always have at least one vertex
			 * when we get, so this is safe. */
			assert(gt.ia.queue.head - tail >= 1);
			gt.ia.trifan_first_vertex = gt.ia.queue.vue[tail & 15];
			/* Bump the queue tail now so we don't free
			 * the vue below */
			gt.ia.queue.tail++;
			tail++;
			gt.ia_primitives_count++;
		}

		while (gt.ia.queue.head - tail >= 2) {
			vue[0] = gt.ia.trifan_first_vertex;
			vue[1] = gt.ia.queue.vue[(tail + 0) & 15];
			vue[2] = gt.ia.queue.vue[(tail + 1) & 15];
			setup_prim(vue, gt.ia.tristrip_parity);
			tail += 1;
			gt.ia_primitives_count++;
		}
		break;
	case _3DPRIM_QUADLIST:
		while (gt.ia.queue.head - tail >= 4) {
			vue[0] = gt.ia.queue.vue[(tail + 3) & 15];
			vue[1] = gt.ia.queue.vue[(tail + 0) & 15];
			vue[2] = gt.ia.queue.vue[(tail + 1) & 15];
			setup_prim(vue, 0);
			vue[0] = gt.ia.queue.vue[(tail + 3) & 15];
			vue[1] = gt.ia.queue.vue[(tail + 1) & 15];
			vue[2] = gt.ia.queue.vue[(tail + 2) & 15];
			setup_prim(vue, 0);
			tail += 4;
			gt.ia_primitives_count++;
		}
		break;
	case _3DPRIM_QUADSTRIP:
		while (gt.ia.queue.head - tail >= 4) {
			vue[0] = gt.ia.queue.vue[(tail + 3) & 15];
			vue[1] = gt.ia.queue.vue[(tail + 0) & 15];
			vue[2] = gt.ia.queue.vue[(tail + 1) & 15];
			setup_prim(vue, 0);
			vue[0] = gt.ia.queue.vue[(tail + 3) & 15];
			vue[1] = gt.ia.queue.vue[(tail + 2) & 15];
			vue[2] = gt.ia.queue.vue[(tail + 0) & 15];
			setup_prim(vue, 0);
			tail += 2;
			gt.ia_primitives_count++;
		}
		break;

	case _3DPRIM_RECTLIST:
		while (gt.ia.queue.head - tail >= 3) {
			vue[0] = gt.ia.queue.vue[(tail + 0) & 15];
			vue[1] = gt.ia.queue.vue[(tail + 1) & 15];
			vue[2] = gt.ia.queue.vue[(tail + 2) & 15];
			setup_prim(vue, 0);
			tail += 3;
		}
		break;

	default:
		stub("topology %d", gt.ia.topology);
		tail = gt.ia.queue.head;
		break;
	}

	while (tail - gt.ia.queue.tail > 0) {
		struct value *vue =
			gt.ia.queue.vue[gt.ia.queue.tail++ & 15];
		free_urb_entry(&gt.vs.urb, vue);
	}
}

void
reset_ia_state(void)
{
	if (gt.ia.trifan_first_vertex) {
		free_urb_entry(&gt.vs.urb, gt.ia.trifan_first_vertex);
		gt.ia.trifan_first_vertex = NULL;
	}

	while (gt.ia.queue.head - gt.ia.queue.tail > 0) {
		struct value *vue =
			gt.ia.queue.vue[gt.ia.queue.tail++ & 15];
		free_urb_entry(&gt.vs.urb, vue);
	}

	gt.ia.queue.head = 0;
	gt.ia.queue.tail = 0;
	gt.ia.tristrip_parity = 0;
	gt.ia.trifan_first_vertex = NULL;
}

void
dispatch_primitive(void)
{
	uint32_t i = 0;
	struct value *vue[8];
	uint32_t iid, vid;

	validate_vf_state();

	validate_urb_state();

	if (gt.sf.viewport_transform_enable)
		dump_sf_clip_viewport();

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

	for (iid = 0; iid < gt.prim.instance_count; iid++) {
		for (vid = 0; vid < gt.prim.vertex_count; vid++) {
			vue[i++] = fetch_vertex(iid, vid);
			if (gt.vf.statistics)
				gt.ia_vertices_count++;
			if (i == 8) {
				dispatch_vs(vue, 255);
				transform_and_queue_vues(vue, i);
				assemble_primitives(vue, i);
				i = 0;
			}
		}
		if (i > 0) {
			dispatch_vs(vue, (1 << i) - 1);
			transform_and_queue_vues(vue, i);
			assemble_primitives(vue, i);
			i = 0;
		}

		reset_ia_state();
	}

	wm_flush();
}
