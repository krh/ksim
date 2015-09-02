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

struct free_urb {
	uint32_t next;
};

#define EMPTY 1

static void *
alloc_urb_entry(struct urb *urb)
{
	struct free_urb *f;
	void *p;

	if (urb->free_list != EMPTY) {
		f = p = urb->data + urb->free_list;
		urb->free_list = f->next;
	} else {
		ksim_assert(urb->count < urb->total);
		p = urb->data + urb->size * urb->count++;
	}

	ksim_assert(p >= urb->data && p < urb->data + urb->total * urb->size);
	ksim_assert(p >= (void *) gt.urb && p < (void *) gt.urb + sizeof(gt.urb));

	return p;
}

static void
free_urb_entry(struct urb* urb, void *entry)
{
	struct free_urb *f = entry;

	f->next = urb->free_list;
	urb->free_list = entry - urb->data;
}

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
		ksim_assert(valid_vertex_format(ve->format));
		if (offset + format_size(ve->format) > vb->size) {
			ksim_trace(TRACE_WARN, "vertex element overflow");
			v = vec4(0, 0, 0, 0);
		} else {
			v = fetch_format(vb->address + offset, ve->format);
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
		const uint32_t count = gt.vs.urb.size / 16;
		for (uint32_t i = 0; i < count; i++)
			ksim_trace(TRACE_VF, "    %8.2f  %8.2f  %8.2f  %8.2f\n",
				   vue[i].f[0], vue[i].f[1], vue[i].f[2], vue[i].f[3]);
	}

	return vue;
}

static uint32_t
load_constants(struct thread *t, struct curbe *c, uint32_t start)
{
	uint32_t grf = start;
	struct reg *regs;
	uint64_t base, range;

	for (uint32_t b = 0; b < 4; b++) {
		if (b == 0 && gt.curbe_dynamic_state_base)
			base = gt.dynamic_state_base_address;
		else
			base = 0;
			
		if (c->buffer[b].length > 0) {
			regs = map_gtt_offset(c->buffer[b].address + base, &range);
			ksim_assert(c->buffer[b].length * sizeof(regs[0]) <= range);
		}

		for (uint32_t i = 0; i < c->buffer[b].length; i++)
			t->grf[grf++] = regs[i];
	}

	return grf;
}

static void
dispatch_vs(struct value **vue, uint32_t mask)
{
	struct thread t;
	uint32_t g = 0, c;

	if (!gt.vs.enable)
		return;

	assert(gt.vs.simd8);
	
	/* Fixed function header */
	t.grf[g++] = (struct reg) {
		.u = {
			0, /* MBZ */
			0, /* MBZ */
			0, /* MBZ */
			0, /* per-thread scratch space, sampler ptr */
			0, /* binding table pointer */
			0, /* fftid, scratch offset */
			0, /* thread id */
			0, /* snapshot flag */
		}
	};

	/* VUE handles. FIXME: VUE handles are supposed to be 16 bits. */
	for_each_bit(c, mask)
		t.grf[g++].f[c] = (char *) vue[c] - gt.urb;

	g = load_constants(&t, &gt.vs.curbe, gt.vs.urb_start_grf);

	/* SIMD8 VS payload */
	for (uint32_t i = 0; i < gt.vs.vue_read_length; i++) {
		for_each_bit(c, mask) {
			for (uint32_t j = 0; j < 4; j++)
				t.grf[g++].f[c] = vue[c][gt.vs.vue_read_offset + i].v[j];
		}
	}

	if (gt.vs.statistics)
		gt.vs_invocation_count++;

	run_thread(&t, gt.vs.ksp);

	for_each_bit(c, mask)
		free_urb_entry(&gt.vs.urb, vue[c]);
}

static void
validate_vf_state(void)
{
	uint32_t vb, vb_used;
	
	/* Make sure vue is big enough to hold all vertex elements */
	ksim_assert(gt.vf.ve_count * 16 < gt.vs.urb.size);

	vb_used = 0;
	for (uint32_t i = 0; i < gt.vf.ve_count; i++) {
		if (gt.vf.ve[i].valid)
			vb_used |= 1 << gt.vf.ve[i].vb;
	}

	/* Check all VEs reference valid VBs. */
	ksim_assert(vb_used & gt.vf.vb_valid == vb_used);
}

static void
validate_urb_state(void)
{
	struct urb *all_urbs[] = {
		&gt.vs.urb,
		&gt.hs.urb,
		&gt.ds.urb,
		&gt.gs.urb,
	}, *u, *v;

	/* Validate that the URB allocations are properly sized and
	 * don't overlap
	 */

	for (uint32_t i = 0; i < ARRAY_LENGTH(all_urbs); i++) {
		u = all_urbs[i];
		char *ustart = u->data;
		char *uend = ustart + u->total * u->size;
		ksim_assert(gt.urb <= ustart && uend <= gt.urb + sizeof(gt.urb));

		for (uint32_t j = i + 1; j < ARRAY_LENGTH(all_urbs); j++) {
			v = all_urbs[j];
			char *vstart = v->data;
			char *vend = v->data + v->total * v->size;
			ksim_assert(vend <= ustart || uend <= vstart);
		}
	}
}

static void
dump_sf_clip_viewport(void)
{
	uint64_t range;
	float *p = map_gtt_offset(gt.sf.viewport_pointer, &range);

	spam("gt.sf.viewport_pointer: 0x%08x, (w/o dyn base: 0x%08x)\n",
	     gt.sf.viewport_pointer,
	     gt.sf.viewport_pointer - gt.dynamic_state_base_address);

	ksim_assert(range >= 14 * sizeof(*p));

	spam("sf_clip viewport: %08x\n", gt.sf.viewport_pointer);
	for (uint32_t i = 0; i < 14; i++)
		spam("  %20.4f\n", p[i]);
}

void
dispatch_primitive(void)
{
	struct value *v;
	uint32_t i = 0, mask = 0;
	struct value *vue[8];
	uint32_t iid, vid, vb;
	
	validate_vf_state();

	validate_urb_state();

	dump_sf_clip_viewport();

	for (iid = 0; iid < gt.prim.instance_count; iid++) {
		for (vid = 0; vid < gt.prim.vertex_count; vid++) {
			vue[i] = fetch_vertex(iid, vid);
			if (gt.vf.statistics)
				gt.ia_vertices_count++;
			mask |= 1 << i;
			i++;
			if (i == 8) {
				dispatch_vs(vue, mask);
				i = mask = 0;
			}

			/* assemble primitive, 
			if (gt.vf.statistics)
				gt.ia_primitives_count++;
			*/

		}
	}

	if (mask)
		dispatch_vs(vue, mask);
}
