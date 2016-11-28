/*
 * Copyright © 2016 Kristian H. Kristensen
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

#include "eu.h"
#include "avx-builder.h"

struct free_urb {
	uint32_t next;
};

void
set_urb_allocation(struct urb *urb, uint32_t address, uint32_t size, uint32_t total)
{
	const uint32_t chunk_size_bytes = 8192;

	urb->data = gt.urb + address * chunk_size_bytes;
	urb->size = (size + 1) * 64;
	urb->total = total;

	urb->free_list = URB_EMPTY;
	urb->count = 0;
}

void *
alloc_urb_entry(struct urb *urb)
{
	struct free_urb *f;
	void *p;

	if (urb->free_list != URB_EMPTY) {
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

void
free_urb_entry(struct urb* urb, void *entry)
{
	struct free_urb *f = entry;

	f->next = urb->free_list;
	urb->free_list = entry - urb->data;
}

void
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

	/* If we're doing SIMD8 vs dispatch, we need at least 8 VUEs,
	 * but the BDW hw limit is even higher: 64. */
	ksim_assert(64 <= gt.vs.urb.total && gt.vs.urb.total <= 2560);

}

struct sfid_urb_args {
	int src;
	int offset;
	int len;
};

static void
sfid_urb_simd8_write(struct thread *t, struct sfid_urb_args *args)
{
	if (trace_mask & TRACE_URB) {
		ksim_trace(TRACE_URB,
			   "urb simd8 write, src g%d, global offset %d, mlen %lu, mask %02x\n",
			   args->src, args->offset, args->len, t->mask);

		ksim_trace(TRACE_URB, "  grf%d:", args->src);
		for (int c = 0; c < 8; c++)
			ksim_trace(TRACE_URB, "  %6d", t->grf[args->src].d[c]);
		ksim_trace(TRACE_URB, "\n");

		for (int i = 1; i < args->len; i++) {
			ksim_trace(TRACE_URB, "  grf%d:", args->src + i);
			for (int c = 0; c < 8; c++)
				ksim_trace(TRACE_URB,
					   "  %6.1f", t->grf[args->src + i].f[c]);
			ksim_trace(TRACE_URB, "\n");
		}
	}

	static const struct reg offsets = { .d = {  0, 32, 64, 96, 128, 160, 192, 224 } };
	uint32_t *handles = t->grf[args->src].ud;
	uint32_t *values = t->grf[args->src + 1].ud;
	uint32_t c;

	if (args->len == 9) {
		for_each_bit (c, t->mask) {
			struct value *vue = urb_handle_to_entry(handles[c]);
			__m256i e = _mm256_i32gather_epi32((void *) &values[c],
							   offsets.ireg, 1);
			_mm256_storeu_si256((void *) &vue[args->offset], e);
		}
	} else if (args->len == 5) {
		for_each_bit (c, t->mask) {
			struct value *vue = urb_handle_to_entry(handles[c]);
			__m128i e = _mm_i32gather_epi32((void *) &values[c],
							offsets.ihreg, 1);
			_mm_storeu_si128((void *) &vue[args->offset], e);
		}
	} else {
		for_each_bit (c, t->mask) {
			struct value *vue = urb_handle_to_entry(handles[c]);
			for (int i = 0; i < args->len - 1; i++)
				vue[args->offset + i / 4].v[i % 4] =
					t->grf[args->src + 1 + i].ud[c];
		}
	}
}

void
builder_emit_sfid_urb(struct builder *bld, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	struct sfid_urb_args *args;
	void *func = NULL;

	args = builder_get_const_data(bld, sizeof *args, 8);
	args->src = unpack_inst_2src_src0(inst).num;
	args->len = send.mlen;
	args->offset = field(send.function_control, 4, 14);

	uint32_t opcode = field(send.function_control, 0, 3);
	bool per_slot_offset = field(send.function_control, 17, 17);

	builder_emit_load_rsi_rip_relative(bld, builder_offset(bld, args));

	switch (opcode) {
	case 0: /* write HWord */
	case 1: /* write OWord */
	case 2: /* read HWord */
	case 3: /* read OWord */
	case 4: /* atomic mov */
	case 5: /* atomic inc */
	case 6: /* atomic add */
		stub("sfid urb opcode %d", opcode);
		return;
	case 7: /* SIMD8 write */
		ksim_assert(send.header_present);
		ksim_assert(send.rlen == 0);
		ksim_assert(!per_slot_offset);
		func = sfid_urb_simd8_write;
		break;
	default:
		ksim_unreachable("out of range urb opcode: %d", opcode);
		break;
	}

	if (send.eot) {
		builder_emit_jmp_relative(bld, (uint8_t *) func - bld->p);
	} else {
		builder_emit_call(bld, func);
	}
}
