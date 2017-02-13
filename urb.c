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
#include "kir.h"

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

static void
builder_emit_sfid_urb_simd8_write(struct kir_program *prog, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	uint32_t src = (unpack_inst_2src_src0(inst).num + 1) * 32;
	uint32_t vue_offset = field(send.function_control, 4, 14);
	uint32_t dst = prog->urb_offset + vue_offset * 4 * 32;

	/* We should only get here if there's a urb offset set. */
	ksim_assert(prog->urb_offset != 0);

	kir_program_comment(prog, "urb write: length %d, offset %d",
			    send.mlen - 1, vue_offset);

	for (uint32_t i = 0; i < send.mlen - 1; i++) {
		kir_program_load_v8(prog, src + i * 32);
		kir_program_store_v8(prog, dst + i  * 32, prog->dst);
	}
}

enum urb_opcode {
	URB_WRITE_HWORD = 0,
	URB_WRITE_OWORD = 1,
	URB_READ_HWORD = 2,
	URB_READ_OWORD = 3,
	URB_ATOMIC_MOV = 4,
	URB_ATOMIC_INC = 5,
	URB_ATOMIC_ADD = 6,
	URB_SIMD8_WRITE = 7,
	URB_SIMD8_READ = 8,
};

enum urb_swizzle_control {
	URB_NOSWIZZLE = 0,
	URB_INTERLEAVED = 1,
};

struct urb_message_descriptor {
	enum urb_opcode			opcode;
	uint32_t			global_offset;
	enum urb_swizzle_control	swizzle;
	bool				per_slot_offset;
	bool				header_present;
	uint32_t			response_length;
	uint32_t			message_length;
	bool				eot;
};

static inline struct urb_message_descriptor
unpack_urb_message_descriptor(uint32_t function_control)
{
	return (struct urb_message_descriptor) {
		.opcode			= field(function_control,   0,    3),
		.global_offset		= field(function_control,   4,   14),
		.swizzle		= field(function_control,  15,   15),
		.per_slot_offset	= field(function_control,  17,   17),
		.header_present		= field(function_control,  19,   19),
		.response_length	= field(function_control,  20,   24),
		.message_length		= field(function_control,  25,   28),
		.eot			= field(function_control,  31,   31),
	};
}

void
builder_emit_sfid_urb(struct kir_program *prog, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	struct urb_message_descriptor md =
		unpack_urb_message_descriptor(send.function_control);

	switch (md.opcode) {
	case URB_WRITE_HWORD:
	case URB_WRITE_OWORD:
	case URB_READ_HWORD:
	case URB_READ_OWORD:
	case URB_ATOMIC_MOV:
	case URB_ATOMIC_INC:
	case URB_ATOMIC_ADD:
	case URB_SIMD8_READ:
		stub("sfid urb opcode %d", md.opcode);
		return;
	case URB_SIMD8_WRITE:
		ksim_assert(send.header_present);
		ksim_assert(send.rlen == 0);
		ksim_assert(!md.per_slot_offset);
		builder_emit_sfid_urb_simd8_write(prog, inst);
		break;
	default:
		ksim_unreachable("out of range urb opcode: %d", md.opcode);
		break;
	}

}
