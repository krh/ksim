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

uint32_t
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

void
run_thread(struct thread *t, uint64_t ksp, uint32_t trace_flag)
{
	static struct gen_disasm *disasm;
	const int gen = 8;
	uint64_t range;
	void *kernel;;
	FILE *out = NULL;

	if (disasm == NULL)
		disasm = gen_disasm_create(gen);

	kernel = map_gtt_offset(ksp + gt.instruction_base_address, &range);

	if (trace_mask & trace_flag)
		out = trace_file;

	execute_thread(disasm, t, kernel, out);
}

void
sfid_urb_simd8_write(struct thread *t, int reg, int offset, int mlen)
{
	spam("urb simd8 write, src g%d, global offset %d, mlen %lu, mask %02x\n",
	     reg, offset, mlen, t->mask);

	spam("  grf%d:", reg);
	for (int c = 0; c < 8; c++)
		spam("  %6d", t->grf[reg].d[c]);
	spam("\n");

	for (int i = 1; i < mlen; i++) {
		spam("  grf%d:", reg + i);
		for (int c = 0; c < 8; c++)
			spam("  %6.1f", t->grf[reg + i].f[c]);
		spam("\n");
	}

	uint32_t c;
	for_each_bit (c, t->mask) {
		struct value *vue = urb_handle_to_entry(t->grf[reg].ud[c]);
		for (int i = 0; i < mlen - 1; i++)
			vue[offset + i / 4].v[i % 4] = t->grf[reg + 1 + i].ud[c];
	}
}

void
sfid_urb(struct thread *t,
	 uint32_t dst, uint32_t src,
	 uint32_t function_control,
	 bool header_present, int mlen, int rlen)
{
	uint32_t opcode = field(function_control, 0, 3);
	uint32_t global_offset = field(function_control, 4, 14);
	bool per_slot_offset = field(function_control, 17, 17);

	switch (opcode) {
	case 0: /* write HWord */
	case 1: /* write OWord */
	case 2: /* read HWord */
	case 3: /* read OWord */
	case 4: /* atomic mov */
	case 5: /* atomic inc */
	case 6: /* atomic add */
		break;
	case 7: /* SIMD8 write */
		sfid_urb_simd8_write(t, src, global_offset, mlen);
		break;
	}
}
