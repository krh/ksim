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

uint32_t
emit_load_constants(struct kir_program *prog, struct curbe *c, uint32_t start)
{
	uint32_t grf = start;
	uint32_t bc = 0;

	kir_program_comment(prog, "load constants");
	for (uint32_t b = 0; b < 4; b++) {
		for (uint32_t i = 0; i < c->buffer[b].length; i++) {
			kir_program_load_v8(prog, offsetof(struct thread, constants[bc++]));
			kir_program_store_v8(prog, offsetof(struct thread, grf[grf++]), prog->dst);
		}
	}

	return grf;
}

void
load_constants(struct thread *t, struct curbe *c)
{
	struct reg *regs;
	uint64_t base, range;
	uint32_t bc = 0;

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
			t->constants[bc++] = regs[i].ireg;
	}

	ksim_assert(bc < ARRAY_LENGTH(t->constants));
}

shader_t
compile_shader(uint64_t kernel_offset,
	       uint64_t surfaces, uint64_t samplers)
{
	struct kir_program prog;

	kir_program_init(&prog, surfaces, samplers);

	kir_program_emit_shader(&prog, kernel_offset);

	kir_program_add_insn(&prog, kir_eot);

	return kir_program_finish(&prog);
}
