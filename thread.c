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

#include "libdisasm/gen_disasm.h"

void
run_thread(struct thread *t, uint64_t ksp)
{
	static struct gen_disasm *disasm;
	const int gen = 8;
	uint64_t range;
	void *kernel;;

	if (disasm == NULL)
		disasm = gen_disasm_create(gen);

	kernel = map_gtt_offset(ksp + gt.instruction_base_address, &range);

	if (trace_mask & TRACE_KERNELS) {
		ksim_trace(TRACE_KERNELS, "disassembled kernel:\n");
		gen_disasm_disassemble(disasm, kernel, 0, range, trace_file);
		ksim_trace(TRACE_KERNELS, "\n");
	}
}

