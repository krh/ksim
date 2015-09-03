/*
 * Copyright © 2014 Intel Corporation
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

#ifndef GEN_DISASM_H
#define GEN_DISASM_H

struct gen_disasm;

struct gen_disasm *gen_disasm_create(int gen);
void gen_disasm_disassemble(struct gen_disasm *disasm,
			    void *assembly, int start, int end, FILE *out);

void gen_disasm_destroy(struct gen_disasm *disasm);

struct reg {
	union {
		float f[8];
		uint32_t ud[8];
		int32_t d[8];
		uint16_t uw[16];
		int16_t w[16];
		uint8_t ub[16];
		int8_t b[16];
		uint64_t uq[4];
		int64_t q[4];
	};
};

struct thread {
	struct reg grf[128];
};

void execute_thread(struct gen_disasm *disasm, struct thread *t, void *insns, FILE *out);

#endif /* GEN_DISASM_H */
