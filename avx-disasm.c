/*
 * Copyright © 2016 Intel Corporation
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

#include <bfd.h>
#include <dis-asm.h>

#include "ksim.h"

void
print_avx(struct shader *shader, int start, int end)
{
	struct disassemble_info info;
	int pc, count;

	init_disassemble_info(&info, trace_file,
			      (fprintf_ftype)fprintf);
	/* info.print_address_func = override_print_address; */
	info.arch = bfd_arch_i386;
	info.mach = bfd_mach_x86_64;
	info.buffer_vma = 0;
	info.buffer_length = 64 * 4096;
	info.section = NULL;
	info.buffer = shader->code;
	disassemble_init_for_target(&info);

	for (pc = start; pc < end; pc += count) {
		fprintf(trace_file, "      ");
		count = print_insn_i386(pc, &info);
		fprintf(trace_file, "\n");
	}
}
