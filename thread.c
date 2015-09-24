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


static struct gen_disasm *
get_disasm(void)
{
	const int gen = 8;
	static struct gen_disasm *disasm;

	if (disasm == NULL)
		disasm = gen_disasm_create(gen);

	return disasm;
}

void
prepare_shaders(void)
{
	int offset = 0;
	uint64_t ksp, range;
	static char cache[64 * 1024];
	void *kernel, *end;

	gt.vs.shader = cache + offset;
	ksp = gt.vs.ksp + gt.instruction_base_address;
	kernel = map_gtt_offset(ksp, &range);
	offset = gen_disasm_uncompact(get_disasm(), kernel,
				      gt.vs.shader, sizeof(cache) - offset);

	offset = align_u64(offset, 64);
	gt.ps.shader = cache + offset;
	ksp = gt.ps.ksp0 + gt.instruction_base_address;
	kernel = map_gtt_offset(ksp, &range);
	offset = gen_disasm_uncompact(get_disasm(), kernel,
				      gt.ps.shader, sizeof(cache) - offset);

	static void *pool;
	const size_t size = 64 * 1024;
	if (pool == NULL) {
		int fd = memfd_create("jit", MFD_CLOEXEC);
		ftruncate(fd, size);
		pool = mmap(NULL, size, PROT_WRITE | PROT_READ | PROT_EXEC, MAP_SHARED, fd, 0);
		close(fd);
	}

	offset = 0;
	gt.vs.avx_shader = pool + offset;
	end = compile_shader(gt.vs.shader, gt.vs.avx_shader);

	offset = end - (void *) gt.vs.avx_shader;
	gt.ps.avx_shader = pool + align_u64(offset, 64);
	end = compile_shader(gt.ps.shader, gt.ps.avx_shader);
	ksim_assert(end - pool < size);
}

void
sfid_urb_simd8_write(struct thread *t, int reg, int offset, int mlen)
{
	if (trace_mask & TRACE_URB) {
		ksim_trace(TRACE_URB,
			   "urb simd8 write, src g%d, global offset %d, mlen %lu, mask %02x\n",
			   reg, offset, mlen, t->mask);

		ksim_trace(TRACE_URB, "  grf%d:", reg);
		for (int c = 0; c < 8; c++)
			ksim_trace(TRACE_URB, "  %6d", t->grf[reg].d[c]);
		ksim_trace(TRACE_URB, "\n");

		for (int i = 1; i < mlen; i++) {
			ksim_trace(TRACE_URB, "  grf%d:", reg + i);
			for (int c = 0; c < 8; c++)
				ksim_trace(TRACE_URB,
					   "  %6.1f", t->grf[reg + i].f[c]);
			ksim_trace(TRACE_URB, "\n");
		}
	}

	uint32_t c;
	for_each_bit (c, t->mask) {
		struct value *vue = urb_handle_to_entry(t->grf[reg].ud[c]);
		for (int i = 0; i < mlen - 1; i++)
			vue[offset + i / 4].v[i % 4] = t->grf[reg + 1 + i].ud[c];
	}
}

void
sfid_urb(struct thread *t, const struct send_args *args)
{
	uint32_t opcode = field(args->function_control, 0, 3);
	uint32_t global_offset = field(args->function_control, 4, 14);
	bool per_slot_offset = field(args->function_control, 17, 17);

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
		sfid_urb_simd8_write(t, args->src, global_offset, args->mlen);
		break;
	}
}
