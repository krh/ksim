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

#include <stddef.h>
#include <bfd.h>
#include <dis-asm.h>

#include "ksim.h"
#include "avx-builder.h"

static void *shader_pool, *shader_end;
const size_t shader_pool_size = 64 * 1024;

void
reset_shader_pool(void)
{
	if (shader_pool == NULL) {
		int fd = memfd_create("jit", MFD_CLOEXEC);
		ftruncate(fd, shader_pool_size);
		shader_pool = mmap(NULL, shader_pool_size,
				   PROT_WRITE | PROT_READ | PROT_EXEC,
				   MAP_SHARED, fd, 0);
		close(fd);
	}

	shader_end = shader_pool;
}

void
builder_init(struct builder *bld, uint64_t surfaces, uint64_t samplers)
{
	bld->shader = align_ptr(shader_end, 64);
	bld->p = bld->shader->code;
	bld->pool_index = 0;
	bld->binding_table_address = surfaces;
	bld->sampler_state_address = samplers;

	list_init(&bld->regs_lru_list);
	list_init(&bld->used_regs_list);

	for (int i = 0; i < ARRAY_LENGTH(bld->regs); i++)
		list_insert(&bld->regs_lru_list, &bld->regs[i].link);
}

void
builder_finish(struct builder *bld)
{
	shader_end = bld->p;

	ksim_assert(shader_end - shader_pool < shader_pool_size);
}

int
builder_get_reg(struct builder *bld)
{
	struct avx2_reg *reg = container_of(bld->regs_lru_list.prev, reg, link);

	ksim_assert(!list_empty(&bld->regs_lru_list));

	list_remove(&reg->link);
	list_insert(&bld->used_regs_list, &reg->link);

	return reg - bld->regs;
}

void
builder_release_regs(struct builder *bld)
{
	list_insert_list(bld->regs_lru_list.prev, &bld->used_regs_list);
	list_init(&bld->used_regs_list);
}

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
