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

static int
builder_disasm_printf(void *_bld, const char *fmt, ...)
{
	struct builder *bld = _bld;
	va_list va;
	int length;

	va_start(va, fmt);
	length = vsnprintf(bld->disasm_output + bld->disasm_length,
			   sizeof(bld->disasm_output) - bld->disasm_length, fmt, va);
	va_end(va);
	bld->disasm_length += length;

	return length;
}

void
builder_init(struct builder *bld, uint64_t surfaces, uint64_t samplers)
{
	bld->shader = align_ptr(shader_end, 64);
	bld->p = bld->shader->code;
	bld->pool_index = 0;
	bld->binding_table_address = surfaces;
	bld->sampler_state_address = samplers;

	bld->disasm_tail = bld->p - (uint8_t *) shader_pool;
	init_disassemble_info(&bld->info, bld, builder_disasm_printf);
	/* info.print_address_func = override_print_address; */
	bld->info.arch = bfd_arch_i386;
	bld->info.mach = bfd_mach_x86_64;
	bld->info.buffer_vma = 0;
	bld->info.buffer_length = shader_pool_size;
	bld->info.buffer = shader_pool;
	bld->info.section = NULL;
	disassemble_init_for_target(&bld->info);

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

bool
builder_disasm(struct builder *bld)
{
	const int end = bld->p - (uint8_t *) shader_pool;

	bld->disasm_length = 0;
	if (bld->disasm_tail < end) {
		bld->disasm_tail +=
			print_insn_i386(bld->disasm_tail, &bld->info);
		return true;
	} else {
		return false;
	}
}
