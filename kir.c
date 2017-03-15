/*
 * Copyright © 2016 Kristian H. Kristensen <hoegsberg@gmail.com>
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

#include <string.h>

#include "eu.h"
#include "avx-builder.h"
#include "kir.h"

struct kir_insn *
kir_insn_create(uint32_t opcode, struct kir_reg dst, struct list *other)
{
	struct kir_insn *insn;

	insn = malloc(sizeof(*insn));
	insn->opcode = opcode;
	insn->dst = dst;
	if (other)
		list_insert(other, &insn->link);

	return insn;
}

struct kir_insn *
kir_program_add_insn(struct kir_program *prog, uint32_t opcode)
{
	struct kir_reg dst = kir_reg(prog->next_reg.n++);
	struct kir_insn *insn = kir_insn_create(opcode, dst, prog->insns.prev);

	prog->dst = dst;
	insn->scope = prog->scope;

	return insn;
}

void
kir_program_comment(struct kir_program *prog, const char *fmt, ...)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_comment);

	va_list va;

	va_start(va, fmt);
	vasprintf(&insn->comment, fmt, va);
	va_end(va);
}

struct kir_reg
kir_program_load_region(struct kir_program *prog, const struct eu_region *region)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_load_region);

	insn->xfer.region = *region;

	return insn->dst;
}

void
kir_program_store_region_mask(struct kir_program *prog, const struct eu_region *region,
			      struct kir_reg src, struct kir_reg mask)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_store_region_mask);

	insn->xfer.region = *region;
	insn->xfer.src = src;
	insn->xfer.mask = mask;
}

void
kir_program_store_region(struct kir_program *prog, const struct eu_region *region, struct kir_reg src)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_store_region);

	insn->xfer.region = *region;
	insn->xfer.src = src;
}

struct kir_reg
kir_program_alu(struct kir_program *prog, enum kir_opcode opcode, ...)
{
	struct kir_insn *insn = kir_program_add_insn(prog, opcode);
	va_list va;

	va_start(va, opcode);
	insn->alu.src0 = va_arg(va, struct kir_reg);
	insn->alu.src1 = va_arg(va, struct kir_reg);
	insn->alu.src2 = va_arg(va, struct kir_reg);
	va_end(va);

	return insn->dst;
}

void
__kir_program_send(struct kir_program *prog, struct inst *inst,
		   enum kir_opcode opcode, void *func, void *args)
{
	struct inst_send send = unpack_inst_send(inst);
	struct kir_insn *insn = kir_program_add_insn(prog, kir_send);

	insn->send.src = unpack_inst_2src_src0(inst).num;
	insn->send.mlen = send.mlen;
	insn->send.dst = unpack_inst_2src_dst(inst).num;
	insn->send.rlen = send.rlen;
	insn->send.func = func;
	insn->send.args = args;
}

struct kir_reg
kir_program_call(struct kir_program *prog, void *func, uint32_t args, ...)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_call);
	va_list va;

	insn->call.func = func;
	insn->call.args = args;
	
	va_start(va, args);
	insn->call.src0 = va_arg(va, struct kir_reg);
	insn->call.src1 = va_arg(va, struct kir_reg);
	va_end(va);

	return insn->dst;
}

struct kir_reg
kir_program_const_call(struct kir_program *prog, void *func, uint32_t args, ...)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_const_call);
	va_list va;

	insn->call.func = func;
	insn->call.args = args;
	
	va_start(va, args);
	insn->call.src0 = va_arg(va, struct kir_reg);
	insn->call.src1 = va_arg(va, struct kir_reg);
	va_end(va);

	return insn->dst;
}

struct kir_reg
kir_program_gather(struct kir_program *prog, 
		   struct kir_reg base,
		   struct kir_reg offset, struct kir_reg mask,
		   uint32_t scale, uint32_t base_offset)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_gather);

	insn->gather.base = base;
	insn->gather.offset = offset;
	insn->gather.mask = mask;
	insn->gather.scale = scale;
	insn->gather.base_offset = base_offset;

	return insn->dst;
}

struct kir_reg
kir_program_set_load_base_indirect(struct kir_program *prog, uint32_t offset)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_set_load_base_indirect);

	insn->set_load_base.offset = offset;

	return insn->dst;
}

struct kir_reg
kir_program_set_load_base_imm(struct kir_program *prog, void *pointer)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_set_load_base_imm);

	insn->set_load_base.pointer = pointer;

	return insn->dst;
}

struct kir_reg
kir_program_set_load_base_imm_offset(struct kir_program *prog, void *pointer, struct kir_reg src)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_set_load_base_imm_offset);

	insn->set_load_base.pointer = pointer;
	insn->set_load_base.src = src;

	return insn->dst;
}

struct kir_reg
kir_program_load(struct kir_program *prog, struct kir_reg base, uint32_t offset)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_load);

	insn->load.base = base;
	insn->load.offset = offset;

	return insn->dst;
}

void
kir_program_mask_store(struct kir_program *prog,
		       struct kir_reg base, uint32_t offset,
		       struct kir_reg src, struct kir_reg mask)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_mask_store);

	insn->store.base = base;
	insn->store.offset = offset;
	insn->store.src = src;
	insn->store.mask = mask;
}

static char *
format_region(char *buf, int len, struct eu_region *region)
{
	snprintf(buf, len,
		 "g%d.%d<%d,%d,%d>%d",
		 region->offset / 32, region->offset & 31,
		 region->vstride, region->width, region->hstride,
		 region->type_size);

	return buf;
}

static const char *
kir_insn_format(struct kir_insn *insn, char *buf, size_t size)
{
	char region[32];
	int len;

	switch (insn->opcode) {
	case kir_comment:
		snprintf(buf, size, "# %s", insn->comment);
		break;
	case kir_load_region:
		snprintf(buf, size, "r%-3d = load_region %s",
			 insn->dst.n,
			 format_region(region, sizeof(region), &insn->xfer.region));
		break;
	case kir_store_region_mask:
		snprintf(buf, size, "       store_region_mask r%d, r%d, %s",
			 insn->xfer.mask.n, insn->xfer.src.n,
			 format_region(region, sizeof(region), &insn->xfer.region));
		break;
	case kir_store_region:
		snprintf(buf, size, "       store_region r%d, %s",
			 insn->xfer.src.n,
			 format_region(region, sizeof(region), &insn->xfer.region));
		break;
	case kir_set_load_base_indirect:
		snprintf(buf, size, "r%-3d = set_load_base (%d)",
			 insn->dst.n, insn->set_load_base.offset);
		break;
	case kir_set_load_base_imm:
		snprintf(buf, size, "r%-3d = set_load_base %p",
			 insn->dst.n, insn->set_load_base.pointer);
		break;
	case kir_set_load_base_imm_offset:
		snprintf(buf, size, "r%-3d = set_load_base %p + r%d.2",
			 insn->dst.n, insn->set_load_base.pointer, insn->set_load_base.src.n);
		break;
	case kir_load:
		snprintf(buf, size, "r%-3d = load %d(r%d)",
			 insn->dst.n, insn->load.offset, insn->load.base.n);
		break;
	case kir_mask_store:
		snprintf(buf, size, "       mask_store r%d, r%d, %d(r%d)",
			 insn->store.mask.n, insn->store.src.n,
			 insn->store.offset, insn->store.base.n);
		break;
	case kir_immd:
		snprintf(buf, size, "r%-3d = imm %dd %ff", insn->dst.n, insn->imm.d,
			 u32_to_float(insn->imm.d));
		break;
	case kir_immw:
		snprintf(buf, size, "r%-3d = imm %dw", insn->dst.n, insn->imm.d);
		break;
	case kir_immv:
		snprintf(buf, size, "r%-3d = imm [ %d, %d, %d, %d, %d, %d, %d, %d ]",
			 insn->dst.n,
			 insn->imm.v[0], insn->imm.v[1],
			 insn->imm.v[2], insn->imm.v[3],
			 insn->imm.v[4], insn->imm.v[5],
			 insn->imm.v[6], insn->imm.v[7]);
		break;
	case kir_immvf:
		snprintf(buf, size, "r%-3d = imm [ %f, %f, %f, %f ]",
			 insn->dst.n,
			 insn->imm.vf[0], insn->imm.vf[1],
			 insn->imm.vf[2], insn->imm.vf[3]);
		break;
	case kir_send:
	case kir_const_send:
		len = snprintf(buf, size, "       %ssend src g%d-g%d",
			       insn->opcode == kir_const_send ? "const_" : "",
			       insn->send.src, insn->send.src + insn->send.mlen - 1);
		if (insn->send.rlen > 0)
			len += snprintf(buf + len, size - len, ", dst g%d-g%d",
					insn->send.dst, insn->send.dst + insn->send.rlen - 1);
		break;
	case kir_call:
	case kir_const_call:
		len = snprintf(buf, size, "r%-3d = %scall %p",
			       insn->dst.n,
			       insn->opcode == kir_const_call ? "const_" : "",
			       insn->call.func);
		if (insn->call.args > 0)
			len += snprintf(buf + len, size - len,
					", r%d", insn->call.src0.n);
		if (insn->call.args > 1)
			len += snprintf(buf + len, size - len,
					", r%d", insn->call.src1.n);
		break;
	case kir_mov:
		snprintf(buf, size, "r%-3d = mov r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_zxwd:
		snprintf(buf, size, "r%-3d = zxwd r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_sxwd:
		snprintf(buf, size, "r%-3d = sxwd r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_ps2d:
		snprintf(buf, size, "r%-3d = ps2d r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_d2ps:
		snprintf(buf, size, "r%-3d = d2ps r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_absd:
		snprintf(buf, size, "r%-3d = absd r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rcp:
		snprintf(buf, size, "r%-3d = rcp r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_sqrt:
		snprintf(buf, size, "r%-3d = sqrt r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rsqrt:
		snprintf(buf, size, "r%-3d = rsqrt r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rndu:
		snprintf(buf, size, "r%-3d = rndu r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rndd:
		snprintf(buf, size, "r%-3d = rndd r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rnde:
		snprintf(buf, size, "r%-3d = rnde r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rndz:
		snprintf(buf, size, "r%-3d = rndz r%d", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_and:
		snprintf(buf, size, "r%-3d = and r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_andn:
		snprintf(buf, size, "r%-3d = andn r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_or:
		snprintf(buf, size, "r%-3d = or r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_xor:
		snprintf(buf, size, "r%-3d = xor r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_shri:
		snprintf(buf, size, "r%-3d = shri r%d, %d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_shr:
		snprintf(buf, size, "r%-3d = shr r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_shli:
		snprintf(buf, size, "r%-3d = shli r%d, %d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.imm1);
		break;
	case kir_shl:
		snprintf(buf, size, "r%-3d = shl r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_asr:
		snprintf(buf, size, "r%-3d = asr r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_maxd:
		snprintf(buf, size, "r%-3d = maxd r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_maxw:
		snprintf(buf, size, "r%-3d = maxw r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_maxf:
		snprintf(buf, size, "r%-3d = maxf r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_mind:
		snprintf(buf, size, "r%-3d = mind r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_minw:
		snprintf(buf, size, "r%-3d = minw r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_minf:
		snprintf(buf, size, "r%-3d = minf r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_divf:
		snprintf(buf, size, "r%-3d = divf r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_int_div_q_and_r:
		stub("int_div_q_and_r");
		break;
	case kir_int_div_q:
		stub("int_div_q");
		break;
	case kir_int_div_r:
		stub("int_div_r");
		break;
	case kir_int_invm:
		stub("int_invm");
		break;
	case kir_int_rsqrtm:
		stub("int_rsqrtm");
		break;
	case kir_addd:
		snprintf(buf, size, "r%-3d = addd r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_addw:
		snprintf(buf, size, "r%-3d = addw r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_addf:
		snprintf(buf, size, "r%-3d = addf r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;

	case kir_subd:
		snprintf(buf, size, "r%-3d = subd r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_subw:
		snprintf(buf, size, "r%-3d = subw r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_subf:
		snprintf(buf, size, "r%-3d = subf r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_muld:
		snprintf(buf, size, "r%-3d = muld r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_mulw:
		snprintf(buf, size, "r%-3d = mulw r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_mulf:
		snprintf(buf, size, "r%-3d = mulf r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_maddf:
		snprintf(buf, size, "r%-3d = maddf r%d, r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n, insn->alu.src2.n);
		break;
	case kir_nmaddf:
		snprintf(buf, size, "r%-3d = nmaddf r%d, r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n, insn->alu.src2.n);
		break;
	case kir_cmp:
		snprintf(buf, size, "r%-3d = cmp r%d, r%d, op %d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n, insn->alu.imm2);
		break;
	case kir_blend:
		snprintf(buf, size, "r%-3d = blend r%d, r%d, r%d", insn->dst.n,
			 insn->alu.src0.n, insn->alu.src1.n, insn->alu.src2.n);
		break;
	case kir_gather:
		snprintf(buf, size, "r%-3d = gather r%d, %d(r%d,r%d,%d)",
			 insn->dst.n,
			 insn->gather.mask.n,
			 insn->gather.base_offset,
			 insn->gather.offset.n, insn->gather.base.n, insn->gather.scale);
		break;
	case kir_eot:
		snprintf(buf, size, "       eot");
		break;
	case kir_eot_if_dead:
		snprintf(buf, size, "       eot_if_dead r%d", insn->eot.src.n);
		break;
	}

	return buf;
}

void
kir_program_print(struct kir_program *prog, FILE *fp)
{
	struct kir_insn *insn;
	char buf[128];

	list_for_each_entry(insn, &prog->insns, link)
		fprintf(fp, "%s\n", kir_insn_format(insn, buf, sizeof(buf)));
}

static void
region_to_mask(struct eu_region *region, uint32_t mask[2])
{
	uint32_t type_mask = (1 << region->type_size) - 1;
	uint32_t x = 0, y = 0;

	mask[0] = 0;
	mask[1] = 0;
	for (uint32_t i = 0; i < region->exec_size; i++) {
		uint32_t offset = (region->offset & 31) +
			(x * region->hstride + y * region->vstride) * region->type_size;
		ksim_assert((offset & 31) + region->type_size <= 32);
		ksim_assert(offset < 64);
		mask[offset / 32] |= type_mask << (offset & 31);
		x++;
		if (x == region->width) {
			x = 0;
			y++;
		}
	}
}

static bool
region_is_live(struct eu_region *region, uint32_t *region_map)
{
	uint32_t mask[2];
	int reg = region->offset / 32;

	ksim_assert(reg < 512);

	region_to_mask(region, mask);

	return (region_map[reg] & mask[0]) || (region_map[reg + 1] & mask[1]);
}

static void
set_region_live(struct eu_region *region, bool live, uint32_t *region_map)
{
	uint32_t mask[2];
	uint32_t reg = region->offset / 32;

	ksim_assert(reg < 512);

	region_to_mask(region, mask);
	if (live) {
		region_map[reg] |= mask[0];
		region_map[reg + 1] |= mask[1];
	} else {
		region_map[reg] &= ~mask[0];
		region_map[reg + 1] &= ~mask[1];
	}
}


static inline void
set_live(struct kir_reg r, bool live, struct kir_insn *insn, uint32_t *range, bool *live_regs)
{
	if (live) {
		if (!live_regs[r.n])
			range[r.n] = insn->dst.n;
		live_regs[r.n] = true;
	}
}

static struct eu_region
region_for_reg(int reg)
{
	return (struct eu_region) {
		.offset = reg * 32,
		.type_size = 4,
		.exec_size = 8,
		.vstride = 8,
		.width = 8,
		.hstride = 1
	};
}

void
kir_program_compute_live_ranges(struct kir_program *prog)
{
	struct kir_insn *insn;
	uint32_t *range;
	bool *live_regs;
	uint32_t region_map[512];
	int count = prog->next_reg.n;

	live_regs = malloc(count * sizeof(live_regs[0]));
	memset(live_regs, 0, count * sizeof(live_regs[0]));
	range = malloc(count * sizeof(range[0]));
	memset(range, 0, count * sizeof(range[0]));
	memset(region_map, 0, 512 * sizeof(region_map[0]));

	/* Initialize URB buffer live if we have one. URB offset
	 * and size are in bytes. */
	memset(region_map + prog->urb_offset / 32, ~0,
	       prog->urb_length / 32 * sizeof(region_map[0]));

	insn = container_of(prog->insns.prev, insn, link);
	while (&insn->link != &prog->insns) {
		bool live = false;
		switch (insn->opcode) {
		case kir_comment:
			range[insn->dst.n] = insn->dst.n + 1;
			break;
		case kir_load_region:
			live = live_regs[insn->dst.n];
			if (live)
				set_region_live(&insn->xfer.region, live, region_map);
			break;
		case kir_store_region_mask:
			live = region_is_live(&insn->xfer.region, region_map);
			set_live(insn->xfer.src, live, insn, range, live_regs);
			set_live(insn->xfer.mask, live, insn, range, live_regs);
			if (live)
				range[insn->dst.n] = insn->dst.n + 1;
			set_region_live(&insn->xfer.region, false, region_map);
			break;
		case kir_store_region:
			live = region_is_live(&insn->xfer.region, region_map);
			set_live(insn->xfer.src, live, insn, range, live_regs);
			if (live)
				range[insn->dst.n] = insn->dst.n + 1;
			set_region_live(&insn->xfer.region, false, region_map);
			break;
		case kir_set_load_base_indirect:
		case kir_set_load_base_imm:
			break;
		case kir_set_load_base_imm_offset:
			live = live_regs[insn->dst.n];
			set_live(insn->set_load_base.src, live, insn, range, live_regs);
			break;
		case kir_load:
			live = live_regs[insn->dst.n];
			set_live(insn->load.base, live, insn, range, live_regs);
			break;
		case kir_mask_store:
			live = true;
			set_live(insn->store.src, live, insn, range, live_regs);
			set_live(insn->store.mask, live, insn, range, live_regs);
			set_live(insn->store.base, live, insn, range, live_regs);
			if (live)
				range[insn->dst.n] = insn->dst.n + 1;
			break;
		case kir_immd:
		case kir_immw:
		case kir_immv:
		case kir_immvf:
			break;
		case kir_send:
		case kir_const_send:
			live = insn->opcode == kir_send ? true : false;
			for (uint32_t i = 0; i < insn->send.rlen; i++) {
				struct eu_region region = region_for_reg(insn->send.dst + i);
				live |= region_is_live(&region, region_map);
				set_region_live(&region, false, region_map);
			}
			if (live)
				range[insn->dst.n] = insn->dst.n + 1;

			for (uint32_t i = 0; i < insn->send.mlen; i++) {
				struct eu_region region = region_for_reg(insn->send.src + i);
				set_region_live(&region, live, region_map);
			}

			{
				/* The send helper typically need to
				 * read the mask register. */
				struct eu_region region = {
					.offset = offsetof(struct thread, mask_stack[insn->scope]),
					.type_size = 4,
					.exec_size = 8,
					.vstride = 8,
					.width = 8,
					.hstride = 1
				};
				set_region_live(&region, live, region_map);
			}

			break;
		case kir_call:
		case kir_const_call:
			if (insn->opcode == kir_call) {
				live = true;
				range[insn->dst.n] = insn->dst.n + 1;
			} else {
				live = live_regs[insn->dst.n];
			}

			if (insn->call.args > 0)
				set_live(insn->call.src0, live, insn, range, live_regs);
			if (insn->call.args > 1)
				set_live(insn->call.src1, live, insn, range, live_regs);
			break;
		case kir_mov:
			ksim_unreachable();
			break;
		case kir_zxwd:
		case kir_sxwd:
		case kir_ps2d:
		case kir_d2ps:
		case kir_absd:
		case kir_rcp:
		case kir_sqrt:
		case kir_rsqrt:
		case kir_rndu:
		case kir_rndd:
		case kir_rnde:
		case kir_rndz:
		case kir_shri:
		case kir_shli:
			live = live_regs[insn->dst.n];
			set_live(insn->alu.src0, live, insn, range, live_regs);
			break;
		case kir_and:
		case kir_andn:
		case kir_or:
		case kir_xor:
		case kir_shr:
		case kir_shl:
		case kir_asr:
		case kir_maxd:
		case kir_maxw:
		case kir_maxf:
		case kir_mind:
		case kir_minw:
		case kir_minf:
		case kir_divf:
		case kir_addd:
		case kir_addw:
		case kir_addf:
		case kir_subd:
		case kir_subw:
		case kir_subf:
		case kir_muld:
		case kir_mulw:
		case kir_mulf:
		case kir_cmp:
			live = live_regs[insn->dst.n];
			set_live(insn->alu.src0, live, insn, range, live_regs);
			set_live(insn->alu.src1, live, insn, range, live_regs);
			break;
		case kir_int_div_q_and_r:
		case kir_int_div_q:
		case kir_int_div_r:
		case kir_int_invm:
		case kir_int_rsqrtm:
			break;
		case kir_maddf:
		case kir_nmaddf:
		case kir_blend:
			live = live_regs[insn->dst.n];
			set_live(insn->alu.src0, live, insn, range, live_regs);
			set_live(insn->alu.src1, live, insn, range, live_regs);
			set_live(insn->alu.src2, live, insn, range, live_regs);
			break;
		case kir_gather:
			live = live_regs[insn->dst.n];
			set_live(insn->gather.mask, live, insn, range, live_regs);
			set_live(insn->gather.offset, live, insn, range, live_regs);
			set_live(insn->gather.base, live, insn, range, live_regs);
			break;
		case kir_eot:
			range[insn->dst.n] = insn->dst.n + 1;
			break;
		case kir_eot_if_dead:
			set_live(insn->eot.src, true, insn, range, live_regs);
			range[insn->dst.n] = insn->dst.n + 1;

			/* FIXME: assert that rax is not live */
			break;

		}

		insn = container_of(insn->link.prev, insn, link);
	}

	free(live_regs);

	prog->live_ranges = range;
}

struct resident_region {
	uint32_t mask[2]; /* bitmask of region */
	struct kir_reg reg;
	struct list link;
};

void
kir_program_copy_propagation(struct kir_program *prog)
{
	struct kir_insn *insn;
	struct resident_region *rr, *next, *rr_pool;
	int count = prog->next_reg.n, rr_pool_next;
	struct kir_reg *remap;

	remap = malloc(count * sizeof(remap[0]));
	for (uint32_t i = 0; i < count; i++)
		remap[i] = kir_reg(i);

	/* We allocate these up front instead a malloc call per
	 * insn. More efficient and they're easier to clean up. Each
	 * insn allocates at most 1 so we won't need more than count. */
	rr_pool = malloc(count * sizeof(rr_pool[0]));
	rr_pool_next = 0;

	const uint32_t max_eu_regs = 400;
	struct list region_to_reg[max_eu_regs];
	for (uint32_t i = 0; i < max_eu_regs; i++)
		list_init(&region_to_reg[i]);

	list_for_each_entry(insn, &prog->insns, link) {
		uint32_t mask[2];
		switch (insn->opcode) {
		case kir_comment:
			break;
		case kir_load_region: {
			uint32_t grf = insn->xfer.region.offset / 32;
			ksim_assert(grf < max_eu_regs);
			region_to_mask(&insn->xfer.region, mask);
			list_for_each_entry(rr, &region_to_reg[grf], link) {
				if (rr->mask[0] == mask[0] && rr->mask[1] == mask[1]) {
					remap[insn->dst.n] = rr->reg;
					goto load_region_done;
				}
			}

			/* FIXME: Insert an rr for each region_to_reg it overlaps */
			rr = &rr_pool[rr_pool_next++];
			rr->mask[0] = mask[0];
			rr->mask[1] = mask[1];
			rr->reg = insn->dst;
			list_insert(&region_to_reg[grf], &rr->link);
		load_region_done:
			break;
		}
		case kir_store_region_mask: {
			uint32_t grf = insn->xfer.region.offset / 32;

			insn->xfer.src = remap[insn->xfer.src.n];
			insn->xfer.mask = remap[insn->xfer.mask.n];

			ksim_assert(grf < max_eu_regs);
			region_to_mask(&insn->xfer.region, mask);

			/* Invalidate registers overlapping region */
			list_for_each_entry_safe(rr, next, &region_to_reg[grf], link) {
				if ((mask[0] & rr->mask[0]) || (mask[1] & rr->mask[1]))
					list_remove(&rr->link);
			}

			break;
		}

		case kir_store_region: {
			uint32_t grf = insn->xfer.region.offset / 32;

			insn->xfer.src = remap[insn->xfer.src.n];

			ksim_assert(grf < max_eu_regs);
			region_to_mask(&insn->xfer.region, mask);

			/* Invalidate registers overlapping region */
			list_for_each_entry_safe(rr, next, &region_to_reg[grf], link) {
				if ((mask[0] & rr->mask[0]) || (mask[1] & rr->mask[1]))
					list_remove(&rr->link);
			}
			if (mask[1]) {
				ksim_assert(grf + 1 < max_eu_regs);
				list_for_each_entry_safe(rr, next, &region_to_reg[grf + 1], link) {
					if (mask[1] & rr->mask[0])
						list_remove(&rr->link);
				}
			}

			rr = &rr_pool[rr_pool_next++];
			rr->mask[0] = mask[0];
			rr->mask[1] = mask[1];
			rr->reg = insn->xfer.src;
			list_insert(&region_to_reg[grf], &rr->link);
			break;
		}
		case kir_set_load_base_indirect:
		case kir_set_load_base_imm:
			/* Detect duplicate base loads here. */
			break;
		case kir_set_load_base_imm_offset:
			insn->set_load_base.src = remap[insn->set_load_base.src.n];
			break;
		case kir_load:
			insn->load.base = remap[insn->load.base.n];
			break;
		case kir_mask_store:
			insn->store.base = remap[insn->store.base.n];
			break;

		case kir_immd:
		case kir_immw:
		case kir_immv:
		case kir_immvf:
			break;
		case kir_send:
		case kir_const_send:
			/* Invalidate registers overlapping region */
			for (uint32_t i = 0; i < insn->send.rlen; i++) {
				uint32_t grf = (insn->send.dst + i);
				struct list *head = &region_to_reg[grf];
				list_for_each_entry_safe(rr, next, head, link)
					list_remove(&rr->link);
			}
			break;
		case kir_call:
		case kir_const_call:
			if (insn->call.args == 1) {
				insn->call.src0 = remap[insn->call.src0.n];
			} else if (insn->call.args == 2) {
				insn->call.src0 = remap[insn->call.src0.n];
				insn->call.src1 = remap[insn->call.src1.n];
			}
			break;

		case kir_mov:
			ksim_unreachable();
			break;
		case kir_zxwd:
		case kir_sxwd:
		case kir_ps2d:
		case kir_d2ps:
		case kir_absd:
		case kir_rcp:
		case kir_sqrt:
		case kir_rsqrt:
		case kir_rndu:
		case kir_rndd:
		case kir_rnde:
		case kir_rndz:
		case kir_shri:
		case kir_shli:
			insn->alu.src0 = remap[insn->alu.src0.n];
			break;
		case kir_and:
		case kir_andn:
		case kir_or:
		case kir_xor:
		case kir_shr:
		case kir_shl:
		case kir_asr:
		case kir_maxd:
		case kir_maxw:
		case kir_maxf:
		case kir_mind:
		case kir_minw:
		case kir_minf:
		case kir_divf:
		case kir_addd:
		case kir_addw:
		case kir_addf:
		case kir_subd:
		case kir_subw:
		case kir_subf:
		case kir_muld:
		case kir_mulw:
		case kir_mulf:
		case kir_cmp:
			insn->alu.src0 = remap[insn->alu.src0.n];
			insn->alu.src1 = remap[insn->alu.src1.n];
			break;
		case kir_int_div_q_and_r:
		case kir_int_div_q:
		case kir_int_div_r:
		case kir_int_invm:
		case kir_int_rsqrtm:
			break;
		case kir_maddf:
		case kir_nmaddf:
		case kir_blend:
			insn->alu.src0 = remap[insn->alu.src0.n];
			insn->alu.src1 = remap[insn->alu.src1.n];
			insn->alu.src2 = remap[insn->alu.src2.n];
			break;
		case kir_gather:
			/* Don't copy propagate mask: gather overwrites
			 * the mask and we need a fresh copy each time. */
			insn->gather.offset = remap[insn->gather.offset.n];
			insn->gather.base = remap[insn->gather.base.n];
			break;
		case kir_eot:
			break;
		case kir_eot_if_dead:
			insn->eot.src = remap[insn->eot.src.n];
			break;
		}
	}

	free(remap);
	free(rr_pool);
}

void
kir_insn_destroy(struct kir_insn *insn)
{
	if (insn->opcode == kir_comment)
		free(insn->comment);
	free(insn);
}

void
kir_program_dead_code_elimination(struct kir_program *prog)
{
	struct kir_insn *insn, *next;
	uint32_t *range = prog->live_ranges;

	list_for_each_entry_safe(insn, next, &prog->insns, link) {
		if (insn->dst.n >= range[insn->dst.n]) {
			list_remove(&insn->link);
			kir_insn_destroy(insn);
		}
	}
}

struct bit_vector {
	uint64_t bits[2];
};

static void
bit_vector_init(struct bit_vector *v)
{
	v->bits[0] = ~0ul;
	v->bits[1] = ~0ul;
}

static uint32_t
bit_vector_alloc(struct bit_vector *v)
{
	for (uint32_t i = 0; i < ARRAY_LENGTH(v->bits); i++) {
		if (v->bits[i]) {
			uint32_t b = __builtin_ffsl(v->bits[i]) - 1;
			v->bits[i] &= ~(1ul << b);
			return b + i * 64;
		}
	}

	ksim_unreachable();
	return 0;
}

static void
bit_vector_free(struct bit_vector *v, uint32_t b)
{
	ksim_assert((v->bits[b >> 6] & (1ul << (b & 63))) == 0);
	v->bits[b >> 6] |= 1ul << (b & 63);
}

struct ra_state {
	uint32_t *range;
	uint32_t regs;
	uint8_t *reg_to_avx;
	struct kir_reg avx_to_reg[16];
	struct bit_vector spill_slots;
	uint32_t locked_regs;	/* Don't spill these */
	uint32_t exclude_regs;	/* Don't allocate these */

	uint32_t next_reg;
	uint32_t next_spill_reg;
};

static const struct kir_reg void_reg = { };

static int
pick_spill_reg(struct ra_state *state)
{
	uint32_t start_reg = state->next_spill_reg++ & 15;
	uint32_t regs = 0xffff ^ state->locked_regs;

	if (regs >> start_reg)
		return __builtin_ffs(regs >> start_reg) + start_reg - 1;
	else
		return __builtin_ffs(regs) - 1;
}

/* Insert spill instruction of register reg before instruction insn */
static void
spill_reg(struct ra_state *state, struct kir_insn *insn, int avx_reg)
{
	int slot = bit_vector_alloc(&state->spill_slots);

	ksim_trace(TRACE_RA, "\tspill ymm%d to slot %d\n", avx_reg, slot);

	/* FIXME: Don't spill regs that are simple region loads or
	 * immediates, just make the unspill reload.  Not trivial for
	 * regions as they're not SSA.  Prefer spilling one of these
	 * regs when possible. */

	struct kir_insn *spill =
		kir_insn_create(kir_store_region, void_reg, insn->link.prev);

	spill->xfer.src = kir_reg(avx_reg);
	spill->xfer.region = (struct eu_region) {
		.offset = offsetof(struct thread, spill[slot]),
		.type_size = 4,
		.exec_size = 8,
		.vstride = 8,
		.width = 8,
		.hstride = 1
	};

	struct kir_reg def = state->avx_to_reg[avx_reg];

	state->regs |= (1 << avx_reg);
	state->reg_to_avx[def.n] = 16 + slot;
}

static void
assign_reg(struct ra_state *state, struct kir_insn *insn, int avx_reg)
{
	ksim_assert(avx_reg < 16);

	state->avx_to_reg[avx_reg] = insn->dst;
	state->reg_to_avx[insn->dst.n] = avx_reg;
	insn->dst.n = avx_reg;
	state->regs &= ~(1 << avx_reg);
}

static void
unspill_reg(struct ra_state *state, struct kir_insn *insn, struct kir_reg reg)
{
	uint32_t regs = state->regs & ~state->locked_regs;

	if (regs == 0) {
		/* Spill something else if we don't have a register
		 * for unspilling into. */
		int n = pick_spill_reg(state);
		spill_reg(state, insn, n);
		regs = state->regs & ~state->locked_regs;
	}

	ksim_assert(regs);

	int avx_reg = __builtin_ffs(regs) - 1;
	uint32_t slot = state->reg_to_avx[reg.n] - 16;
	bit_vector_free(&state->spill_slots, slot);

	ksim_trace(TRACE_RA, "\tunspill slot %d to ymm%d\n", slot, avx_reg);

	struct kir_insn *unspill =
		kir_insn_create(kir_load_region, reg, insn->link.prev);

	unspill->xfer.region = (struct eu_region) {
		.offset = offsetof(struct thread, spill[slot]),
		.type_size = 4,
		.exec_size = 8,
		.vstride = 8,
		.width = 8,
		.hstride = 1
	};

	assign_reg(state, unspill, avx_reg);
}

static inline bool
reg_dead(struct ra_state *state, struct kir_insn *insn, struct kir_reg reg)
{
	return insn->dst.n >= state->range[reg.n];
}

static inline void
lock_reg(struct ra_state *state, struct kir_reg reg)
{
	if (state->reg_to_avx[reg.n] < 16)
		state->locked_regs |= (1 << state->reg_to_avx[reg.n]);
}

static inline struct kir_reg
use_reg(struct ra_state *state, struct kir_insn *insn, struct kir_reg reg)
{
	/* Don't use a register that hasn't been assigned anything yet. */
	ksim_assert(state->reg_to_avx[reg.n] != 0xff);

	if (state->reg_to_avx[reg.n] >= 16)
		unspill_reg(state, insn, reg);

	struct kir_reg avx_reg = {
		.n = state->reg_to_avx[reg.n]
	};

	ksim_assert(avx_reg.n != 0xff);
	if (reg_dead(state, insn, reg)) {
		ksim_trace(TRACE_RA, "\tuse ymm%d for r%d, dead now\n",
			   avx_reg.n, reg.n);
		state->regs |= (1 << avx_reg.n);
	} else {
		ksim_trace(TRACE_RA, "\tuse ymm%d for r%d\n",
			   avx_reg.n, reg.n);
	}

	state->locked_regs |= (1 << avx_reg.n);

	return avx_reg;
}

static void
spill_all(struct ra_state *state, struct kir_insn *insn)
{
	uint32_t live_regs = 0xffff & ~state->regs;
	uint32_t avx_reg;

	for_each_bit(avx_reg, live_regs)
		spill_reg(state, insn, avx_reg);
}

static void
allocate_reg(struct ra_state *state, struct kir_insn *insn)
{
	uint32_t regs = state->regs & ~state->exclude_regs;
	if (regs == 0) {
		int n = pick_spill_reg(state);
		spill_reg(state, insn, n);
		regs = state->regs;
	}
	uint32_t start_reg = state->next_reg & 15;
	int avx_reg;

	if (regs >> start_reg)
		avx_reg = __builtin_ffs(regs >> start_reg) + start_reg - 1;
	else
		avx_reg = __builtin_ffs(regs) - 1;

	ksim_trace(TRACE_RA, "\tallocate ymm%d for r%d\n",
		   avx_reg, insn->dst.n, state->next_reg);
	state->next_reg++;

	assign_reg(state, insn, avx_reg);
}

static void
kir_program_allocate_registers(struct kir_program *prog)
{
	struct kir_insn *insn;
	struct ra_state state;
	char buf[128];
	int count = prog->next_reg.n;

	ksim_trace(TRACE_RA, "# --- ra debug dump\n");

	bit_vector_init(&state.spill_slots);
	state.regs = 0xffff;
	state.reg_to_avx = malloc(count * sizeof(state.reg_to_avx[0]));
	memset(state.reg_to_avx, 0xff, count * sizeof(state.reg_to_avx[0]));
	state.range = prog->live_ranges;
	state.next_reg = 0;
	state.next_spill_reg = 0;

	list_for_each_entry(insn, &prog->insns, link) {
		ksim_trace(TRACE_RA, "%s\n", kir_insn_format(insn, buf, sizeof(buf)));

		state.exclude_regs = 0;
		state.locked_regs = 0;
		switch (insn->opcode) {
		case kir_comment:
			break;
		case kir_load_region:
			allocate_reg(&state, insn);
			break;
		case kir_store_region_mask:
			insn->xfer.src = use_reg(&state, insn, insn->xfer.src);
			insn->xfer.mask = use_reg(&state, insn, insn->xfer.mask);
			break;
		case kir_store_region:
			insn->xfer.src = use_reg(&state, insn, insn->xfer.src);
			break;
		case kir_immd:
		case kir_immw:
		case kir_immv:
		case kir_immvf:
			allocate_reg(&state, insn);
			break;

		case kir_send:
		case kir_const_send:
			spill_all(&state, insn);
			break;
			
		case kir_call:
		case kir_const_call:
			spill_all(&state, insn);
			if (insn->call.args > 0)
				insn->call.src0 = use_reg(&state, insn, insn->call.src0);
			if (insn->call.args > 1)
				insn->call.src1 = use_reg(&state, insn, insn->call.src1);

			/* FIXME: Only if has return value. */
			if (insn->call.args > 0)
				assign_reg(&state, insn, insn->call.src0.n);
			else
				allocate_reg(&state, insn);
			break;

		case kir_mov:
			ksim_unreachable();
			break;
		case kir_zxwd:
		case kir_sxwd:
		case kir_ps2d:
		case kir_d2ps:
		case kir_absd:
		case kir_rcp:
		case kir_sqrt:
		case kir_rsqrt:
		case kir_rndu:
		case kir_rndd:
		case kir_rnde:
		case kir_rndz:
		case kir_shri:
		case kir_shli:
			insn->alu.src0 = use_reg(&state, insn, insn->alu.src0);
			allocate_reg(&state, insn);
			break;
		case kir_and:
		case kir_andn:
		case kir_or:
		case kir_xor:
		case kir_shr:
		case kir_shl:
		case kir_asr:
		case kir_maxd:
		case kir_maxw:
		case kir_maxf:
		case kir_mind:
		case kir_minw:
		case kir_minf:
		case kir_divf:
		case kir_addd:
		case kir_addw:
		case kir_addf:
		case kir_subd:
		case kir_subw:
		case kir_subf:
		case kir_muld:
		case kir_mulw:
		case kir_mulf:
		case kir_cmp:
			lock_reg(&state, insn->alu.src0);
			lock_reg(&state, insn->alu.src1);
			insn->alu.src0 = use_reg(&state, insn, insn->alu.src0);
			insn->alu.src1 = use_reg(&state, insn, insn->alu.src1);
			allocate_reg(&state, insn);
			break;
		case kir_int_div_q_and_r:
		case kir_int_div_q:
		case kir_int_div_r:
		case kir_int_invm:
		case kir_int_rsqrtm:
			stub("ra insns");
			allocate_reg(&state, insn);
			break;
		case kir_maddf:
		case kir_nmaddf: {
			struct kir_reg *reuse_src;

			lock_reg(&state, insn->alu.src0);
			lock_reg(&state, insn->alu.src1);
			lock_reg(&state, insn->alu.src2);

			if (reg_dead(&state, insn, insn->alu.src0)) {
				reuse_src = &insn->alu.src0;
			} else if (reg_dead(&state, insn, insn->alu.src1)) {
				reuse_src = &insn->alu.src1;
			} else if (reg_dead(&state, insn, insn->alu.src2)) {
				reuse_src = &insn->alu.src2;
			} else {
				reuse_src = NULL;
			}

			/* FIXME: There's another option here, if all
			 * regs are live but one of them are spilled
			 * and we have free regs: load the spilled
			 * register into dst and leave the src marked
			 * as spilled. */

			insn->alu.src0 = use_reg(&state, insn, insn->alu.src0);
			insn->alu.src1 = use_reg(&state, insn, insn->alu.src1);
			insn->alu.src2 = use_reg(&state, insn, insn->alu.src2);

			if (reuse_src) {
				ksim_trace(TRACE_RA, "\treuse ymm%d for r%d\n",
					   reuse_src->n, insn->dst.n);
				assign_reg(&state, insn, reuse_src->n);
			} else if (state.regs & state.exclude_regs) {
				struct kir_insn *mov =
					kir_insn_create(kir_mov, kir_reg(0), insn->link.prev);
				allocate_reg(&state, insn);

				mov->dst = insn->dst;
				mov->alu.src0 = insn->alu.src0;
				insn->alu.src0 = mov->dst;

				ksim_trace(TRACE_RA, "\tmove ymm%d to ymm%d to not clobber r%d\n",
					   mov->alu.src0.n, mov->dst.n, state.avx_to_reg[insn->alu.src0.n]);
			} else {
				ksim_trace(TRACE_RA, "\tspill ymm%d for r%d and reuse for r%d\n",
					   insn->alu.src0.n,
					   state.avx_to_reg[insn->alu.src0.n].n,
					   insn->dst.n);
				spill_reg(&state, insn, insn->alu.src0.n);
				assign_reg(&state, insn, insn->alu.src0.n);
			}
			break;
		case kir_blend:
			lock_reg(&state, insn->alu.src0);
			lock_reg(&state, insn->alu.src1);
			lock_reg(&state, insn->alu.src2);
			insn->alu.src0 = use_reg(&state, insn, insn->alu.src0);
			insn->alu.src1 = use_reg(&state, insn, insn->alu.src1);
			insn->alu.src2 = use_reg(&state, insn, insn->alu.src2);
			allocate_reg(&state, insn);
			break;
		}

		case kir_gather:
			lock_reg(&state, insn->gather.mask);
			lock_reg(&state, insn->gather.offset);

			/* dst must be different than mask and offset
			 * for vpgatherdd. set exclude_regs to avoid
			 * allocting any of the currently used regs. */
			state.exclude_regs = ~state.regs;
			insn->gather.mask = use_reg(&state, insn, insn->gather.mask);
			insn->gather.offset = use_reg(&state, insn, insn->gather.offset);
			allocate_reg(&state, insn);
			break;

		case kir_set_load_base_indirect:
		case kir_set_load_base_imm:
			break;
		case kir_set_load_base_imm_offset:
			lock_reg(&state, insn->set_load_base.src);
			insn->set_load_base.src = use_reg(&state, insn, insn->set_load_base.src);
			break;
		case kir_load:
			allocate_reg(&state, insn);
			break;
		case kir_mask_store:
			lock_reg(&state, insn->store.src);
			lock_reg(&state, insn->store.mask);
			insn->store.src = use_reg(&state, insn, insn->store.src);
			insn->store.mask = use_reg(&state, insn, insn->store.mask);
			break;

		case kir_eot:
			break;
		case kir_eot_if_dead:
			lock_reg(&state, insn->eot.src);
			insn->eot.src = use_reg(&state, insn, insn->eot.src);
			break;
		}
	}

	free(state.reg_to_avx);

	ksim_trace(TRACE_RA, "\n");
}

static void
emit_region_load(struct builder *bld, const struct eu_region *region, int reg)
{
	if (region->hstride == 1 && region->width == region->vstride) {
		switch (region->type_size * region->exec_size) {
		case 32:
			builder_emit_m256i_load(bld, reg, region->offset);
			break;
		case 16:
		default:
			/* Could do broadcastq/d/w for sizes 8, 4 and
			 * 2 to avoid loading too much */
			builder_emit_m128i_load(bld, reg, region->offset);
			break;
		}
	} else if (region->hstride == 0 && region->vstride == 0 && region->width == 1) {
		switch (region->type_size) {
		case 4:
			builder_emit_vpbroadcastd(bld, reg, region->offset);
			break;
		default:
			stub("unhandled broadcast load size %d\n", region->type_size);
			break;
		}
	} else if (region->hstride == 0 && region->width == 4 && region->vstride == 1 &&
		   region->type_size == 2) {
		int tmp0_reg = 14;
		int tmp1_reg = 15;

		/* Handle the frag coord region */
		builder_emit_vpbroadcastw(bld, tmp0_reg, region->offset);
		builder_emit_vpbroadcastw(bld, tmp1_reg, region->offset + 4);
		builder_emit_vinserti128(bld, tmp0_reg, tmp1_reg, tmp0_reg, 1);

		builder_emit_vpbroadcastw(bld, reg, region->offset + 2);
		builder_emit_vpbroadcastw(bld, tmp1_reg, region->offset + 6);
		builder_emit_vinserti128(bld, reg, tmp1_reg, reg, 1);

		builder_emit_vpblendd(bld, reg, 0xcc, reg, tmp0_reg);
	} else if (region->hstride == 1 && region->width * region->type_size) {
		for (int i = 0; i < region->exec_size / region->width; i++) {
			int offset = region->offset + i * region->vstride * region->type_size;
			builder_emit_vpinsrq_rdi_relative(bld, reg, reg, offset, i & 1);
		}
	} else if (region->type_size == 4) {
		int offset, i = 0, tmp_reg = reg;

		for (int y = 0; y < region->exec_size / region->width; y++) {
			for (int x = 0; x < region->width; x++) {
				if (i == 4)
					tmp_reg = 14;
				offset = region->offset + (y * region->vstride + x * region->hstride) * region->type_size;
				builder_emit_vpinsrd_rdi_relative(bld, tmp_reg, tmp_reg, offset, i & 3);
				i++;
			}
		}
		if (tmp_reg != reg)
			builder_emit_vinserti128(bld, reg, tmp_reg, reg, 1);
	} else {
		stub("src: g%d.%d<%d,%d,%d>",
		     region->offset / 32, region->offset & 31,
		     region->vstride, region->width, region->hstride);
	}
}

static void
emit_region_store_mask(struct builder *bld,
			       const struct eu_region *region,
			       int dst, int mask)
{
	/* Don't have a good way to do mask stores for
	 * type_size < 4 and need builder_emit functions for
	 * exec_size < 8. */
	ksim_assert(region->exec_size == 8 && region->type_size == 4);

	switch (region->exec_size * region->type_size) {
	case 32:
		builder_emit_vpmaskmovd(bld, dst, mask, region->offset);
		break;
	default:
		stub("eu: type size %d in dest store", region->type_size);
		break;
	}
}

static void
emit_region_store(struct builder *bld,
			  const struct eu_region *region, int dst)
{
	switch (region->exec_size * region->type_size) {
	case 32:
		builder_emit_m256i_store(bld, dst, region->offset);
		break;
	case 16:
		builder_emit_m128i_store(bld, dst, region->offset);
		break;
	case 4:
		builder_emit_u32_store(bld, dst, region->offset);
		break;
	default:
		stub("eu: type size %d in dest store", region->type_size);
		break;
	}
}

void
kir_program_emit(struct kir_program *prog, struct builder *bld)
{
	struct kir_insn *insn;

	list_for_each_entry(insn, &prog->insns, link) {
		switch (insn->opcode) {
		case kir_comment:
			break;
		case kir_load_region:
			emit_region_load(bld, &insn->xfer.region, insn->dst.n);
			break;
		case kir_store_region_mask:
			emit_region_store_mask(bld, &insn->xfer.region,
					       insn->xfer.src.n,
					       insn->xfer.mask.n);
			break;
		case kir_store_region:
			emit_region_store(bld, &insn->xfer.region,
					  insn->xfer.src.n);
			break;
		case kir_set_load_base_indirect:
			builder_emit_load_rax_from_offset(bld, insn->set_load_base.offset);
			break;
		case kir_set_load_base_imm: {
			const void **p = get_const_data(sizeof(*p), sizeof(*p));
			*p = insn->set_load_base.pointer;
			builder_emit_load_rax_rip_relative(bld, builder_offset(bld, p));
			break;
		}
		case kir_set_load_base_imm_offset: {
			const void **p = get_const_data(sizeof(*p), sizeof(*p));
			*p = insn->set_load_base.pointer;

			builder_emit_vpextrd(bld, insn->set_load_base.src.n, 2);
			builder_emit_add_rax_rip_relative(bld, builder_offset(bld, p));
			break;
		}
		case kir_load:
			builder_emit_vmovdqa_from_rax(bld, insn->dst.n, insn->load.offset);
			break;
		case kir_mask_store:
			builder_emit_vpmaskmovd_to_rax(bld, insn->store.src.n,
						       insn->store.mask.n, insn->store.offset);
			break;

		case kir_immd:
		case kir_immw: {
			uint32_t *p = get_const_data(sizeof(*p), sizeof(*p));
			*p = insn->imm.d;
			builder_emit_vpbroadcastd_rip_relative(bld, insn->dst.n, builder_offset(bld, p));
			break;
		}

		case kir_immv: {
			void *p = get_const_data(8 * 2, 16);
			memcpy(p, insn->imm.v, 8 * 2);
			builder_emit_vbroadcasti128_rip_relative(bld, insn->dst.n,
								 builder_offset(bld, p));
			break;
		}

		case kir_immvf: {
			void *p = get_const_data(4 * 4, 4);
			memcpy(p, insn->imm.vf, 4 * 4);
			builder_emit_vbroadcasti128_rip_relative(bld, insn->dst.n,
								 builder_offset(bld, p));
			break;
		}

		case kir_send:
		case kir_const_send:
			if (insn->send.func == NULL) {
				stub("send func is NULL");
				break;
			}
			builder_emit_load_rsi_rip_relative(bld, builder_offset(bld, insn->send.args));
			if (insn->link.next == &prog->insns) {
				int32_t offset = (uint8_t *) insn->send.func - bld->p;
				builder_emit_jmp_relative(bld, offset);
			} else {
				builder_emit_push_rdi(bld);
				int32_t offset = (uint8_t *) insn->send.func - bld->p;
				builder_emit_call_relative(bld, offset);
				builder_emit_pop_rdi(bld);
			}
			break;

		case kir_call:
		case kir_const_call:
			ksim_assert(insn->dst.n == 0);
			if (insn->call.args > 0)
				ksim_assert(insn->call.src0.n == 0);
			if (insn->call.args > 1)
				ksim_assert(insn->call.src1.n == 1);

			builder_emit_push_rdi(bld);
			builder_emit_call_relative(bld, (uint8_t *) insn->call.func - bld->p);
			builder_emit_pop_rdi(bld);
			break;
		case kir_mov:
			builder_emit_vmovdqa(bld, insn->dst.n, insn->alu.src0.n);
			break;
		case kir_zxwd:
			builder_emit_vpmovzxwd(bld, insn->dst.n, insn->alu.src0.n);
			break;
		case kir_sxwd:
			builder_emit_vpmovsxwd(bld, insn->dst.n, insn->alu.src0.n);
			break;
		case kir_ps2d:
			builder_emit_vcvtps2dq(bld, insn->dst.n, insn->alu.src0.n);
			break;
		case kir_d2ps:
			builder_emit_vcvtdq2ps(bld, insn->dst.n, insn->alu.src0.n);
			break;
		case kir_absd:
			builder_emit_vpabsd(bld, insn->dst.n, insn->alu.src0.n);
			break;
		case kir_rcp:
			builder_emit_vrcpps(bld, insn->dst.n, insn->alu.src0.n);
			break;
		case kir_sqrt:
			builder_emit_vsqrtps(bld, insn->dst.n, insn->alu.src0.n);
			break;
		case kir_rsqrt:
			builder_emit_vrsqrtps(bld, insn->dst.n, insn->alu.src0.n);
			break;
		case kir_rndu:
			builder_emit_vroundps(bld, insn->dst.n, _MM_FROUND_TO_POS_INF, insn->alu.src0.n);
			break;
		case kir_rndd:
			builder_emit_vroundps(bld, insn->dst.n, _MM_FROUND_TO_NEG_INF, insn->alu.src0.n);
			break;
		case kir_rnde:
			builder_emit_vroundps(bld, insn->dst.n, _MM_FROUND_TO_NEAREST_INT, insn->alu.src0.n);
			break;
		case kir_rndz:
			builder_emit_vroundps(bld, insn->dst.n, _MM_FROUND_TO_ZERO, insn->alu.src0.n);
			break;

		/* alu binop */
		case kir_and:
			builder_emit_vpand(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_andn:
			builder_emit_vpandn(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_or:
			builder_emit_vpor(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_xor:
			builder_emit_vpxor(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_shri:
			builder_emit_vpsrld(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_shr:
			builder_emit_vpsrlvd(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_shli:
			builder_emit_vpslld(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_shl:
			builder_emit_vpsllvd(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_asr:
			builder_emit_vpsravd(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_maxd:
			ksim_unreachable("maxd");
			break;
		case kir_maxw:
			ksim_unreachable("maxw");
			break;
		case kir_maxf:
			builder_emit_vmaxps(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_mind:
			ksim_unreachable("mind");
			break;
		case kir_minw:
			ksim_unreachable("minw");
			break;
		case kir_minf:
			builder_emit_vminps(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_divf:
		case kir_int_div_q_and_r:
		case kir_int_div_q:
		case kir_int_div_r:
		case kir_int_invm:
		case kir_int_rsqrtm:
			stub("opcode emit");
			break;

		case kir_addd:
			builder_emit_vpaddd(bld, insn->dst.n,
					    insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_addw:
			builder_emit_vpaddw(bld, insn->dst.n,
					    insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_addf:
			builder_emit_vaddps(bld, insn->dst.n,
					    insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_subd:
			builder_emit_vpsubd(bld, insn->dst.n,
					    insn->alu.src1.n, insn->alu.src0.n);
			break;
		case kir_subw:
			stub("kir_subw");
			break;
		case kir_subf:
			builder_emit_vsubps(bld, insn->dst.n,
					    insn->alu.src1.n, insn->alu.src0.n);
			break;
		case kir_muld:
			builder_emit_vpmulld(bld, insn->dst.n,
					     insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_mulw:
			builder_emit_vpmullw(bld, insn->dst.n,
					     insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_mulf:
			builder_emit_vmulps(bld, insn->dst.n,
					    insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_maddf:
			if (insn->dst.n == insn->alu.src0.n)
				builder_emit_vfmadd132ps(bld, insn->dst.n,
							 insn->alu.src1.n, insn->alu.src2.n);
			else if (insn->dst.n == insn->alu.src1.n)
				builder_emit_vfmadd132ps(bld, insn->dst.n,
							 insn->alu.src0.n, insn->alu.src2.n);
			else if (insn->dst.n == insn->alu.src2.n)
				builder_emit_vfmadd231ps(bld, insn->dst.n,
							 insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_nmaddf:
			if (insn->dst.n == insn->alu.src0.n)
				builder_emit_vfnmadd132ps(bld, insn->dst.n,
							  insn->alu.src1.n, insn->alu.src2.n);
			else if (insn->dst.n == insn->alu.src1.n)
				builder_emit_vfnmadd132ps(bld, insn->dst.n,
							  insn->alu.src0.n, insn->alu.src2.n);
			else if (insn->dst.n == insn->alu.src2.n)
				builder_emit_vfnmadd231ps(bld, insn->dst.n,
							  insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_cmp:
			builder_emit_vcmpps(bld, insn->alu.imm2, insn->dst.n,
					    insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_blend:
			/* FIXME: should use vpblendvb */
			builder_emit_vpblendvps(bld, insn->dst.n, insn->alu.src2.n,
						insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_gather: {
			builder_emit_vpgatherdd(bld, insn->dst.n,
						insn->gather.offset.n,
						insn->gather.mask.n,
						insn->gather.scale,
						insn->gather.base_offset);
			break;
		}			
		case kir_eot:
			builder_emit_ret(bld);
			break;

		case kir_eot_if_dead: {
			builder_emit_vmovmskps(bld, insn->eot.src.n);
			void *branch = builder_emit_jne(bld);
			builder_emit_ret(bld);
			builder_align(bld);
			builder_set_branch_target(bld, branch, bld->p);
			break;
		}
		}

		if (trace_mask & TRACE_AVX) {
			char buf[128];
			int i = 0;
			while (builder_disasm(bld)) {
				if (i == 0)
					fprintf(trace_file, "%-42s  %s\n",
						kir_insn_format(insn, buf, sizeof(buf)),
						bld->disasm_output);
				else
					fprintf(trace_file, "%-42s  %s\n", "", bld->disasm_output);
				i++;
			}
			if (i == 0)
				fprintf(trace_file, "%s\n",
					kir_insn_format(insn, buf, sizeof(buf)));

		}
	}
}

void
kir_program_init(struct kir_program *prog, uint64_t surfaces, uint64_t samplers)
{
	list_init(&prog->insns);
	prog->next_reg = kir_reg(0);
	prog->scope = 0;
	prog->urb_offset = 0;
	prog->urb_length = 0;
	prog->binding_table_address = surfaces;
	prog->sampler_state_address = samplers;
}

shader_t
kir_program_finish(struct kir_program *prog)
{
	struct builder bld;

	if (trace_mask & TRACE_EU) {
		fprintf(trace_file, "# --- initial codegen\n");
		kir_program_print(prog, trace_file);
		fprintf(trace_file, "\n");
	}

	kir_program_copy_propagation(prog);

	if (trace_mask & TRACE_EU) {
		fprintf(trace_file, "# --- after copy propagation\n");
		kir_program_print(prog, trace_file);
		fprintf(trace_file, "\n");
	}

	kir_program_compute_live_ranges(prog);

	kir_program_dead_code_elimination(prog);

	if (trace_mask & TRACE_EU) {
		fprintf(trace_file, "# --- after dce\n");
		kir_program_print(prog, trace_file);
		fprintf(trace_file, "\n");
	}

	kir_program_allocate_registers(prog);

	if (trace_mask & TRACE_EU) {
		fprintf(trace_file, "# --- after ra\n");
		kir_program_print(prog, trace_file);
		fprintf(trace_file, "\n");
	}

	builder_init(&bld);

	ksim_trace(TRACE_AVX | TRACE_EU, "# --- code emit\n");
	kir_program_emit(prog, &bld);

	while (!list_empty(&prog->insns)) {
		struct kir_insn *insn = 
			container_of(prog->insns.next, insn, link);
		list_remove(&insn->link);
		kir_insn_destroy(insn);
	}

	free(prog->live_ranges);

	return builder_finish(&bld);
}
