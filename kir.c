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
kir_program_add_insn(struct kir_program *prog, uint32_t opcode)
{
	struct kir_insn *insn;

	insn = malloc(sizeof(*insn));
	insn->opcode = opcode;
	insn->dst = kir_reg(prog->next_reg.n++);

	list_insert(prog->insns.prev, &insn->link);
	prog->dst = insn->dst;

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
		   const void *base, struct kir_reg offset, struct kir_reg mask,
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

static void
kir_insn_print(struct kir_insn *insn, FILE *fp)
{
	char buf[32];

	switch (insn->opcode) {
	case kir_comment:
		fprintf(fp, "# %s\n", insn->comment);
		break;
	case kir_load_region:
		fprintf(fp, "r%-3d = load_region %s\n",
			insn->dst.n,
			format_region(buf, sizeof(buf),
				      &insn->xfer.region));
		break;
	case kir_store_region_mask:
		fprintf(fp, "r%-3d = store_region_mask r%d, r%d, %s\n",
			insn->dst.n, insn->xfer.mask.n, insn->xfer.src.n,
			format_region(buf, sizeof(buf), &insn->xfer.region));
		break;
	case kir_store_region:
		fprintf(fp, "       store_region, r%d, %s\n",
			insn->xfer.src.n,
			format_region(buf, sizeof(buf), &insn->xfer.region));
		break;
	case kir_immd:
		fprintf(fp, "r%-3d = imm %dd %ff\n", insn->dst.n, insn->imm.d,
			u32_to_float(insn->imm.d));
		break;
	case kir_immw:
		fprintf(fp, "r%-3d = imm %dw\n", insn->dst.n, insn->imm.d);
		break;
	case kir_immv:
		fprintf(fp, "r%-3d = imm [ %d, %d, %d, %d, %d, %d, %d, %d ]\n",
			insn->dst.n, 
			insn->imm.v[0], insn->imm.v[1],
			insn->imm.v[2], insn->imm.v[3],
			insn->imm.v[4], insn->imm.v[5],
			insn->imm.v[6], insn->imm.v[7]);
		break;
	case kir_immvf:
		fprintf(fp, "r%-3d = imm [ %f, %f, %f, %f ]\n",
			insn->dst.n, 
			insn->imm.vf[0], insn->imm.vf[1],
			insn->imm.vf[2], insn->imm.vf[3]);
		break;
	case kir_send:
	case kir_const_send:
		fprintf(fp, "r%-3d = %ssend src g%d-g%d",
			insn->dst.n,
			insn->opcode == kir_const_send ? "const_" : "",
			insn->send.src, insn->send.src + insn->send.mlen - 1);
		if (insn->send.rlen > 0)
			fprintf(fp, ", dst g%d-g%d",
				insn->send.dst, insn->send.dst + insn->send.rlen - 1);
		fprintf(fp, "\n");
		break;
	case kir_call:
	case kir_const_call:
		fprintf(fp, "r%-3d = %scall %p",
			insn->dst.n,
			insn->opcode == kir_const_call ? "const_" : "",
			insn->call.func);
		if (insn->call.args > 0)
			fprintf(fp, ", r%d", insn->call.src0.n);
		if (insn->call.args > 1)
			fprintf(fp, ", r%d", insn->call.src1.n);
		fprintf(fp, "\n");
		break;
	case kir_zxwd:
		fprintf(fp, "r%-3d = zxwd r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_sxwd:
		fprintf(fp, "r%-3d = sxwd r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_ps2d:
		fprintf(fp, "r%-3d = ps2d r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_d2ps:
		fprintf(fp, "r%-3d = d2ps r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_absd:
		fprintf(fp, "r%-3d = absd r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rcp:
		fprintf(fp, "r%-3d = rcp r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_sqrt:
		fprintf(fp, "r%-3d = sqrt r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rsqrt:
		fprintf(fp, "r%-3d = rsqrt r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rndu:
		fprintf(fp, "r%-3d = rndu r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rndd:
		fprintf(fp, "r%-3d = rndd r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rnde:
		fprintf(fp, "r%-3d = rnde r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_rndz:
		fprintf(fp, "r%-3d = rndz r%d\n", insn->dst.n, insn->alu.src0.n);
		break;
	case kir_and:
		fprintf(fp, "r%-3d = and r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_andn:
		fprintf(fp, "r%-3d = andn r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_or:
		fprintf(fp, "r%-3d = or r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_xor:
		fprintf(fp, "r%-3d = xor r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_shri:
		fprintf(fp, "r%-3d = shri r%d, %d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_shr:
		fprintf(fp, "r%-3d = shr r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_shli:
		fprintf(fp, "r%-3d = shli r%d, %d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.imm1);
		break;
	case kir_shl:
		fprintf(fp, "r%-3d = shl r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_asr:
		fprintf(fp, "r%-3d = asr r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_maxd:
		fprintf(fp, "r%-3d = maxd r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_maxw:
		fprintf(fp, "r%-3d = maxw r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_maxf:
		fprintf(fp, "r%-3d = maxf r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_mind:
		fprintf(fp, "r%-3d = mind r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_minw:
		fprintf(fp, "r%-3d = minw r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_minf:
		fprintf(fp, "r%-3d = minf r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_divf:
		fprintf(fp, "r%-3d = divf r%d, r%d\n", insn->dst.n,
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
		fprintf(fp, "r%-3d = addd r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_addw:
		fprintf(fp, "r%-3d = addw r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_addf:
		fprintf(fp, "r%-3d = addf r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;

	case kir_subd:
		fprintf(fp, "r%-3d = subd r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_subw:
		fprintf(fp, "r%-3d = subw r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_subf:
		fprintf(fp, "r%-3d = subf r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_muld:
		fprintf(fp, "r%-3d = muld r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_mulw:
		fprintf(fp, "r%-3d = mulw r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_mulf:
		fprintf(fp, "r%-3d = mulf r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n);
		break;
	case kir_avg:
		fprintf(fp, "r%-3d = avg\n", insn->dst.n);
		break;
	case kir_maddf:
		fprintf(fp, "r%-3d = maddf r%d, r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n, insn->alu.src2.n);
		break;
	case kir_nmaddf:
		fprintf(fp, "r%-3d = nmaddf r%d, r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n, insn->alu.src2.n);
		break;
	case kir_cmp:
		fprintf(fp, "r%-3d = cmp r%d, r%d, op %d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n, insn->alu.imm2);
		break;
	case kir_blend:
		fprintf(fp, "r%-3d = blend r%d, r%d, r%d\n", insn->dst.n,
			insn->alu.src0.n, insn->alu.src1.n, insn->alu.src2.n);
		break;
	case kir_gather:
		fprintf(fp, "r%-3d = gather r%d, %d(%p,r%d,%d)\n",
			insn->dst.n,
			insn->gather.mask.n,
			insn->gather.base_offset, insn->gather.base,
			insn->gather.offset.n, insn->gather.scale);
		break;
	case kir_eot:
		fprintf(fp, "       eot\n");
		break;
	}
}

void
kir_program_print(struct kir_program *prog, FILE *fp)
{
	struct kir_insn *insn;

	list_for_each_entry(insn, &prog->insns, link)
		kir_insn_print(insn, fp);
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

uint32_t *
kir_program_compute_live_ranges(struct kir_program *prog)
{
	struct kir_insn *insn;
	uint32_t *range;
	bool *live_regs;
	uint32_t region_map[512];

	live_regs = malloc(prog->next_reg.n * sizeof(live_regs[0]));
	memset(live_regs, 0, prog->next_reg.n * sizeof(live_regs[0]));
	range = malloc(prog->next_reg.n * sizeof(range[0]));
	memset(range, 0, prog->next_reg.n * sizeof(range[0]));
	memset(region_map, 0, 128 * sizeof(region_map[0]));
	/* Initialize regions past the eu registers to live. */
	memset(region_map + 128, ~0, 384 * sizeof(region_map[0]));
	
	insn = container_of(prog->insns.prev, insn, link);
	while (&insn->link != &prog->insns) {
		bool live = false;
		switch (insn->opcode) {
		case kir_comment:
			range[insn->dst.n] = insn->dst.n + 1;
			break;
		case kir_load_region:
			live = live_regs[insn->dst.n];
			set_region_live(&insn->xfer.region, live, region_map);
			break;
		case kir_store_region_mask:
		case kir_store_region:
			live = region_is_live(&insn->xfer.region, region_map);
			set_live(insn->xfer.src, live, insn, range, live_regs);
			if (live)
				range[insn->dst.n] = insn->dst.n + 1;
			set_region_live(&insn->xfer.region, false, region_map);
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
				set_region_live(&insn->xfer.region, false, region_map);
			}
			if (live)
				range[insn->dst.n] = insn->dst.n + 1;

			for (uint32_t i = 0; i < insn->send.mlen; i++) {
				struct eu_region region = region_for_reg(insn->send.src + i);
				set_region_live(&region, live, region_map);
			}
			break;

		case kir_call:
			range[insn->dst.n] = insn->dst.n + 1;
			if (insn->call.args > 0)
				set_live(insn->call.src0, live, insn, range, live_regs);
			if (insn->call.args > 1)
				set_live(insn->call.src1, live, insn, range, live_regs);
			break;
		case kir_const_call:
			live = live_regs[insn->dst.n];
			if (insn->call.args > 0)
				set_live(insn->call.src0, live, insn, range, live_regs);
			if (insn->call.args > 1)
				set_live(insn->call.src1, live, insn, range, live_regs);
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
		case kir_avg:
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
			break;
		case kir_eot:
			range[insn->dst.n] = insn->dst.n + 1;
			break;
		}

		insn = container_of(insn->link.prev, insn, link);
	}

	free(live_regs);

	return range;
}

struct resident_region {
	struct eu_region region;
	struct kir_reg reg;
	struct list region_link;
	struct list reg_link;
};

static bool
regions_equal(const struct eu_region *a, const struct eu_region *b)
{
	return memcmp(a, b, sizeof(*a)) == 0;
}

static bool
regions_overlap(const struct eu_region *a, const struct eu_region *b)
{
	uint32_t a_size = (a->exec_size / a->width) * a->vstride * a->type_size;
	uint32_t b_size = (b->exec_size / b->width) * b->vstride * b->type_size;

	/* This is a coarse approximation, but probably sufficient: if
	 * the "bounding boxs" of the regions overlap, we consider the
	 * regions overlapping. This misses cases where a region could
	 * be contained in a gap (ie, where width < vstride) of
	 * another region or two regions could be interleaved. */

	return a->offset + a_size > b->offset &&
		 b->offset + b_size > a->offset;
}

void
kir_program_copy_propagation(struct kir_program *prog)
{
	struct kir_insn *insn;
	struct list *reg_rr;
	struct resident_region *rr, *next;
	int count = prog->next_reg.n;
	struct kir_reg *remap;

	reg_rr = malloc(count * sizeof(reg_rr[0]));
	remap = malloc(count * sizeof(remap[0]));

	for (uint32_t i = 0; i < count; i++)
		remap[i] = kir_reg(i);

	const uint32_t max_eu_regs = 400;
	struct list region_to_reg[max_eu_regs];
	for (uint32_t i = 0; i < max_eu_regs; i++)
		list_init(&region_to_reg[i]);

	list_for_each_entry(insn, &prog->insns, link) {
		list_init(&reg_rr[insn->dst.n]);
		switch (insn->opcode) {
		case kir_comment:
			break;
		case kir_load_region: {
			list_for_each_entry(rr, &region_to_reg[insn->xfer.region.offset / 32], region_link) {
				if (regions_equal(&insn->xfer.region, &rr->region)) {
				    remap[insn->dst.n] = rr->reg;
				    goto load_region_done;
				}
			}

			struct list *head = &reg_rr[insn->dst.n];
			list_for_each_entry_safe(rr, next, head, reg_link) {
				list_remove(&rr->region_link);
				list_remove(&rr->reg_link);
				free(rr);
			}

			/* FIXME: Insert an rr for each region_to_reg it overlaps */
			rr = malloc(sizeof(*rr));
			rr->region = insn->xfer.region;
			rr->reg = insn->dst;
			list_insert(&reg_rr[insn->dst.n], &rr->reg_link);
			list_insert(&region_to_reg[insn->xfer.region.offset / 32], &rr->region_link);
		load_region_done:
			break;
		}
		case kir_store_region_mask:
		case kir_store_region: {

			insn->xfer.src = remap[insn->xfer.src.n];

			/* Invalidate registers overlapping region */
			struct list *head = &region_to_reg[insn->xfer.region.offset / 32];
			list_for_each_entry_safe(rr, next, head, region_link) {
			if (regions_overlap(&rr->region, &insn->xfer.region)) {
					list_remove(&rr->region_link);
					list_remove(&rr->reg_link);
					free(rr);
				}
			}

			rr = malloc(sizeof(*rr));
			rr->region = insn->xfer.region;
			rr->reg = insn->xfer.src;
			list_insert(&reg_rr[insn->dst.n], &rr->reg_link);
			list_insert(&region_to_reg[insn->xfer.region.offset / 32], &rr->region_link);
			break;
		}
		case kir_immd:
		case kir_immw:
		case kir_immv:
		case kir_immvf:
			break;
		case kir_send:
		case kir_const_send:
			/* Don't need regs. */
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
		case kir_avg:
			break;
		case kir_maddf:
		case kir_nmaddf:
		case kir_blend:
			insn->alu.src0 = remap[insn->alu.src0.n];
			insn->alu.src1 = remap[insn->alu.src1.n];
			insn->alu.src2 = remap[insn->alu.src2.n];
			break;
		case kir_gather:
			/* Don't copy propagate mask: gather overwites
			 * the mask and we need a fresh copy each time. */
			insn->gather.offset = remap[insn->gather.offset.n];
			break;
		case kir_eot:
			break;


		}
	}
}

void
kir_program_dce(struct kir_program *prog)
{
	struct kir_insn *insn;
	uint32_t *range;

	range = kir_program_compute_live_ranges(prog);

	insn = container_of(prog->insns.next, insn, link);
	while (&insn->link != &prog->insns) {
		struct kir_insn *next = container_of(insn->link.next, insn, link);
		if (insn->dst.n >= range[insn->dst.n]) {
			list_remove(&insn->link);
			free(insn);
		}
		insn = next;
	}

	free(range);
}

struct ra_state {
	uint32_t *range;
	uint32_t regs;
	uint8_t *reg_to_avx;
	struct kir_reg avx_to_reg[16];
	uint32_t spill_slots;
};

/* Insert spill instruction of register reg before instruction insn */
static void
spill_reg(struct ra_state *state, struct kir_insn *insn, struct kir_reg reg)
{
	ksim_assert(state->spill_slots);
	int slot = __builtin_ffs(state->spill_slots) - 1;
	state->spill_slots &= ~(1 << slot);

	ksim_trace(TRACE_RA, "spill ymm%d to slot %d\n", reg.n, slot);

	/* FIXME: Need constructor for this */
	struct kir_insn *spill;

	spill = malloc(sizeof(*spill));
	spill->opcode = kir_store_region;
	spill->dst = kir_reg(0);

	list_insert(insn->link.prev, &spill->link);

	spill->xfer.src = reg;
	spill->xfer.region = (struct eu_region) {
		.offset = offsetof(struct thread, spill[slot]),
		.type_size = 4,
		.exec_size = 8,
		.vstride = 8,
		.width = 8,
		.hstride = 1
	};

	struct kir_reg def = state->avx_to_reg[reg.n];

	state->regs |= (1 << reg.n);
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
	/* FIXME: Need to spill something else if no regs. But not the
	 * other src reg in a binop... */
	ksim_assert(state->regs);

	int avx_reg = __builtin_ffs(state->regs) - 1;
	uint32_t slot = state->reg_to_avx[reg.n] - 16;
	state->spill_slots |= (1 << slot);

	ksim_trace(TRACE_RA, "unspill slot %d to ymm%d\n", slot, avx_reg);

	struct kir_insn *unspill;
	unspill = malloc(sizeof(*unspill));
	unspill->opcode = kir_load_region;
	unspill->dst = reg;

	list_insert(insn->link.prev, &unspill->link);

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

static void
unspill1(struct ra_state *state, struct kir_insn *insn, struct kir_reg a)
{
	if (state->reg_to_avx[a.n] >= 16)
		unspill_reg(state, insn, a);
}

static void
unspill2(struct ra_state *state, struct kir_insn *insn, struct kir_reg a, struct kir_reg b)
{
	if (state->reg_to_avx[a.n] >= 16)
		unspill_reg(state, insn, a);
	if (state->reg_to_avx[b.n] >= 16)
		unspill_reg(state, insn, b);
}

static void
unspill3(struct ra_state *state, struct kir_insn *insn,
	 struct kir_reg a, struct kir_reg b, struct kir_reg c)
{
	if (state->reg_to_avx[a.n] >= 16)
		unspill_reg(state, insn, a);
	if (state->reg_to_avx[b.n] >= 16)
		unspill_reg(state, insn, b);
	if (state->reg_to_avx[c.n] >= 16)
		unspill_reg(state, insn, c);
}

static inline struct kir_reg
use_reg(struct ra_state *state, struct kir_insn *insn, struct kir_reg reg)
{
	struct kir_reg avx_reg = {
		.n = state->reg_to_avx[reg.n]
	};

	ksim_assert(avx_reg.n != 0xff);
	if (insn->dst.n >= state->range[reg.n]) {
		ksim_trace(TRACE_RA, "use ymm%d for r%d, dead now\n",
			   avx_reg.n, reg.n);
		state->regs |= (1 << avx_reg.n);
	} else {
		ksim_trace(TRACE_RA, "use ymm%d for r%d\n",
			   avx_reg.n, reg.n);
	}

	return avx_reg;
}

static void
spill_all(struct ra_state *state, struct kir_insn *insn)
{
	uint32_t live_regs = 0xffff & ~state->regs;
	uint32_t avx_reg;

	for_each_bit(avx_reg, live_regs)
		spill_reg(state, insn, kir_reg(avx_reg));
}

static void
kir_program_allocate_registers(struct kir_program *prog)
{
	struct kir_insn *insn;
	struct ra_state state;
	uint32_t exclude_regs;

	state.spill_slots = 0xffffffff;
	state.regs = 0xffff;
	state.reg_to_avx = malloc(prog->next_reg.n * sizeof(state.reg_to_avx[0]));
	memset(state.reg_to_avx, 0xff, prog->next_reg.n * sizeof(state.reg_to_avx[0]));
	state.range = kir_program_compute_live_ranges(prog);

	insn = container_of(prog->insns.next, insn, link);
	while (&insn->link != &prog->insns) {
		if (trace_mask & TRACE_RA)
			kir_insn_print(insn, trace_file);
		exclude_regs = 0;
		switch (insn->opcode) {
		case kir_comment:
			break;
		case kir_load_region:
			break;
		case kir_store_region_mask:
		case kir_store_region:
			unspill1(&state, insn, insn->xfer.src);
			insn->xfer.src = use_reg(&state, insn, insn->xfer.src);
			break;
		case kir_immd:
		case kir_immw:
		case kir_immv:
		case kir_immvf:
			break;

		case kir_send:
		case kir_const_send:
			/* Don't need regs. */
			break;
			
		case kir_call:
		case kir_const_call:
			spill_all(&state, insn);
			if (insn->call.args == 1) {
				unspill1(&state, insn, insn->call.src0);
				insn->call.src0 = use_reg(&state, insn, insn->call.src0);
			} else if (insn->call.args == 2) {
				unspill2(&state, insn, insn->call.src0, insn->call.src1);
				insn->call.src0 = use_reg(&state, insn, insn->call.src0);
				insn->call.src1 = use_reg(&state, insn, insn->call.src1);
			}
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
			unspill1(&state, insn, insn->alu.src0);
			insn->alu.src0 = use_reg(&state, insn, insn->alu.src0);
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
			unspill2(&state, insn, insn->alu.src0, insn->alu.src1);
			insn->alu.src0 = use_reg(&state, insn, insn->alu.src0);
			insn->alu.src1 = use_reg(&state, insn, insn->alu.src1);
			break;
		case kir_int_div_q_and_r:
		case kir_int_div_q:
		case kir_int_div_r:
		case kir_int_invm:
		case kir_int_rsqrtm:
		case kir_avg:
			break;
		case kir_maddf:
		case kir_nmaddf:
		case kir_blend:
			unspill3(&state, insn, insn->alu.src0, insn->alu.src1, insn->alu.src2);
			insn->alu.src0 = use_reg(&state, insn, insn->alu.src0);
			insn->alu.src1 = use_reg(&state, insn, insn->alu.src1);
			insn->alu.src2 = use_reg(&state, insn, insn->alu.src2);

			ksim_trace(TRACE_RA, "reuse ymm%d for r%d\n",
				   insn->alu.src0.n, insn->dst.n);

			/* src0 becomes dst */
			assign_reg(&state, insn, insn->alu.src0.n);
			break;
		case kir_gather:
			/* dst must be different that mask and offset for vpgatherdd. */
			unspill2(&state, insn, insn->gather.mask, insn->gather.offset);
			exclude_regs = ~state.regs;
			insn->gather.mask = use_reg(&state, insn, insn->gather.mask);
			insn->gather.offset = use_reg(&state, insn, insn->gather.offset);
			break;
		case kir_eot:
			break;
		}

		switch (insn->opcode) {
		case kir_comment:
		case kir_store_region_mask:
		case kir_store_region:
		case kir_send:
		case kir_const_send:
		case kir_eot:
			break;
		case kir_maddf:
		case kir_nmaddf:
			/* These two reuse a src as dst. */
			break;
		default: {
			uint32_t regs = state.regs & ~exclude_regs;
			ksim_assert(regs); /* Don't run out regs :D */
			int avx_reg = __builtin_ffs(regs) - 1;
			ksim_trace(TRACE_RA, "allocate ymm%d for r%d\n",
				   avx_reg, insn->dst.n);
			assign_reg(&state, insn, avx_reg);
			break;
		}
		}

		insn = container_of(insn->link.next, insn, link);
	}

	free(state.reg_to_avx);
	free(state.range);
}

void
kir_program_emit(struct kir_program *prog, struct builder *bld)
{
	struct kir_insn *insn;

	list_for_each_entry(insn, &prog->insns, link) {
		if (trace_mask & TRACE_AVX)
			kir_insn_print(insn, trace_file);
		switch (insn->opcode) {
		case kir_comment:
			break;
		case kir_load_region:
			builder_emit_region_load(bld, &insn->xfer.region, insn->dst.n);
			break;
		case kir_store_region_mask:
			builder_emit_region_store_mask(bld, &insn->xfer.region,
						       insn->xfer.src.n,
						       insn->xfer.mask.n);
			break;
		case kir_store_region:
			builder_emit_region_store(bld, &insn->xfer.region,
						  insn->xfer.src.n);
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

		/* send */
		case kir_send:
		case kir_const_send:
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
			builder_emit_vpsrlvd(bld, insn->dst.n, insn->alu.src1.n, insn->alu.src0.n);
			break;
		case kir_shli:
			builder_emit_vpslld(bld, insn->dst.n, insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_shl:
			builder_emit_vpsllvd(bld, insn->dst.n, insn->alu.src1.n, insn->alu.src0.n);
			break;
		case kir_asr:
			builder_emit_vpsravd(bld, insn->dst.n, insn->alu.src1.n, insn->alu.src0.n);
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
					    insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_subw:
			stub("kir_subw");
			break;
		case kir_subf:
			builder_emit_vsubps(bld, insn->dst.n,
					    insn->alu.src0.n, insn->alu.src1.n);
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
		case kir_avg:
			stub("kir_avg");
			break;
		case kir_maddf:
			builder_emit_vfmadd132ps(bld, insn->dst.n,
						 insn->alu.src1.n, insn->alu.src2.n);
			break;
		case kir_nmaddf:
			builder_emit_vfnmadd132ps(bld, insn->dst.n,
						  insn->alu.src1.n, insn->alu.src2.n);
			break;
		case kir_cmp:
			builder_emit_vcmpps(bld, insn->alu.imm2, insn->dst.n,
					    insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_blend:
			/* FIXME: should be vpblendvb */
			builder_emit_vpblendvps(bld, insn->dst.n, insn->alu.src2.n,
						insn->alu.src0.n, insn->alu.src1.n);
			break;
		case kir_gather: {
			const void **p = get_const_data(sizeof(*p), sizeof(*p));
			*p = insn->gather.base;
			builder_emit_load_rax_rip_relative(bld, builder_offset(bld, p));
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
		}

		if (trace_mask & TRACE_AVX) {
			while (builder_disasm(bld))
				fprintf(trace_file, "\t\t\t\t\t%s\n", bld->disasm_output);
	}


	}
}

void
kir_program_init(struct kir_program *prog, uint64_t surfaces, uint64_t samplers)
{
	list_init(&prog->insns);
	prog->next_reg = kir_reg(0);
	prog->scope = 0;
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
		fprintf(trace_file, "# --- after copy propatation\n");
		kir_program_print(prog, trace_file);
		fprintf(trace_file, "\n");
	}

	kir_program_dce(prog);

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

	/* builder_emit_program()? */
	ksim_trace(TRACE_AVX | TRACE_EU, "# --- code emit\n");
	kir_program_emit(prog, &bld);

	while (!list_empty(&prog->insns)) {
		struct kir_insn *insn = 
			container_of(prog->insns.next, insn, link);
		list_remove(&insn->link);
		free(insn);
	}

	return builder_finish(&bld);
}
