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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>
#include <assert.h>
#include <immintrin.h>

#include "eu.h"
#include "avx-builder.h"

static const struct {
   int num_srcs;
   bool store_dst;
} opcode_info[] = {
   [BRW_OPCODE_MOV]             = { .num_srcs = 1, .store_dst = false },
   [BRW_OPCODE_SEL]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_NOT]             = { .num_srcs = 1, .store_dst = true },
   [BRW_OPCODE_AND]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_OR]              = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_XOR]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_SHR]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_SHL]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_ASR]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_CMP]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_CMPN]            = { },
   [BRW_OPCODE_CSEL]            = { .num_srcs = 3, .store_dst = true },
   [BRW_OPCODE_F32TO16]         = { },
   [BRW_OPCODE_F16TO32]         = { },
   [BRW_OPCODE_BFREV]           = { },
   [BRW_OPCODE_BFE]             = { .num_srcs = 3, .store_dst = true },
   [BRW_OPCODE_BFI1]            = { },
   [BRW_OPCODE_BFI2]            = { .num_srcs = 3, .store_dst = true },
   [BRW_OPCODE_JMPI]            = { .num_srcs = 0, .store_dst = false },
   [BRW_OPCODE_IF]              = { },
   [BRW_OPCODE_IFF]             = { },
   [BRW_OPCODE_ELSE]            = { },
   [BRW_OPCODE_ENDIF]           = { },
   [BRW_OPCODE_DO]              = { .num_srcs = 0, .store_dst = false },
   [BRW_OPCODE_WHILE]           = { },
   [BRW_OPCODE_BREAK]           = { },
   [BRW_OPCODE_CONTINUE]        = { },
   [BRW_OPCODE_HALT]            = { },
   [BRW_OPCODE_MSAVE]           = { },
   [BRW_OPCODE_MRESTORE]        = { },
   [BRW_OPCODE_GOTO]            = { },
   [BRW_OPCODE_POP]             = { },
   [BRW_OPCODE_WAIT]            = { },
   [BRW_OPCODE_SEND]            = { },
   [BRW_OPCODE_SENDC]           = { },
   [BRW_OPCODE_MATH]            = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_ADD]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_MUL]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_AVG]             = { },
   [BRW_OPCODE_FRC]             = { .num_srcs = 1,. store_dst = true },
   [BRW_OPCODE_RNDU]            = { .num_srcs = 1,. store_dst = true },
   [BRW_OPCODE_RNDD]            = { .num_srcs = 1,. store_dst = true },
   [BRW_OPCODE_RNDE]            = { .num_srcs = 1,. store_dst = true },
   [BRW_OPCODE_RNDZ]            = { .num_srcs = 1,. store_dst = true },
   [BRW_OPCODE_MAC]             = { },
   [BRW_OPCODE_MACH]            = { },
   [BRW_OPCODE_LZD]             = { },
   [BRW_OPCODE_FBH]             = { },
   [BRW_OPCODE_FBL]             = { },
   [BRW_OPCODE_CBIT]            = { },
   [BRW_OPCODE_ADDC]            = { },
   [BRW_OPCODE_SUBB]            = { },
   [BRW_OPCODE_SAD2]            = { },
   [BRW_OPCODE_SADA2]           = { },
   [BRW_OPCODE_DP4]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_DPH]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_DP3]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_DP2]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_LINE]            = { .num_srcs = 0, .store_dst = true },
   [BRW_OPCODE_PLN]             = { .num_srcs = 0, .store_dst = true },
   [BRW_OPCODE_MAD]             = { .num_srcs = 3, .store_dst = false },
   [BRW_OPCODE_LRP]             = { .num_srcs = 3, .store_dst = true },
   [BRW_OPCODE_NENOP]           = { .num_srcs = 0, .store_dst = false },
   [BRW_OPCODE_NOP]             = { .num_srcs = 0, .store_dst = false },
};

static void
fill_region_for_src(struct eu_region *region, struct inst_src *src,
		    uint32_t subnum_bytes, struct builder *bld)
{
	int row_offset = bld->exec_offset / src->width;

	region->offset = src->num * 32 + subnum_bytes +
		row_offset * src->vstride * type_size(src->type);
	region->type_size = type_size(src->type);
	region->exec_size = bld->exec_size;
	region->hstride = src->hstride;

	if (src->width == src->vstride && src->hstride == 1) {
		region->vstride = bld->exec_size;
		region->width = bld->exec_size;
	} else {
		region->vstride = src->vstride;
		region->width = src->width;
	}
}

static void
fill_region_for_dst(struct eu_region *region, struct inst_dst *dst,
		    uint32_t subnum, struct builder *bld)
{
	region->offset =
		offsetof(struct thread, grf[dst->num]) + subnum +
		bld->exec_offset * dst->hstride * type_size(dst->type);

	region->type_size = type_size(dst->type);
	region->exec_size = bld->exec_size;
	region->vstride = bld->exec_size;
	region->width = bld->exec_size;
	region->hstride = 1;
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

static int
builder_emit_src_modifiers(struct builder *bld,
			   struct inst *inst, struct inst_src *src, int reg)
{
	/* FIXME: Build the load above into the source modifier when possible, eg:
	 *
	 *     vpabsd 0x456(%rdi), %ymm1
	 */

	if (src->abs) {
		if (src->type == BRW_HW_REG_TYPE_F) {
			int tmp_reg = builder_get_reg_with_uniform(bld, 0x7fffffff);
			builder_emit_vpand(bld, tmp_reg, reg, tmp_reg);
			bld->regs[tmp_reg].contents = 0;
			reg = tmp_reg;
		} else {
			int tmp_reg = builder_get_reg(bld);
			builder_emit_vpabsd(bld, tmp_reg, reg);
			reg = tmp_reg;
		}
	}

	if (src->negate) {
		int tmp_reg = builder_get_reg_with_uniform(bld, 0);

		if (is_logic_instruction(inst)) {
			builder_emit_vpxor(bld, tmp_reg, reg, tmp_reg);
		} else if (src->type == BRW_HW_REG_TYPE_F) {
			builder_emit_vsubps(bld, tmp_reg, reg, tmp_reg);
		} else {
			builder_emit_vpsubd(bld, tmp_reg, reg, tmp_reg);
		}
		bld->regs[tmp_reg].contents = 0;
		reg = tmp_reg;
	}

	return reg;
}

int
builder_emit_region_load(struct builder *bld, const struct eu_region *region)
{
	struct avx2_reg *areg;

	if (list_find(areg, &bld->regs_lru_list, link,
		      (areg->contents & BUILDER_REG_CONTENTS_EU_REG) &&
		      memcmp(region, &areg->region, sizeof(*region)) == 0)) {
		int reg = builder_use_reg(bld, areg);
		if (trace_mask & TRACE_AVX) {
			ksim_trace(TRACE_AVX, "*** found match for reg g%d.%d<%d,%d,%d>%d: %%ymm%d\n",
				   region->offset / 32, region->offset & 31,
				   region->vstride, region->width, region->hstride,
				   region->type_size, reg);
		}

		return reg;
	}

	int reg = builder_get_reg(bld);
	bld->regs[reg].contents |= BUILDER_REG_CONTENTS_EU_REG;
	memcpy(&bld->regs[reg].region, region, sizeof(*region));

	if (region->hstride == 1 && region->width == region->vstride) {
		switch (region->type_size * bld->exec_size) {
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
		int tmp0_reg = builder_get_reg(bld);
		int tmp1_reg = builder_get_reg(bld);

		/* Handle the frag coord region */
		builder_emit_vpbroadcastw(bld, tmp0_reg, region->offset);
		builder_emit_vpbroadcastw(bld, tmp1_reg, region->offset + 4);
		builder_emit_vinserti128(bld, tmp0_reg, tmp1_reg, tmp0_reg, 1);

		builder_emit_vpbroadcastw(bld, reg, region->offset + 2);
		builder_emit_vpbroadcastw(bld, tmp1_reg, region->offset + 6);
		builder_emit_vinserti128(bld, reg, tmp1_reg, reg, 1);

		builder_emit_vpblendd(bld, reg, 0xcc, reg, tmp0_reg);
	} else if (region->hstride == 1 && region->width * region->type_size) {
		for (int i = 0; i < bld->exec_size / region->width; i++) {
			int offset = region->offset + i * region->vstride * region->type_size;
			builder_emit_vpinsrq_rdi_relative(bld, reg, reg, offset, i & 1);
		}
	} else if (region->type_size == 4) {
		int offset, i = 0, tmp_reg = reg;

		for (int y = 0; y < bld->exec_size / region->width; y++) {
			for (int x = 0; x < region->width; x++) {
				if (i == 4)
					tmp_reg = builder_get_reg(bld);
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

	return reg;
}

static int
builder_emit_type_conversion(struct builder *bld, int reg, int dst_type, int src_type)
{
	switch (dst_type) {
	case BRW_HW_REG_TYPE_UD:
	case BRW_HW_REG_TYPE_D: {
		if (src_type == BRW_HW_REG_TYPE_UD || src_type == BRW_HW_REG_TYPE_D)
			return reg;

		int new_reg = builder_get_reg(bld);
		if (src_type == BRW_HW_REG_TYPE_UW)
			builder_emit_vpmovzxwd(bld, new_reg, reg);
		else if (src_type == BRW_HW_REG_TYPE_W)
			builder_emit_vpmovsxwd(bld, new_reg, reg);
		else if (src_type == BRW_HW_REG_TYPE_F)
			builder_emit_vcvtps2dq(bld, new_reg, reg);
		else
			stub("src type %d for ud/d dst type\n", src_type);
		return new_reg;
	}
	case BRW_HW_REG_TYPE_UW:
	case BRW_HW_REG_TYPE_W: {
		if (src_type == BRW_HW_REG_TYPE_UW || src_type == BRW_HW_REG_TYPE_W)
			return reg;

		int tmp_reg = builder_get_reg(bld);
		if (src_type == BRW_HW_REG_TYPE_UD) {
			stub("not sure this pack is right");
			builder_emit_vextractf128(bld, tmp_reg, reg, 1);
			builder_emit_vpackusdw(bld, tmp_reg, tmp_reg, reg);
		} else if (src_type == BRW_HW_REG_TYPE_D) {
			stub("not sure this pack is right");
			builder_emit_vextractf128(bld, tmp_reg, reg, 1);
			builder_emit_vpackssdw(bld, tmp_reg, tmp_reg, reg);
		} else
			stub("src type %d for uw/w dst type\n", src_type);
		return tmp_reg;
	}
	case BRW_HW_REG_TYPE_F: {
		if (src_type == BRW_HW_REG_TYPE_F)
			return reg;

		int new_reg = builder_get_reg(bld);
		if (src_type == BRW_HW_REG_TYPE_UW) {
			builder_emit_vpmovzxwd(bld, new_reg, reg);
			builder_emit_vcvtdq2ps(bld, new_reg, new_reg);
		} else if (src_type == BRW_HW_REG_TYPE_W) {
			builder_emit_vpmovsxwd(bld, new_reg, reg);
			builder_emit_vcvtdq2ps(bld, new_reg, new_reg);
		} else if (src_type == BRW_HW_REG_TYPE_UD) {
			/* FIXME: Need to convert to int64 and then
			 * convert to floats as there is no uint32 to
			 * float cvt. */
			builder_emit_vcvtdq2ps(bld, new_reg, reg);
		} else if (src_type == BRW_HW_REG_TYPE_D) {
			builder_emit_vcvtdq2ps(bld, new_reg, reg);
		} else {
			stub("src type %d for float dst\n", src_type);
		}
		return new_reg;
	}
	case GEN8_HW_REG_TYPE_UQ:
	case GEN8_HW_REG_TYPE_Q:
	default:
		stub("dst type\n", dst_type);
		return reg;
	}
}

static int
builder_emit_src_load(struct builder *bld,
		      struct inst *inst, struct inst_src *src)
{
	struct inst_common common = unpack_inst_common(inst);
	struct inst_dst dst;
	uint32_t *p;
	int reg, src_type;

	src_type = src->type;
	if (src->file == BRW_ARCHITECTURE_REGISTER_FILE) {
		switch (src->num & 0xf0) {
		case BRW_ARF_NULL:
			reg = 0;
			break;
		default:
			stub("architecture register file load");
			reg = 0;
			break;
		}
	} else if (src->file == BRW_IMMEDIATE_VALUE) {
		switch (src->type) {
		case BRW_HW_REG_TYPE_UD:
		case BRW_HW_REG_TYPE_D:
		case BRW_HW_REG_TYPE_F: {
			uint32_t ud = unpack_inst_imm(inst).ud;
			reg = builder_get_reg_with_uniform(bld, ud);
			break;
		}
		case BRW_HW_REG_TYPE_UW:
		case BRW_HW_REG_TYPE_W:
			reg = builder_get_reg(bld);
			stub("unhandled imm type in src load");
			break;

		case BRW_HW_REG_IMM_TYPE_UV:
			/* Gen6+ packed unsigned immediate vector */
			reg = builder_get_reg(bld);
			p = builder_get_const_data(bld, 8 * 2, 16);
			memcpy(p, unpack_inst_imm(inst).uv, 8 * 2);
			builder_emit_vbroadcasti128_rip_relative(bld, reg, builder_offset(bld, p));
			src_type = BRW_HW_REG_TYPE_UW;
			break;

		case BRW_HW_REG_IMM_TYPE_VF:
			/* packed float immediate vector */
			reg = builder_get_reg(bld);
			p = builder_get_const_data(bld, 4 * 4, 4);
			memcpy(p, unpack_inst_imm(inst).vf, 4 * 4);
			builder_emit_vbroadcasti128_rip_relative(bld, reg, builder_offset(bld, p));
			src_type = BRW_HW_REG_TYPE_F;
			break;

		case BRW_HW_REG_IMM_TYPE_V:
			/* packed int imm. vector; uword dest only */
			reg = builder_get_reg(bld);
			p = builder_get_const_data(bld, 8 * 2, 16);
			memcpy(p, unpack_inst_imm(inst).v, 8 * 2);
			builder_emit_vbroadcasti128_rip_relative(bld, reg, builder_offset(bld, p));
			src_type = BRW_HW_REG_TYPE_W;
			break;

		case GEN8_HW_REG_TYPE_UQ:
		case GEN8_HW_REG_TYPE_Q:
		case GEN8_HW_REG_IMM_TYPE_DF:
		case GEN8_HW_REG_IMM_TYPE_HF:
			stub("unhandled imm type in src load");
			break;
		default:
			ksim_unreachable("invalid imm type");
		}

	} else if (src->file == BRW_GENERAL_REGISTER_FILE) {
		struct eu_region region;

		if (common.access_mode == BRW_ALIGN_1)
			fill_region_for_src(&region, src, src->da1_subnum, bld);
		else
			fill_region_for_src(&region, src, src->da16_subnum, bld);

		reg = builder_emit_region_load(bld, &region);
		reg = builder_emit_src_modifiers(bld, inst, src, reg);
	} else {
		stub("unhandled src");
		reg = 0;
	}

	if (opcode_info[common.opcode].num_srcs == 3)
		dst = unpack_inst_3src_dst(inst);
	else
		dst = unpack_inst_2src_dst(inst);

	reg = builder_emit_type_conversion(bld, reg, dst.type, src_type);

	return reg;
}

static void
builder_emit_cmp(struct builder *bld, int modifier, int dst, int src0, int src1)
{
	static uint32_t eu_to_avx_cmp[] = {
		[BRW_CONDITIONAL_NONE]	= 0,
		[BRW_CONDITIONAL_Z]	= _CMP_EQ_OQ,
		[BRW_CONDITIONAL_NZ]	= _CMP_NEQ_OQ,
		[BRW_CONDITIONAL_G]	= _CMP_GT_OQ,
		[BRW_CONDITIONAL_GE]	= _CMP_GE_OQ,
		[BRW_CONDITIONAL_L]	= _CMP_LT_OQ,
		[BRW_CONDITIONAL_LE]	= _CMP_LE_OQ,
		[BRW_CONDITIONAL_R]	= 0,
		[BRW_CONDITIONAL_O]	= 0,
		[BRW_CONDITIONAL_U]	= 0,
	};

	builder_emit_vcmpps(bld, eu_to_avx_cmp[modifier], dst, src0, src1);
}

static void
builder_invalidate_region(struct builder *bld, const struct eu_region *r)
{
	for (int i = 0; i < ARRAY_LENGTH(bld->regs); i++) {
		if ((bld->regs[i].contents & BUILDER_REG_CONTENTS_EU_REG) &&
		    regions_overlap(r, &bld->regs[i].region)) {
			bld->regs[i].contents &= ~BUILDER_REG_CONTENTS_EU_REG;
			ksim_trace(TRACE_AVX,
				   "*** invalidate g%d.%d (ymm%d)\n",
				   bld->regs[i].region.offset / 32,
				   bld->regs[i].region.offset & 31, i);
		}
	}
}

void
builder_emit_region_store(struct builder *bld,
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

	builder_invalidate_region(bld, region);

	/* FIXME: For a straight move, this makes the AVX2 register
	 * refer to the dst region.  That's fine, but the register may
	 * still also shadow the src region, but since we only track
	 * one region per AVX2 reg, that is lost. */
	bld->regs[dst].region = *region;
	bld->regs[dst].contents |= BUILDER_REG_CONTENTS_EU_REG;
}

static void
builder_emit_dst_store(struct builder *bld, int avx_reg,
		       struct inst *inst, struct inst_dst *dst)
{
	struct inst_common common = unpack_inst_common(inst);
	int subnum;

	/* FIXME: write masks */

	if (dst->hstride > 1)
		stub("eu: dst hstride %d is > 1", dst->hstride);

	if (common.saturate) {
		int zero = builder_get_reg_with_uniform(bld, 0);
		int one = builder_get_reg_with_uniform(bld, float_to_u32(1.0f));
		ksim_assert(is_float(dst->file, dst->type));
		builder_emit_vmaxps(bld, avx_reg, avx_reg, zero);
		builder_emit_vminps(bld, avx_reg, avx_reg, one);
	}

	if (common.access_mode == BRW_ALIGN_1)
		subnum = dst->da1_subnum;
	else
		subnum = dst->da16_subnum;

	struct eu_region region;
	fill_region_for_dst(&region, dst, subnum, bld);
	builder_emit_region_store(bld, &region, avx_reg);
}

static inline int
reg_offset(int num, int subnum)
{
	return offsetof(struct thread, grf[num].ud[subnum]);
}

static void
builder_emit_sfid_thread_spawner(struct builder *bld, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);

	uint32_t opcode = field(send.function_control, 0, 0);
	uint32_t request = field(send.function_control, 1, 1);
	uint32_t resource_select = field(send.function_control, 4, 4);

	ksim_assert(send.eot);
	ksim_assert(opcode == 0 && request == 0 && resource_select == 1);
}

struct sfid_dataport1_args {
	uint32_t src;
	void *buffer;
	uint32_t mask;
};

static void
sfid_dataport1_untyped_write(struct thread *t, struct sfid_dataport1_args *args)
{
	uint32_t c;
	const uint32_t mask = t->mask & t->grf[args->src].ud[7];

	for_each_bit (c, mask) {
		uint32_t *dst = args->buffer + t->grf[args->src + 1].ud[c];
		uint32_t *src = &t->grf[args->src + 2].ud[c];
		for (int comp = 0; comp < 4; comp++) {
			if (args->mask & (1 << comp))
				continue;
			*dst++ = *src;
			src += 8;
		}
	}
}

static void
builder_emit_sfid_dataport1(struct builder *bld, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	struct surface buffer;

	uint32_t bti = field(send.function_control, 0, 7);
	uint32_t mask = field(send.function_control, 8, 11);
	uint32_t simd_mode = field(send.function_control, 12, 13);
	uint32_t opcode = field(send.function_control, 14, 18);
	//uint32_t header_present = field(send.function_control, 19, 19);

	struct sfid_dataport1_args *args;
	args = builder_get_const_data(bld, sizeof *args, 8);

	switch (opcode) {
	case 9:
		/* Command reference: MSD1W_US */
		ksim_assert(simd_mode == 2); /* SIMD8 */
		args->src = unpack_inst_2src_src0(inst).num;
		args->mask = mask;
		bool valid = get_surface(bld->binding_table_address,
					 bti, &buffer);
		ksim_assert(valid);
		args->buffer = buffer.pixels;

		builder_emit_load_rsi_rip_relative(bld, builder_offset(bld, args));
		builder_emit_call(bld, sfid_dataport1_untyped_write);
		break;

	default:
		stub("dataport1 opcode");
		break;
	}


}

/* Vectorized AVX2 math functions from glibc's libmvec */
__m256 _ZGVdN8vv___powf_finite(__m256 x, __m256 y);
__m256 _ZGVdN8v___logf_finite(__m256 x);
__m256 _ZGVdN8v___expf_finite(__m256 x);
__m256 _ZGVdN8v_sinf(__m256 x);
__m256 _ZGVdN8v_cosf(__m256 x);
__m256 _ZGVdN8vv___powf_finite(__m256 x, __m256 y);

bool
compile_inst(struct builder *bld, struct inst *inst)
{
	struct inst_src src0, src1, src2;
	struct inst_dst dst;
	int dst_reg, src0_reg, src1_reg, src2_reg;
	uint32_t opcode = unpack_inst_common(inst).opcode;
	bool eot = false;

	if (opcode == BRW_OPCODE_MATH) {
		/* If we need to call one of the libmvec math helpers,
		 * we need to load src0 into ymm0 to match the x86-64
		 * calling convention. Move it to the front of the LRU
		 * list so the src load will pick it. */
		builder_invalidate_all(bld);
	}

	if (opcode_info[opcode].num_srcs == 3) {
		src0 = unpack_inst_3src_src0(inst);
		src0_reg = builder_emit_src_load(bld, inst, &src0);
		src1 = unpack_inst_3src_src1(inst);
		src1_reg = builder_emit_src_load(bld, inst, &src1);
		src2 = unpack_inst_3src_src2(inst);
		src2_reg = builder_emit_src_load(bld, inst, &src2);
	} else if (opcode_info[opcode].num_srcs >= 1) {
		src0 = unpack_inst_2src_src0(inst);
		src0_reg = builder_emit_src_load(bld, inst, &src0);
	}

	if (opcode_info[opcode].num_srcs == 2) {
		src1 = unpack_inst_2src_src1(inst);
		src1_reg = builder_emit_src_load(bld, inst, &src1);
	}

	if (opcode_info[opcode].num_srcs == 3)
		dst = unpack_inst_3src_dst(inst);
	else
		dst = unpack_inst_2src_dst(inst);

	if (opcode_info[opcode].store_dst)
		dst_reg = builder_get_reg(bld);

	switch (opcode) {
	case BRW_OPCODE_MOV:
		builder_emit_dst_store(bld, src0_reg, inst, &dst);
		break;
	case BRW_OPCODE_SEL: {
		int modifier = unpack_inst_common(inst).cond_modifier;
		if (modifier == BRW_CONDITIONAL_GE) {
			builder_emit_vmaxps(bld, dst_reg, src0_reg, src1_reg);
			break;
		}
		if (modifier == BRW_CONDITIONAL_L) {
			builder_emit_vminps(bld, dst_reg, src0_reg, src1_reg);
			break;
		}
		int tmp_reg = builder_get_reg(bld);
		builder_emit_cmp(bld, modifier, tmp_reg, src0_reg, src1_reg);
		/* AVX2 blendv is opposite of the EU sel order, so we
		 * swap src0 and src1 operands. */
		builder_emit_vpblendvps(bld, dst_reg, tmp_reg, src1_reg, src0_reg);
		break;
	}
	case BRW_OPCODE_NOT: {
		int tmp_reg = builder_get_reg_with_uniform(bld, 0);
		builder_emit_vpxor(bld, dst_reg, src0_reg, tmp_reg);
		break;
	}
	case BRW_OPCODE_AND:
		builder_emit_vpand(bld, dst_reg, src0_reg, src1_reg);
		break;
	case BRW_OPCODE_OR:
		builder_emit_vpor(bld, dst_reg, src0_reg, src1_reg);
		break;
	case BRW_OPCODE_XOR:
		builder_emit_vpxor(bld, dst_reg, src0_reg, src1_reg);
		break;
	case BRW_OPCODE_SHR:
		builder_emit_vpsrlvd(bld, dst_reg, src1_reg, src0_reg);
		break;
	case BRW_OPCODE_SHL:
		builder_emit_vpsllvd(bld, dst_reg, src1_reg, src0_reg);
		break;
	case BRW_OPCODE_ASR:
		builder_emit_vpsravd(bld, dst_reg, src1_reg, src0_reg);
		break;
	case BRW_OPCODE_CMP: {
		int modifier = unpack_inst_common(inst).cond_modifier;
		int tmp_reg = builder_get_reg(bld);
		builder_emit_cmp(bld, modifier, tmp_reg, src0_reg, src1_reg);
		break;
	}
	case BRW_OPCODE_CMPN:
		stub("BRW_OPCODE_CMPN");
		break;
	case BRW_OPCODE_CSEL:
		stub("BRW_OPCODE_CSEL");
		break;
	case BRW_OPCODE_F32TO16:
		stub("BRW_OPCODE_F32TO16");
		break;
	case BRW_OPCODE_F16TO32:
		stub("BRW_OPCODE_F16TO32");
		break;
	case BRW_OPCODE_BFREV:
		stub("BRW_OPCODE_BFREV");
		break;
	case BRW_OPCODE_BFE:
		stub("BRW_OPCODE_BFE");
		break;
	case BRW_OPCODE_BFI1:
		stub("BRW_OPCODE_BFI1");
		break;
	case BRW_OPCODE_BFI2:
		stub("BRW_OPCODE_BFI2");
		break;
	case BRW_OPCODE_JMPI:
		stub("BRW_OPCODE_JMPI");
		break;
	case BRW_OPCODE_IF:
		stub("BRW_OPCODE_IF");
		break;
	case BRW_OPCODE_IFF:
		stub("BRW_OPCODE_IFF");
		break;
	case BRW_OPCODE_ELSE:
		stub("BRW_OPCODE_ELSE");
		break;
	case BRW_OPCODE_ENDIF:
		stub("BRW_OPCODE_ENDIF");
		break;
	case BRW_OPCODE_DO:
		stub("BRW_OPCODE_DO");
		break;
	case BRW_OPCODE_WHILE:
		stub("BRW_OPCODE_WHILE");
		break;
	case BRW_OPCODE_BREAK:
		stub("BRW_OPCODE_BREAK");
		break;
	case BRW_OPCODE_CONTINUE:
		stub("BRW_OPCODE_CONTINUE");
		break;
	case BRW_OPCODE_HALT:
		stub("BRW_OPCODE_HALT");
		break;
	case BRW_OPCODE_MSAVE:
		stub("BRW_OPCODE_MSAVE");
		break;
	case BRW_OPCODE_MRESTORE:
		stub("BRW_OPCODE_MRESTORE");
		break;
	case BRW_OPCODE_GOTO:
		stub("BRW_OPCODE_GOTO");
		break;
	case BRW_OPCODE_POP:
		stub("BRW_OPCODE_POP");
		break;
	case BRW_OPCODE_WAIT:
		stub("BRW_OPCODE_WAIT");
		break;
	case BRW_OPCODE_SEND:
	case BRW_OPCODE_SENDC: {
		struct inst_send send = unpack_inst_send(inst);
		eot = send.eot;

		switch (send.sfid) {
		case BRW_SFID_SAMPLER:
			builder_emit_sfid_sampler(bld, inst);
			break;
		case GEN6_SFID_DATAPORT_RENDER_CACHE:
			builder_emit_sfid_render_cache(bld, inst);
			break;
		case BRW_SFID_URB:
			builder_emit_sfid_urb(bld, inst);
			break;
		case BRW_SFID_THREAD_SPAWNER:
			builder_emit_sfid_thread_spawner(bld, inst);
			break;
		case HSW_SFID_DATAPORT_DATA_CACHE_1:
			builder_emit_sfid_dataport1(bld, inst);
			break;
		default:
			stub("sfid: %d", send.sfid);
			break;
		}
		break;
	}
	case BRW_OPCODE_MATH:
		switch (unpack_inst_common(inst).math_function) {
		case BRW_MATH_FUNCTION_INV:
			builder_emit_vrcpps(bld, dst_reg, src0_reg);
			break;
		case BRW_MATH_FUNCTION_LOG:
			dst_reg = builder_emit_call(bld, _ZGVdN8v___logf_finite);
			builder_invalidate_all(bld);
			break;
		case BRW_MATH_FUNCTION_EXP:
			dst_reg = builder_emit_call(bld, _ZGVdN8v___expf_finite);
			builder_invalidate_all(bld);
			break;
		case BRW_MATH_FUNCTION_SQRT:
			builder_emit_vsqrtps(bld, dst_reg, src0_reg);
			break;
		case BRW_MATH_FUNCTION_RSQ:
			builder_emit_vrsqrtps(bld, dst_reg, src0_reg);
			break;
		case BRW_MATH_FUNCTION_SIN:
			dst_reg = builder_emit_call(bld, _ZGVdN8v_sinf);
			builder_invalidate_all(bld);
			break;
		case BRW_MATH_FUNCTION_COS:
			dst_reg = builder_emit_call(bld, _ZGVdN8v_cosf);
			builder_invalidate_all(bld);
			break;
		case BRW_MATH_FUNCTION_SINCOS:
			ksim_unreachable("sincos only gen4/5");
			break;
		case BRW_MATH_FUNCTION_FDIV:
			builder_emit_vdivps(bld, dst_reg, src0_reg, src1_reg);
			break;
		case BRW_MATH_FUNCTION_POW: {
			dst_reg = builder_emit_call(bld, _ZGVdN8vv___powf_finite);
			builder_invalidate_all(bld);
			break;
		}
		case BRW_MATH_FUNCTION_INT_DIV_QUOTIENT_AND_REMAINDER:
			stub("BRW_MATH_FUNCTION_INT_DIV_QUOTIENT_AND_REMAINDER");
			break;
		case BRW_MATH_FUNCTION_INT_DIV_QUOTIENT:
			stub("BRW_MATH_FUNCTION_INT_DIV_QUOTIENT");
			break;
		case BRW_MATH_FUNCTION_INT_DIV_REMAINDER:
			stub("BRW_MATH_FUNCTION_INT_DIV_REMAINDER");
			break;
		case GEN8_MATH_FUNCTION_INVM:
			stub("GEN8_MATH_FUNCTION_INVM");
			break;
		case GEN8_MATH_FUNCTION_RSQRTM:
			stub("GEN8_MATH_FUNCTION_RSQRTM");
			break;
		default:
			ksim_assert(false);
			break;
		}
		break;
	case BRW_OPCODE_ADD:
		switch (dst.type) {
		case BRW_HW_REG_TYPE_UD:
		case BRW_HW_REG_TYPE_D:
			builder_emit_vpaddd(bld, dst_reg, src0_reg, src1_reg);
			break;
		case BRW_HW_REG_TYPE_UW:
		case BRW_HW_REG_TYPE_W:
			builder_emit_vpaddw(bld, dst_reg, src0_reg, src1_reg);
			break;
		case BRW_HW_REG_TYPE_F:
			builder_emit_vaddps(bld, dst_reg, src0_reg, src1_reg);
			break;
		default:
			stub("unhandled type for add");
			break;
		}
		break;
	case BRW_OPCODE_MUL:
		switch (dst.type) {
		case BRW_HW_REG_TYPE_UD:
		case BRW_HW_REG_TYPE_D:
			builder_emit_vpmulld(bld, dst_reg, src0_reg, src1_reg);
			break;
		case BRW_HW_REG_TYPE_UW:
		case BRW_HW_REG_TYPE_W:
			builder_emit_vpmullw(bld, dst_reg, src0_reg, src1_reg);
			break;
		case BRW_HW_REG_TYPE_F:
			builder_emit_vmulps(bld, dst_reg, src0_reg, src1_reg);
			break;
		default:
			stub("unhandled type for mul");
			break;
		}
		break;
	case BRW_OPCODE_AVG:
		stub("BRW_OPCODE_AVG");
		break;
	case BRW_OPCODE_FRC: {
		int tmp_reg = builder_get_reg(bld);
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		builder_emit_vroundps(bld, tmp_reg, _MM_FROUND_TO_NEG_INF, src0_reg);
		builder_emit_vsubps(bld, dst_reg, src0_reg, tmp_reg);
		break;
	}
	case BRW_OPCODE_RNDU:
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		builder_emit_vroundps(bld, dst_reg, _MM_FROUND_TO_POS_INF, src0_reg);
		break;
	case BRW_OPCODE_RNDD:
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		builder_emit_vroundps(bld, dst_reg, _MM_FROUND_TO_NEG_INF, src0_reg);
		break;
	case BRW_OPCODE_RNDE:
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		builder_emit_vroundps(bld, dst_reg, _MM_FROUND_TO_NEAREST_INT, src0_reg);
		break;
	case BRW_OPCODE_RNDZ:
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		builder_emit_vroundps(bld, dst_reg, _MM_FROUND_TO_ZERO, src0_reg);
		break;
	case BRW_OPCODE_MAC:
		stub("BRW_OPCODE_MAC");
		break;
	case BRW_OPCODE_MACH:
		stub("BRW_OPCODE_MACH");
		break;
	case BRW_OPCODE_LZD:
		stub("BRW_OPCODE_LZD");
		break;
	case BRW_OPCODE_FBH:
		stub("BRW_OPCODE_FBH");
		break;
	case BRW_OPCODE_FBL:
		stub("BRW_OPCODE_FBL");
		break;
	case BRW_OPCODE_CBIT:
		stub("BRW_OPCODE_CBIT");
		break;
	case BRW_OPCODE_ADDC:
		stub("BRW_OPCODE_ADDC");
		break;
	case BRW_OPCODE_SUBB:
		stub("BRW_OPCODE_SUBB");
		break;
	case BRW_OPCODE_SAD2:
		stub("BRW_OPCODE_SAD2");
		break;
	case BRW_OPCODE_SADA2:
		stub("BRW_OPCODE_SADA2");
		break;
	case BRW_OPCODE_DP4: {
		int tmp0_reg = builder_get_reg(bld);
		int tmp1_reg = builder_get_reg(bld);
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		builder_emit_vmulps(bld, tmp0_reg, src0_reg, src1_reg);
		builder_emit_vpermilps(bld, dst_reg, 0, tmp0_reg);
		builder_emit_vpermilps(bld, tmp1_reg, 0x55, tmp0_reg);
		builder_emit_vaddps(bld, dst_reg, dst_reg, tmp1_reg);
		builder_emit_vpermilps(bld, tmp1_reg, 0xaa, tmp0_reg);
		builder_emit_vaddps(bld, dst_reg, dst_reg, tmp1_reg);
		builder_emit_vpermilps(bld, tmp1_reg, 0xff, tmp0_reg);
		builder_emit_vaddps(bld, dst_reg, dst_reg, tmp1_reg);
		break;
	}
	case BRW_OPCODE_DPH: {
		int tmp0_reg = builder_get_reg(bld);
		int tmp1_reg = builder_get_reg(bld);
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		builder_emit_vmulps(bld, tmp0_reg, src0_reg, src1_reg);
		builder_emit_vpermilps(bld, dst_reg, 0, tmp0_reg);
		builder_emit_vpermilps(bld, tmp1_reg, 0x55, tmp0_reg);
		builder_emit_vaddps(bld, dst_reg, dst_reg, tmp1_reg);
		builder_emit_vpermilps(bld, tmp1_reg, 0xaa, tmp0_reg);
		builder_emit_vaddps(bld, dst_reg, dst_reg, tmp1_reg);
		builder_emit_vpermilps(bld, tmp1_reg, 0xff, src1_reg);
		builder_emit_vaddps(bld, dst_reg, dst_reg, tmp1_reg);
		break;
	}
	case BRW_OPCODE_DP3: {
		int tmp0_reg = builder_get_reg(bld);
		int tmp1_reg = builder_get_reg(bld);
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		builder_emit_vmulps(bld, tmp0_reg, src0_reg, src1_reg);
		builder_emit_vpermilps(bld, dst_reg, 0, tmp0_reg);
		builder_emit_vpermilps(bld, tmp1_reg, 0x55, tmp0_reg);
		builder_emit_vaddps(bld, dst_reg, dst_reg, tmp1_reg);
		builder_emit_vpermilps(bld, tmp1_reg, 0xaa, tmp0_reg);
		builder_emit_vaddps(bld, dst_reg, dst_reg, tmp1_reg);
		break;
	}
	case BRW_OPCODE_DP2: {
		int tmp_reg = builder_get_reg(bld);
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		builder_emit_vmulps(bld, tmp_reg, src0_reg, src1_reg);
		builder_emit_vpermilps(bld, dst_reg, 0, tmp_reg);
		builder_emit_vpermilps(bld, tmp_reg, 0x55, tmp_reg);
		builder_emit_vaddps(bld, dst_reg, dst_reg, tmp_reg);
		break;
	}
	case BRW_OPCODE_LINE: {
		src0 = unpack_inst_2src_src0(inst);
		src1 = unpack_inst_2src_src1(inst);
		ksim_assert(src0.type == BRW_HW_REG_TYPE_F);
		ksim_assert(src1.type == BRW_HW_REG_TYPE_F);
		int subnum = src0.da16_subnum / 4;
		int tmp1_reg = builder_get_reg(bld);

		src0_reg = builder_emit_src_load(bld, inst, &src1);
		builder_emit_vpbroadcastd(bld, dst_reg, reg_offset(src0.num, subnum));
		builder_emit_vpbroadcastd(bld, tmp1_reg, reg_offset(src0.num, subnum + 3));
		builder_emit_vfmadd132ps(bld, dst_reg, src0_reg, tmp1_reg);
		break;
	}
	case BRW_OPCODE_PLN: {
		src0 = unpack_inst_2src_src0(inst);
		src1 = unpack_inst_2src_src1(inst);
		ksim_assert(src0.type == BRW_HW_REG_TYPE_F);
		ksim_assert(src1.type == BRW_HW_REG_TYPE_F);
		int tmp0_reg = builder_get_reg(bld);

		src2 = unpack_inst_2src_src1(inst);
		src2.num++;

		int subnum = src0.da1_subnum / 4;
		src1_reg = builder_emit_src_load(bld, inst, &src1);
		builder_emit_vpbroadcastd(bld, tmp0_reg, reg_offset(src0.num, subnum));
		builder_emit_vpbroadcastd(bld, dst_reg, reg_offset(src0.num, subnum + 3));
		builder_emit_vfmadd132ps(bld, tmp0_reg, src1_reg, dst_reg);
		builder_emit_vpbroadcastd(bld, dst_reg, reg_offset(src0.num, subnum + 1));
		src0_reg = builder_emit_src_load(bld, inst, &src2);
		builder_emit_vfmadd132ps(bld, dst_reg, src0_reg, tmp0_reg);
		break;
	}
	case BRW_OPCODE_MAD:
		if (is_integer(dst.file, dst.type)) {
			dst_reg = builder_get_reg(bld);
			builder_emit_vpmulld(bld, src1_reg, src1_reg, src2_reg);
			builder_emit_vpaddd(bld, dst_reg, src0_reg, src1_reg);
		} else {
			builder_emit_vfmadd231ps(bld, src0_reg, src2_reg, src1_reg);
			dst_reg = builder_use_reg(bld, &bld->regs[src0_reg]);
		}
		builder_emit_dst_store(bld, dst_reg, inst, &dst);
		break;
	case BRW_OPCODE_LRP:
		ksim_assert(src0.type == BRW_HW_REG_TYPE_F);
		ksim_assert(src1.type == BRW_HW_REG_TYPE_F);
		ksim_assert(src2.type == BRW_HW_REG_TYPE_F);
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);

		/* dst = src0 * src1 + (1 - src0) * src2
		 *     = src0 * src1 + src2 - src0 * src2
		 */
		builder_emit_vmulps(bld, dst_reg, src0_reg, src2_reg);
		builder_emit_vsubps(bld, dst_reg, dst_reg, src2_reg);
		builder_emit_vfmadd231ps(bld, dst_reg, src0_reg, src1_reg);
		break;
	case BRW_OPCODE_NENOP:
	case BRW_OPCODE_NOP:
		break;
	}

	if (opcode_info[opcode].store_dst)
		builder_emit_dst_store(bld, dst_reg, inst, &dst);

        builder_release_regs(bld);

	return eot;
}

static bool
do_compile_inst(struct builder *bld, struct inst *inst)
{
	uint32_t opcode = unpack_inst_common(inst).opcode;
	int exec_size = 1 << unpack_inst_common(inst).exec_size;
	struct inst_dst dst;
	bool eot;

	if (opcode_info[opcode].num_srcs == 3)
		dst = unpack_inst_3src_dst(inst);
	else
		dst = unpack_inst_2src_dst(inst);

	if (exec_size * type_size(dst.type) < 64) {
		bld->exec_size = exec_size;
		bld->exec_offset = 0;
		eot = compile_inst(bld, inst);
	} else {
		bld->exec_size = exec_size / 2;
		bld->exec_offset = 0;
		eot = compile_inst(bld, inst);
		bld->exec_offset = exec_size / 2;
		compile_inst(bld, inst);
	}

	return eot;
}

#include "external/gen_device_info.h"

void brw_init_compaction_tables(const struct gen_device_info *devinfo);
int brw_disassemble_inst(FILE *file, const struct gen_device_info *devinfo,
                         struct inst *inst, bool is_compacted);

void brw_uncompact_instruction(const struct gen_device_info *devinfo,
                               struct inst *dst, void *src);

static const struct gen_device_info ksim_devinfo = { .gen = 9 };

void
builder_emit_shader(struct builder *bld, uint64_t kernel_offset)
{
	struct inst uncompacted;
	void *insn;
	bool eot;
	uint64_t ksp, range;
	void *p;

	brw_init_compaction_tables(&ksim_devinfo);

	ksp = kernel_offset + gt.instruction_base_address;
	p = map_gtt_offset(ksp, &range);

	do {
		if (unpack_inst_common(p).cmpt_control) {
			brw_uncompact_instruction(&ksim_devinfo, &uncompacted, p);
			insn = &uncompacted;
			p += 8;
		} else {
			insn = p;
			p += 16;
		}

		if (trace_mask & TRACE_EU)
			brw_disassemble_inst(trace_file, &ksim_devinfo, insn, false);

		eot = do_compile_inst(bld, insn);

		builder_trace(bld, trace_file);
	} while (!eot);

	if (trace_mask & (TRACE_EU | TRACE_AVX))
		fprintf(trace_file, "\n");
}

shader_t
compile_shader(uint64_t kernel_offset,
	       uint64_t surfaces, uint64_t samplers)
{
	struct builder bld;

	builder_init(&bld, surfaces, samplers);

	builder_emit_shader(&bld, kernel_offset);

	builder_emit_ret(&bld);

	return builder_finish(&bld);
}
