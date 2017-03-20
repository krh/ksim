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
#include "kir.h"

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
   [BRW_OPCODE_MAD]             = { .num_srcs = 3, .store_dst = true },
   [BRW_OPCODE_LRP]             = { .num_srcs = 3, .store_dst = true },
   [BRW_OPCODE_NENOP]           = { .num_srcs = 0, .store_dst = false },
   [BRW_OPCODE_NOP]             = { .num_srcs = 0, .store_dst = false },
};

static void
fill_region_for_src(struct eu_region *region, struct inst_src *src,
		    uint32_t subnum_bytes, struct kir_program *prog)
{
	int row_offset = prog->exec_offset / src->width;

	region->offset = src->num * 32 + subnum_bytes +
		row_offset * src->vstride * type_size(src->type);
	region->type_size = type_size(src->type);
	region->exec_size = prog->exec_size;
	region->hstride = src->hstride;

	if (src->width == src->vstride && src->hstride == 1) {
		region->vstride = prog->exec_size;
		region->width = prog->exec_size;
	} else {
		region->vstride = src->vstride;
		region->width = src->width;
	}
}

static void
fill_region_for_dst(struct eu_region *region, struct inst_dst *dst,
		    uint32_t subnum, struct kir_program *prog)
{
	region->offset =
		offsetof(struct thread, grf[dst->num]) + subnum +
		prog->exec_offset * dst->hstride * type_size(dst->type);

	region->type_size = type_size(dst->type);
	region->exec_size = prog->exec_size;
	region->vstride = prog->exec_size;
	region->width = prog->exec_size;
	region->hstride = 1;
}

static struct kir_reg
kir_program_emit_src_modifiers(struct kir_program *prog,
			       struct inst *inst, struct inst_src *src,
			       struct kir_reg reg)
{
	if (src->abs) {
		if (src->type == BRW_HW_REG_TYPE_F) {
			kir_program_immd(prog, 0x7fffffff);
			reg = kir_program_alu(prog, kir_and, reg, prog->dst);
		} else {
			reg = kir_program_alu(prog, kir_absd, reg, prog->dst);
		}
	}

	if (src->negate) {
		kir_program_immd(prog, 0);
		if (is_logic_instruction(inst)) {
			reg = kir_program_alu(prog, kir_xor, prog->dst, reg);
		} else if (src->type == BRW_HW_REG_TYPE_F) {
			reg = kir_program_alu(prog, kir_subf, prog->dst, reg);
		} else {
			reg = kir_program_alu(prog, kir_subd, prog->dst, reg);
		}
	}

	return reg;
}

static struct kir_reg
kir_program_emit_type_conversion(struct kir_program *prog,
				 struct kir_reg reg, int dst_type, int src_type)
{
	switch (dst_type) {
	case BRW_HW_REG_TYPE_UD:
	case BRW_HW_REG_TYPE_D: {
		if (src_type == BRW_HW_REG_TYPE_UD || src_type == BRW_HW_REG_TYPE_D)
			return reg;

		if (src_type == BRW_HW_REG_TYPE_UW)
			return kir_program_alu(prog, kir_zxwd, reg);
		else if (src_type == BRW_HW_REG_TYPE_W)
			return kir_program_alu(prog, kir_sxwd, reg);
		else if (src_type == BRW_HW_REG_TYPE_F)
			return kir_program_alu(prog, kir_ps2d, reg);

		ksim_unreachable("src type %d for ud/d dst type\n", src_type);
	}

	case BRW_HW_REG_TYPE_UW:
	case BRW_HW_REG_TYPE_W: {
		if (src_type == BRW_HW_REG_TYPE_UW || src_type == BRW_HW_REG_TYPE_W)
			return reg;

		ksim_unreachable("src type %d for uw/w dst type\n", src_type);
	}

	case BRW_HW_REG_TYPE_F: {
		if (src_type == BRW_HW_REG_TYPE_F)
			return reg;

		if (src_type == BRW_HW_REG_TYPE_UW) {
			kir_program_alu(prog, kir_zxwd, reg);
			return kir_program_alu(prog, kir_d2ps, prog->dst);
		} else if (src_type == BRW_HW_REG_TYPE_W) {
			kir_program_alu(prog, kir_sxwd, reg);
			return kir_program_alu(prog, kir_d2ps, prog->dst);
		} else if (src_type == BRW_HW_REG_TYPE_UD) {
			/* FIXME: Need to convert to int64 and then
			 * convert to floats as there is no uint32 to
			 * float cvt. */
			return kir_program_alu(prog, kir_d2ps, reg);
		} else if (src_type == BRW_HW_REG_TYPE_D) {
			return kir_program_alu(prog, kir_d2ps, reg);
		}

		ksim_unreachable("src type %d for float dst\n", src_type);
	}

	case GEN8_HW_REG_TYPE_UQ:
	case GEN8_HW_REG_TYPE_Q:
	default:
		ksim_unreachable("dst type\n", dst_type);
	}

	return kir_reg(0);
}

static struct kir_reg
kir_program_emit_src_load(struct kir_program *prog,
			  struct inst *inst, struct inst_src *src)
{
	struct inst_common common = unpack_inst_common(inst);
	struct inst_dst dst;
	int src_type;
	struct kir_insn *insn;
	struct kir_reg reg;

	src_type = src->type;
	if (src->file == BRW_ARCHITECTURE_REGISTER_FILE) {
		switch (src->num & 0xf0) {
		case BRW_ARF_NULL:
			reg = kir_reg(0);
			break;
		default:
			stub("architecture register file load");
			reg = kir_reg(0);
			break;
		}
	} else if (src->file == BRW_IMMEDIATE_VALUE) {
		switch (src->type) {
		case BRW_HW_REG_TYPE_UD:
		case BRW_HW_REG_TYPE_D:
		case BRW_HW_REG_TYPE_F: {
			insn = kir_program_add_insn(prog, kir_immd);
			insn->imm.d = unpack_inst_imm(inst).d;
			break;
		}
		case BRW_HW_REG_TYPE_UW:
		case BRW_HW_REG_TYPE_W:
			insn = kir_program_add_insn(prog, kir_immw);
			insn->imm.d = unpack_inst_imm(inst).d & 0xffff;
			break;

		case BRW_HW_REG_IMM_TYPE_UV:
			/* Gen6+ packed unsigned immediate vector */
			insn = kir_program_add_insn(prog, kir_immv);
			memcpy(insn->imm.v, unpack_inst_imm(inst).v, sizeof(insn->imm.v));
			src_type = BRW_HW_REG_TYPE_UW;
			break;

		case BRW_HW_REG_IMM_TYPE_VF:
			/* packed float immediate vector */
			insn = kir_program_add_insn(prog, kir_immvf);
			memcpy(insn->imm.vf, unpack_inst_imm(inst).vf, sizeof(insn->imm.vf));
			src_type = BRW_HW_REG_TYPE_F;
			break;

		case BRW_HW_REG_IMM_TYPE_V:
			/* packed int imm. vector; uword dest only */
			insn = kir_program_add_insn(prog, kir_immv);
			memcpy(insn->imm.v, unpack_inst_imm(inst).v, sizeof(insn->imm.v));
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
		reg = prog->dst;
	} else if (src->file == BRW_GENERAL_REGISTER_FILE) {
		struct eu_region region;

		if (common.access_mode == BRW_ALIGN_1)
			fill_region_for_src(&region, src, src->da1_subnum, prog);
		else
			fill_region_for_src(&region, src, src->da16_subnum, prog);

		reg = kir_program_load_region(prog, &region);
		reg = kir_program_emit_src_modifiers(prog, inst, src, reg);
	} else {
		stub("unhandled src");
		reg = kir_reg(0);
	}

	if (opcode_info[common.opcode].num_srcs == 3)
		dst = unpack_inst_3src_dst(inst);
	else
		dst = unpack_inst_2src_dst(inst);

	reg = kir_program_emit_type_conversion(prog, reg, dst.type, src_type);

	return reg;
}

static struct kir_reg
emit_not(struct kir_program *prog, struct kir_reg reg)
{
	kir_program_immd(prog, -1);
	return kir_program_alu(prog, kir_xor, reg, prog->dst);
}

static struct kir_reg
emit_cmp(struct kir_program *prog, int file, int type, int modifier,
	 struct kir_reg src0, struct kir_reg src1)
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

	if (is_integer(file, type)) {
		switch (modifier) {
		case BRW_CONDITIONAL_Z:
			return kir_program_alu(prog, kir_cmpeqd, src0, src1);
		case BRW_CONDITIONAL_NZ:
			return emit_not(prog, kir_program_alu(prog, kir_cmpeqd, src0, src1));
		case BRW_CONDITIONAL_G:
			return kir_program_alu(prog, kir_cmpgtd, src1, src0);
		case BRW_CONDITIONAL_GE:
			return emit_not(prog, kir_program_alu(prog, kir_cmpgtd, src0, src1));
		case BRW_CONDITIONAL_L:
			return kir_program_alu(prog, kir_cmpgtd, src0, src1);
		case BRW_CONDITIONAL_LE:
			return emit_not(prog, kir_program_alu(prog, kir_cmpgtd, src1, src0));
		default:
			stub("integer cmp op");
			return src0;
		}
	} else {
		return kir_program_alu(prog, kir_cmpf, src1, src0, eu_to_avx_cmp[modifier]);
	}
}

static void
kir_program_emit_dst_store(struct kir_program *prog,
			   struct kir_reg reg, struct inst *inst, struct inst_dst *dst)
{
	struct inst_common common = unpack_inst_common(inst);
	int subnum;

	/* FIXME: write masks */

	if (dst->file == BRW_ARCHITECTURE_REGISTER_FILE) {
		switch (dst->num) {
		case BRW_ARF_NULL:
			return;
		default:
			stub("arf store: %d\n", dst->num);
			return;
		}
	}

	if (dst->hstride > 1)
		stub("eu: dst hstride %d is > 1", dst->hstride);

	if (common.saturate) {
		struct kir_reg zero = kir_program_immf(prog, 0.0f);
		struct kir_reg one =  kir_program_immf(prog, 1.0f);
		ksim_assert(is_float(dst->file, dst->type));
		reg = kir_program_alu(prog, kir_maxf, reg, zero);
		reg = kir_program_alu(prog, kir_minf, reg, one);
	}

	if (common.access_mode == BRW_ALIGN_1)
		subnum = dst->da1_subnum;
	else
		subnum = dst->da16_subnum;

	struct eu_region region;
	fill_region_for_dst(&region, dst, subnum, prog);

	if (prog->scope > 0 && !common.mask_control) {
		struct kir_reg mask =
			kir_program_load_v8(prog, offsetof(struct thread, mask_stack[prog->scope]));
		kir_program_store_region_mask(prog, &region, reg, mask);
	} else {
		kir_program_store_region(prog, &region, reg);
	}
}

static inline int
reg_offset(int num, int subnum)
{
	return offsetof(struct thread, grf[num].ud[subnum]);
}

static void
builder_emit_sfid_thread_spawner(struct kir_program *prog, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);

	uint32_t opcode = field(send.function_control, 0, 0);
	uint32_t request = field(send.function_control, 1, 1);
	uint32_t resource_select = field(send.function_control, 4, 4);

	ksim_assert(send.eot);
	ksim_assert(opcode == 0 && request == 0 && resource_select == 1);
}

/* Vectorized AVX2 math functions from glibc's libmvec */
__m256 _ZGVdN8vv_powf(__m256 x, __m256 y);
__m256 _ZGVdN8v_logf(__m256 x);
__m256 _ZGVdN8v_expf(__m256 x);
__m256 _ZGVdN8v_sinf(__m256 x);
__m256 _ZGVdN8v_cosf(__m256 x);
__m256 _ZGVdN8vv_powf(__m256 x, __m256 y);

static __m256i
int_div_quotient(__m256i _n, __m256i _d)
{
	struct reg n, d, q, r;

	n.ireg = _n;
	d.ireg = _d;

#define div_channel(c)						\
	do {							\
		q.ud[c] = n.ud[c] / d.ud[c];			\
		r.ud[c] = n.ud[c] % d.ud[c];			\
	} while (0)

	div_channel(0);
	div_channel(1);
	div_channel(2);
	div_channel(3);
	div_channel(4);
	div_channel(5);
	div_channel(6);
	div_channel(7);

	(void) r;

	return q.ireg;
}

static __m256i
int_div_remainder(__m256i _n, __m256i _d)
{
	struct reg n, d, q, r;

	n.ireg = _n;
	d.ireg = _d;

	div_channel(0);
	div_channel(1);
	div_channel(2);
	div_channel(3);
	div_channel(4);
	div_channel(5);
	div_channel(6);
	div_channel(7);

	(void) q;

	return r.ireg;
}

static bool
compile_inst(struct kir_program *prog, struct inst *inst)
{
	struct inst_src src0, src1, src2;
	struct inst_dst dst;
	struct kir_reg dst_reg, src0_reg, src1_reg, src2_reg;
	uint32_t opcode = unpack_inst_common(inst).opcode;
	bool eot = false;

	if (opcode_info[opcode].num_srcs == 3) {
		src0 = unpack_inst_3src_src0(inst);
		src0_reg = kir_program_emit_src_load(prog, inst, &src0);
		src1 = unpack_inst_3src_src1(inst);
		src1_reg = kir_program_emit_src_load(prog, inst, &src1);
		src2 = unpack_inst_3src_src2(inst);
		src2_reg = kir_program_emit_src_load(prog, inst, &src2);
	} else if (opcode_info[opcode].num_srcs >= 1) {
		src0 = unpack_inst_2src_src0(inst);
		src0_reg = kir_program_emit_src_load(prog, inst, &src0);
	}

	if (opcode_info[opcode].num_srcs == 2) {
		src1 = unpack_inst_2src_src1(inst);
		src1_reg = kir_program_emit_src_load(prog, inst, &src1);
	}

	if (opcode_info[opcode].num_srcs == 3)
		dst = unpack_inst_3src_dst(inst);
	else
		dst = unpack_inst_2src_dst(inst);

	prog->new_scope = prog->scope;
	switch (opcode) {
	case BRW_OPCODE_MOV:
		kir_program_emit_dst_store(prog, src0_reg, inst, &dst);
		break;
	case BRW_OPCODE_SEL: {
		int modifier = unpack_inst_common(inst).cond_modifier;
		if (modifier == BRW_CONDITIONAL_GE) {
			kir_program_alu(prog, kir_maxf, src0_reg, src1_reg);
			break;
		}
		if (modifier == BRW_CONDITIONAL_L) {
			kir_program_alu(prog, kir_minf, src0_reg, src1_reg);
			break;
		}
		emit_cmp(prog, src0.file, src0.type, modifier, src0_reg, src1_reg);
		/* AVX2 blendv is opposite of the EU sel order, so we
		 * swap src0 and src1 operands. */
		kir_program_alu(prog, kir_blend, src0_reg, src1_reg, prog->dst);
		break;
	}
	case BRW_OPCODE_NOT: {
		kir_program_immd(prog, 0);
		kir_program_alu(prog, kir_xor, src0_reg, prog->dst);
		break;
	}
	case BRW_OPCODE_AND:
		kir_program_alu(prog, kir_and, src0_reg, src1_reg);
		break;
	case BRW_OPCODE_OR:
		kir_program_alu(prog, kir_or, src0_reg, src1_reg);
		break;
	case BRW_OPCODE_XOR:
		kir_program_alu(prog, kir_xor, src0_reg, src1_reg);
		break;
	case BRW_OPCODE_SHR:
		kir_program_alu(prog, kir_shr, src1_reg, src0_reg);
		break;
	case BRW_OPCODE_SHL:
		kir_program_alu(prog, kir_shl, src1_reg, src0_reg);
		break;
	case BRW_OPCODE_ASR:
		kir_program_alu(prog, kir_asr, src1_reg, src0_reg);
		break;
	case BRW_OPCODE_CMP: {
		int modifier = unpack_inst_common(inst).cond_modifier;
		emit_cmp(prog, src0.file, src0.type, modifier, src0_reg, src1_reg);
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
	case BRW_OPCODE_IF: {
		int flag_nr = unpack_inst_common(inst).flag_nr;
		struct kir_reg f = kir_program_load_v8(prog, offsetof(struct thread, f[flag_nr]));
		struct kir_reg mask = kir_program_load_v8(prog, offsetof(struct thread, mask_stack[prog->scope]));
		if (unpack_inst_common(inst).pred_inv)
			mask = kir_program_alu(prog, kir_andn, mask, f);
		else
			mask = kir_program_alu(prog, kir_and, mask, f);
		kir_program_store_v8(prog, offsetof(struct thread, mask_stack[prog->scope + 1]), mask);
		prog->new_scope = prog->scope + 1;
		break;
	}
	case BRW_OPCODE_IFF:
		stub("BRW_OPCODE_IFF");
		break;
	case BRW_OPCODE_ELSE: {
		ksim_assert(prog->scope > 0);
		struct kir_reg prev_mask = kir_program_load_v8(prog, offsetof(struct thread, mask_stack[prog->scope - 1]));
		struct kir_reg mask = kir_program_load_v8(prog, offsetof(struct thread, mask_stack[prog->scope]));
		mask = kir_program_alu(prog, kir_xor, prev_mask, mask);
		kir_program_store_v8(prog, offsetof(struct thread, mask_stack[prog->scope]), mask);
		break;
	}
	case BRW_OPCODE_ENDIF:
		prog->new_scope = prog->scope - 1;
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
			builder_emit_sfid_sampler(prog, inst);
			break;
		case GEN6_SFID_DATAPORT_RENDER_CACHE:
			builder_emit_sfid_render_cache(prog, inst);
			break;
		case BRW_SFID_URB:
			builder_emit_sfid_urb(prog, inst);
			break;
		case BRW_SFID_THREAD_SPAWNER:
			builder_emit_sfid_thread_spawner(prog, inst);
			break;
		case HSW_SFID_DATAPORT_DATA_CACHE_1:
			builder_emit_sfid_dataport1(prog, inst);
			break;
		case GEN6_SFID_DATAPORT_CONSTANT_CACHE:
			builder_emit_sfid_dataport_ro(prog, inst);
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
			kir_program_alu(prog, kir_rcp, src0_reg);
			break;
		case BRW_MATH_FUNCTION_LOG:
			kir_program_const_call(prog, _ZGVdN8v_logf, 1, src0_reg);
			break;
		case BRW_MATH_FUNCTION_EXP:
			kir_program_const_call(prog, _ZGVdN8v_expf, 1, src0_reg);
			break;
		case BRW_MATH_FUNCTION_SQRT:
			kir_program_alu(prog, kir_sqrt, src0_reg);
			break;
		case BRW_MATH_FUNCTION_RSQ:
			kir_program_alu(prog, kir_rsqrt, src0_reg);
			break;
		case BRW_MATH_FUNCTION_SIN:
			kir_program_const_call(prog, _ZGVdN8v_sinf, 1, src0_reg);
			break;
		case BRW_MATH_FUNCTION_COS:
			kir_program_const_call(prog, _ZGVdN8v_cosf, 1, src0_reg);
			break;
		case BRW_MATH_FUNCTION_SINCOS:
			ksim_unreachable("sincos only gen4/5");
			break;
		case BRW_MATH_FUNCTION_FDIV:
			kir_program_alu(prog, kir_divf, src0_reg, src1_reg);
			break;
		case BRW_MATH_FUNCTION_POW:
			kir_program_const_call(prog, _ZGVdN8vv_powf, 2, src0_reg, src1_reg);
			break;
		case BRW_MATH_FUNCTION_INT_DIV_QUOTIENT_AND_REMAINDER: {
			struct inst_dst dst2 = dst;
			kir_program_const_call(prog, int_div_remainder, 2, src0_reg, src1_reg);
			dst2.num++;
			kir_program_emit_dst_store(prog, prog->dst, inst, &dst2);

			kir_program_const_call(prog, int_div_quotient, 2, src0_reg, src1_reg);
			break;
		}
		case BRW_MATH_FUNCTION_INT_DIV_QUOTIENT: {
			kir_program_const_call(prog, int_div_quotient, 2, src0_reg, src1_reg);
			break;
		}
		case BRW_MATH_FUNCTION_INT_DIV_REMAINDER: {
			kir_program_const_call(prog, int_div_remainder, 2, src0_reg, src1_reg);
			break;
		}
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
			kir_program_alu(prog, kir_addd, src0_reg, src1_reg);
			break;
		case BRW_HW_REG_TYPE_UW:
		case BRW_HW_REG_TYPE_W:
			kir_program_alu(prog, kir_addw, src0_reg, src1_reg);
			break;
		case BRW_HW_REG_TYPE_F:
			kir_program_alu(prog, kir_addf, src0_reg, src1_reg);
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
			kir_program_alu(prog, kir_muld, src0_reg, src1_reg);
			break;
		case BRW_HW_REG_TYPE_UW:
		case BRW_HW_REG_TYPE_W:
			kir_program_alu(prog, kir_mulw, src0_reg, src1_reg);
			break;
		case BRW_HW_REG_TYPE_F:
			kir_program_alu(prog, kir_mulf, src0_reg, src1_reg);
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
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		kir_program_alu(prog, kir_rndd, src0_reg);
		kir_program_alu(prog, kir_subf, src0_reg, kir_program_immf(prog, 0.0f));
		break;
	}
	case BRW_OPCODE_RNDU:
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		kir_program_alu(prog, kir_rndu, src0_reg);
		break;
	case BRW_OPCODE_RNDD:
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		kir_program_alu(prog, kir_rndd, src0_reg);
		break;
	case BRW_OPCODE_RNDE:
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		kir_program_alu(prog, kir_rnde, src0_reg);
		break;
	case BRW_OPCODE_RNDZ:
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);
		kir_program_alu(prog, kir_rndz, src0_reg);
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
	case BRW_OPCODE_DP4:
		stub("BRW_OPCODE_DP4");
		break;
	case BRW_OPCODE_DPH:
		stub("BRW_OPCODE_DPH");
		break;
	case BRW_OPCODE_DP3:
		stub("BRW_OPCODE_DP3");
		break;
	case BRW_OPCODE_DP2:
		stub("BRW_OPCODE_DP2");
		break;
	case BRW_OPCODE_LINE: {
		src0 = unpack_inst_2src_src0(inst);
		src1 = unpack_inst_2src_src1(inst);
		ksim_assert(src0.type == BRW_HW_REG_TYPE_F);
		ksim_assert(src1.type == BRW_HW_REG_TYPE_F);
		int subnum = src0.da16_subnum / 4;

		src1_reg = kir_program_emit_src_load(prog, inst, &src1);
		struct kir_reg a_reg = kir_program_load_uniform(prog, reg_offset(src0.num, subnum));
		struct kir_reg c_reg = kir_program_load_uniform(prog, reg_offset(src0.num, subnum + 3));
		kir_program_alu(prog, kir_maddf, a_reg, src1_reg, c_reg);
		break;
	}
	case BRW_OPCODE_PLN: {
		src0 = unpack_inst_2src_src0(inst);
		src1 = unpack_inst_2src_src1(inst);
		ksim_assert(src0.type == BRW_HW_REG_TYPE_F);
		ksim_assert(src1.type == BRW_HW_REG_TYPE_F);

		src2 = unpack_inst_2src_src1(inst);
		src2.num++;

		int subnum = src0.da1_subnum / 4;
		src1_reg = kir_program_emit_src_load(prog, inst, &src1);
		struct kir_reg a_reg = kir_program_load_uniform(prog, reg_offset(src0.num, subnum));
		struct kir_reg c_reg = kir_program_load_uniform(prog, reg_offset(src0.num, subnum + 3));
		struct kir_reg t = kir_program_alu(prog, kir_maddf, a_reg, src1_reg, c_reg);
		struct kir_reg b_reg = kir_program_load_uniform(prog, reg_offset(src0.num, subnum + 1));
		src2_reg = kir_program_emit_src_load(prog, inst, &src2);
		kir_program_alu(prog, kir_maddf, b_reg, src2_reg, t);
		break;
	}
	case BRW_OPCODE_MAD:
		if (is_integer(dst.file, dst.type)) {
			kir_program_alu(prog, kir_muld, src1_reg, src2_reg);
			kir_program_alu(prog, kir_addd, src0_reg, prog->dst);
		} else {
			kir_program_alu(prog, kir_maddf, src1_reg, src2_reg, src0_reg);
		}
		break;
	case BRW_OPCODE_LRP:
		ksim_assert(src0.type == BRW_HW_REG_TYPE_F);
		ksim_assert(src1.type == BRW_HW_REG_TYPE_F);
		ksim_assert(src2.type == BRW_HW_REG_TYPE_F);
		ksim_assert(dst.type == BRW_HW_REG_TYPE_F);

		/* dst = src0 * src1 + (1 - src0) * src2
		 *     = src0 * src1 + src2 - src0 * src2
		 *     = src0 * (src1 - src2) + src2
		 */
		kir_program_alu(prog, kir_subf, src1_reg, src2_reg);
		kir_program_alu(prog, kir_maddf, src0_reg, prog->dst, src2_reg);
		break;
	case BRW_OPCODE_NENOP:
	case BRW_OPCODE_NOP:
		break;
	}

	dst_reg = prog->dst;

	uint32_t cond_modifier = unpack_inst_common(inst).cond_modifier;
	uint32_t flag = unpack_inst_common(inst).flag_nr;
	if (opcode != BRW_OPCODE_SEND && opcode != BRW_OPCODE_SENDC &&
	    opcode != BRW_OPCODE_MATH &&
	    cond_modifier != BRW_CONDITIONAL_NONE) {
		struct kir_reg flag_reg;
		if (opcode == BRW_OPCODE_CMP) {
			flag_reg = dst_reg;
		} else {
			struct kir_reg zero = kir_program_immd(prog, 0);
			flag_reg = emit_cmp(prog, src0.file, src0.type, cond_modifier, dst_reg, zero);
			/* FIXME: Mask store? */
		}
		kir_program_store_v8(prog, offsetof(struct thread, f[flag]), flag_reg);
	}

	if (opcode_info[opcode].store_dst)
		kir_program_emit_dst_store(prog, dst_reg, inst, &dst);

	return eot;
}

static bool
do_compile_inst(struct kir_program *prog, struct inst *inst)
{
	uint32_t opcode = unpack_inst_common(inst).opcode;
	int exec_size = 1 << unpack_inst_common(inst).exec_size;
	struct inst_dst dst;
	bool eot;

	if (opcode_info[opcode].num_srcs == 3)
		dst = unpack_inst_3src_dst(inst);
	else
		dst = unpack_inst_2src_dst(inst);

	if (exec_size * type_size(dst.type) < 64 ||
	    opcode == BRW_OPCODE_SEND || opcode == BRW_OPCODE_SENDC) {
		prog->exec_size = exec_size;
		prog->exec_offset = 0;
		eot = compile_inst(prog, inst);
	} else {
		prog->exec_size = exec_size / 2;
		prog->exec_offset = 0;
		eot = compile_inst(prog, inst);
		prog->exec_offset = exec_size / 2;
		compile_inst(prog, inst);
	}

	prog->scope = prog->new_scope;

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
kir_program_emit_shader(struct kir_program *prog, uint64_t kernel_offset)
{
	struct inst uncompacted;
	void *insn;
	bool eot;
	uint64_t ksp, range;
	void *p, *start;

	brw_init_compaction_tables(&ksim_devinfo);

	ksp = kernel_offset + gt.instruction_base_address;
	start = map_gtt_offset(ksp, &range);
	p = start;

	do {
		if (trace_mask & TRACE_EU)
			fprintf(trace_file, "%04lx  ", p - start);

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

		eot = do_compile_inst(prog, insn);
	} while (!eot);

	if (trace_mask & (TRACE_EU | TRACE_AVX))
		fprintf(trace_file, "\n");
}
