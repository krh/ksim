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

static int
builder_emit_da_src_load(struct builder *bld,
			 struct inst *inst, struct inst_src *src, int subnum_bytes)
{
	int subnum = subnum_bytes / type_size(src->type);
        int reg = builder_get_reg(bld);

	subnum += bld->exec_offset / src->width * src->vstride;

	if (src->hstride == 1 && src->width == src->vstride) {
		switch (type_size(src->type) * bld->exec_size) {
		case 32:
			builder_emit_m256i_load(bld, reg,
						offsetof(struct thread, grf[src->num].ud[subnum]));
			break;
		case 16:
		default:
			/* Could do broadcastq/d/w for sizes 8, 4 and
			 * 2 to avoid loading too much */
			builder_emit_m128i_load(bld, reg,
						offsetof(struct thread, grf[src->num].uw[subnum]));
			break;
		}
	} else if (src->hstride == 0 && src->vstride == 0 && src->width == 1) {
		switch (type_size(src->type)) {
		case 4:
			builder_emit_vpbroadcastd(bld, reg,
						  offsetof(struct thread,
							   grf[src->num].ud[subnum]));
			break;
		default:
			stub("unhandled broadcast load size %d\n", type_size(src->type));
			break;
		}
	} else if (src->hstride == 0 && src->width == 4 && src->vstride == 1 &&
		   type_size(src->type) == 2) {
		int tmp0_reg = builder_get_reg(bld);
		int tmp1_reg = builder_get_reg(bld);

		/* Handle the frag coord region */
		builder_emit_vpbroadcastw(bld, tmp0_reg,
					  offsetof(struct thread, grf[src->num].uw[subnum]));
		builder_emit_vpbroadcastw(bld, tmp1_reg,
					  offsetof(struct thread, grf[src->num].uw[subnum + 2]));
		builder_emit_vinserti128(bld, tmp0_reg, tmp1_reg, tmp0_reg, 1);

		builder_emit_vpbroadcastw(bld, reg,
					  offsetof(struct thread, grf[src->num].uw[subnum + 1]));
		builder_emit_vpbroadcastw(bld, tmp1_reg,
					  offsetof(struct thread, grf[src->num].uw[subnum + 3]));
		builder_emit_vinserti128(bld, reg, tmp1_reg, reg, 1);

		builder_emit_vpblendd(bld, reg, 0xcc, reg, tmp0_reg);
	} else if (src->hstride == 1 && src->width * src->type == 8) {
		for (int i = 0; i < bld->exec_size / src->width; i++) {
			int offset = offsetof(struct thread, grf[src->num].uw[subnum + i * src->vstride]);
			builder_emit_vpinsrq_rdi_relative(bld, reg, reg, offset, i & 1);
		}
	} else if (type_size(src->type) == 4) {
		int offset, i = 0, tmp_reg = reg;

		for (int y = 0; y < bld->exec_size / src->width; y++) {
			for (int x = 0; x < src->width; x++) {
				if (i == 4)
					tmp_reg = builder_get_reg(bld);
				offset = offsetof(struct thread, grf[src->num].ud[subnum + y * src->vstride + x * src->hstride]);
				builder_emit_vpinsrd_rdi_relative(bld, tmp_reg, tmp_reg, offset, i & 3);
				i++;
			}
		}
		if (tmp_reg != reg)
			builder_emit_vinserti128(bld, reg, tmp_reg, reg, 1);
	} else {
		stub("src: g%d.%d<%d,%d,%d>",
		     src->num, subnum, src->vstride, src->width, src->hstride);
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

	} else if (common.access_mode == BRW_ALIGN_1) {
		reg = builder_emit_da_src_load(bld, inst, src, src->da1_subnum);
	} else if (common.access_mode == BRW_ALIGN_16) {
		reg = builder_emit_da_src_load(bld, inst, src, src->da16_subnum);
	} else {
		stub("unhandled src");
		reg = 0;
	}

	if (opcode_info[common.opcode].num_srcs == 3)
		dst = unpack_inst_3src_dst(inst);
	else
		dst = unpack_inst_2src_dst(inst);

	reg = builder_emit_type_conversion(bld, reg, dst.type, src_type);

	/* FIXME: Build the load above into the source modifier when possible, eg:
	 *
	 *     vpabsd 0x456(%rdi), %ymm1
	 */

	if (src->abs) {
		if (src->type == BRW_HW_REG_TYPE_F) {
			int tmp_reg = builder_get_reg_with_uniform(bld, 0x7fffffff);
			builder_emit_vpand(bld, reg, reg, tmp_reg);
		} else {
			builder_emit_vpabsd(bld, reg, reg);
		}
	}

	if (src->negate) {
		int tmp_reg = builder_get_reg_with_uniform(bld, 0);

		if (is_logic_instruction(inst)) {
			builder_emit_vpxor(bld, reg, reg, tmp_reg);
		} else if (src->type == BRW_HW_REG_TYPE_F) {
			builder_emit_vsubps(bld, reg, tmp_reg, reg);
		} else {
			builder_emit_vpsubd(bld, reg, tmp_reg, reg);
		}
	}

	return reg;
}

static void
builder_emit_cmp(struct builder *bld, int modifier, int dst, int src0, int src1)
{
	switch (modifier) {
	case BRW_CONDITIONAL_NONE:
		/* assert: must have both pred */
		break;
	case BRW_CONDITIONAL_Z:
		builder_emit_vcmpps(bld, 0, dst, src0, src1);
		break;
	case BRW_CONDITIONAL_NZ:
		builder_emit_vcmpps(bld, 4, dst, src0, src1);
		break;
	case BRW_CONDITIONAL_G:
		builder_emit_vcmpps(bld, 14, dst, src0, src1);
		break;
	case BRW_CONDITIONAL_GE:
		builder_emit_vcmpps(bld, 13, dst, src0, src1);
		break;
	case BRW_CONDITIONAL_L:
		builder_emit_vcmpps(bld, 1, dst, src0, src1);
		break;
	case BRW_CONDITIONAL_LE:
		builder_emit_vcmpps(bld, 2, dst, src0, src1);
		break;
	case BRW_CONDITIONAL_R:
		stub("BRW_CONDITIONAL_R");
		break;
	case BRW_CONDITIONAL_O:
		stub("BRW_CONDITIONAL_O");
		break;
	case BRW_CONDITIONAL_U:
		stub("BRW_CONDITIONAL_U");
		break;
	}
}

static void
builder_emit_dst_store(struct builder *bld, int avx_reg,
		       struct inst *inst, struct inst_dst *dst)
{
	struct inst_common common = unpack_inst_common(inst);

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

	switch (bld->exec_size * type_size(dst->type)) {
	case 32:
		builder_emit_m256i_store(bld, avx_reg,
					 bld->exec_offset * type_size(dst->type) +
					 offsetof(struct thread, grf[dst->num]));
		break;
	case 16:
		builder_emit_m128i_store(bld, avx_reg,
					 offsetof(struct thread, grf[dst->num]));
		break;
	case 4:
		builder_emit_u32_store(bld, avx_reg,
				       offsetof(struct thread, grf[dst->num]) +
				       dst->da1_subnum);
		break;
	default:
		stub("eu: type size %d in dest store", type_size(dst->type));
		break;
	}
}

static inline int
reg_offset(int num, int subnum)
{
	return offsetof(struct thread, grf[num].ud[subnum]);
}

static void *
builder_emit_sfid_urb(struct builder *bld, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	struct sfid_urb_args *args;
	args = builder_get_const_data(bld, sizeof *args, 8);

	args->src = unpack_inst_2src_src0(inst).num;
	args->len = send.mlen;
	args->offset = field(send.function_control, 4, 14);

	uint32_t opcode = field(send.function_control, 0, 3);
	bool per_slot_offset = field(send.function_control, 17, 17);

	builder_emit_load_rsi_rip_relative(bld, builder_offset(bld, args));

	switch (opcode) {
	case 0: /* write HWord */
	case 1: /* write OWord */
	case 2: /* read HWord */
	case 3: /* read OWord */
	case 4: /* atomic mov */
	case 5: /* atomic inc */
	case 6: /* atomic add */
		stub("sfid urb opcode %d", opcode);
		return NULL;
	case 7: /* SIMD8 write */
		ksim_assert(send.header_present);
		ksim_assert(send.rlen == 0);
		ksim_assert(!per_slot_offset);
		return sfid_urb_simd8_write;
	default:
		ksim_unreachable("out of range urb opcode: %d", opcode);
		return NULL;
	}
}

static void *
builder_emit_sfid_thread_spawner(struct builder *bld, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);

	uint32_t opcode = field(send.function_control, 0, 0);
	uint32_t request = field(send.function_control, 1, 1);
	uint32_t resource_select = field(send.function_control, 4, 4);

	ksim_assert(send.eot);
	ksim_assert(opcode == 0 && request == 0 && resource_select == 1);

	return NULL;
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

static void *
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

		return sfid_dataport1_untyped_write;
	default:
		stub("dataport1 opcode");
		return NULL;
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
		struct avx2_reg *ymm0 = &bld->regs[0];

		list_remove(&ymm0->link);
		list_insert(&bld->regs_lru_list, &ymm0->link);
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

		void *p;
		switch (send.sfid) {
		case BRW_SFID_SAMPLER:
			p = builder_emit_sfid_sampler(bld, inst);
			break;
		case GEN6_SFID_DATAPORT_RENDER_CACHE:
			p = builder_emit_sfid_render_cache(bld, inst);
			break;
		case BRW_SFID_URB:
			p = builder_emit_sfid_urb(bld, inst);
			break;
		case BRW_SFID_THREAD_SPAWNER:
			p = builder_emit_sfid_thread_spawner(bld, inst);
			break;
		case HSW_SFID_DATAPORT_DATA_CACHE_1:
			p = builder_emit_sfid_dataport1(bld, inst);
			break;
		default:
			stub("sfid: %d", send.sfid);
			break;
		}

		/* If func is NULL, it's the special case of a compute
		 * thread terminating. Just return.
		 */
		if (p == NULL) {
			builder_emit_ret(bld);
			break;
		}

		/* In case of eot, we end the thread by jumping
		 * (instead of calling) to the sfid implementation.
		 * When the sfid implementation returns it will return
		 * to our caller when it's done (tail-call
		 * optimization).
		 */
		if (eot) {
			builder_emit_jmp_relative(bld, (uint8_t *) p - bld->p);
		} else {
			builder_emit_call(bld, p);
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
			break;
		case BRW_MATH_FUNCTION_EXP:
			dst_reg = builder_emit_call(bld, _ZGVdN8v___expf_finite);
			break;
		case BRW_MATH_FUNCTION_SQRT:
			builder_emit_vsqrtps(bld, dst_reg, src0_reg);
			break;
		case BRW_MATH_FUNCTION_RSQ:
			builder_emit_vrsqrtps(bld, dst_reg, src0_reg);
			break;
		case BRW_MATH_FUNCTION_SIN:
			dst_reg = builder_emit_call(bld, _ZGVdN8v_sinf);
			break;
		case BRW_MATH_FUNCTION_COS:
			dst_reg = builder_emit_call(bld, _ZGVdN8v_cosf);
			break;
		case BRW_MATH_FUNCTION_SINCOS:
			ksim_unreachable("sincos only gen4/5");
			break;
		case BRW_MATH_FUNCTION_FDIV:
			builder_emit_vdivps(bld, dst_reg, src0_reg, src1_reg);
			break;
		case BRW_MATH_FUNCTION_POW: {
			dst_reg = builder_emit_call(bld, _ZGVdN8vv___powf_finite);
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
		builder_emit_vsubps(bld, dst_reg, tmp_reg, src0_reg);
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
		stub("BRW_OPCODE_LRP");
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

struct shader *
compile_shader(uint64_t kernel_offset,
	       uint64_t surfaces, uint64_t samplers)
{
	struct builder bld;
	struct inst uncompacted;
	void *insn;
	bool eot;
	uint64_t ksp, range;
	void *p;

	brw_init_compaction_tables(&ksim_devinfo);

	ksp = kernel_offset + gt.instruction_base_address;
	p = map_gtt_offset(ksp, &range);

	builder_init(&bld, surfaces, samplers);

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

		eot = do_compile_inst(&bld, insn);

		if (trace_mask & TRACE_AVX)
			while (builder_disasm(&bld))
				fprintf(trace_file, "      %s\n", bld.disasm_output);
	} while (!eot);

	builder_finish(&bld);

	if (trace_mask & (TRACE_EU | TRACE_AVX))
		fprintf(trace_file, "\n");

	return bld.shader;
}
