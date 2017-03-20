/*
 * Copyright © 2017 Kristian H. Kristensen
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

#include "eu.h"
#include "kir.h"

enum dp1_message_type {
	MSD1R_US		= 0x01,	/* Untyped surface read */
	MSD1R_DWAI2		= 0x02,	/* dword untyped atomic integer */
	MSD1R_DWAI2_4x2		= 0x03, /* SIMD4x2 Untyped atomic integer */
	MSD1R_TS		= 0x05,	/* Typed surface read */
	MSD1R_DWTAI2		= 0x06,	/* dword typed atomic integer */
	MSD1R_DWTAI_4x2		= 0x07,	/* SIMD4x2 typed atomic counter operation */
	MSD1W_US		= 0x09,	/* Untyped surface write */
	MSD1R_DWAC2		= 0x0b,	/* Atomic counter operation */
	MSD1R_DWAC2_4x2		= 0x0c,	/* SIMD4x2 atomic counter operation */
	MSD1W_TS		= 0x0d,	/* Typed surface write */
	MSD1R_A64_BS		= 0x10,	/* Scattered read */
	MSD1R_A64_US		= 0x11,	/* Untyped surface read */
	MSD1_A64_DWAI2		= 0x12,	/* Untyped Atomic Integer */
	MSD1W_A64_HWB		= 0x14,	/* hword block write */
	MSD1W_A64_US		= 0x11,	/* Untyped surface write */
	MSD1W_A64_BS		= 0x1a,	/* Scattered write */
	MSD1R_DWAF2		= 0x1b,	/* dword untyped atomic float */
	MSD1R_DWAF2_4x2		= 0x1c,	/* Untyped atomic float */
};

/* MDC_AOP1, MDC_AOP2 and MDC_AOP3 */
enum mdc_aop {
	MDC_AOP_CMPWR_2W	= 0x00, /* new_dst = (src0_2W == old_dst_2W) ? src1_2W : old_dst_2W */
	MDC_AOP_AND		= 0x01, /* [Default] new_dst = old_dst AND src0 */
	MDC_AOP_OR		= 0x02, /* new_dst = old_dst | src0 */
	MDC_AOP_XOR		= 0x03, /* new_dst = old_dst ^ src0 */
	MDC_AOP_MOV		= 0x04, /* new_dst = src0 */
	MDC_AOP_INC		= 0x05, /* [Default] new_dst = old_dst + 1 */
	MDC_AOP_DEC		= 0x06, /* new_dst = old_dst - 1 */
	MDC_AOP_ADD		= 0x07, /* new_dst = old_dst + src0 */
	MDC_AOP_SUB		= 0x08, /* new_dst = old_dst - src0 */
	MDC_AOP_REVSUB		= 0x09, /* new_dst = src0 - old_dst */
	MDC_AOP_IMAX		= 0x0A, /* new_dst = imax(old_dst, src0) */
	MDC_AOP_IMIN		= 0x0B, /* new_dst = imin(old_dst, src0) */
	MDC_AOP_UMAX		= 0x0C, /* new_dst = umax(old_dst, src0) */
	MDC_AOP_UMIN		= 0x0D, /* new_dst = umin(old_dst, src0) */
	MDC_AOP_CMPWR		= 0x0E, /* [Default] new_dst = (src0 == old_dst) ? src1 : old_dst */
	MDC_AOP_PREDEC		= 0x0F, /* new_dst = old_dst - 1 */
};

enum mdc_sm2r {
	MDC_SM2R_SIMD16		= 0x00,
	MDC_SM2R_SIMD8		= 0x01,
};

struct sfid_dataport1_args {
	enum mdc_sm2r simd_mode;
	int scope;
	uint32_t src;
	void *buffer;
	uint32_t mask;
};

static void
sfid_dataport1_untyped_write(struct thread *t, struct sfid_dataport1_args *args)
{
	uint32_t c;
	const uint32_t mask =
		_mm256_movemask_ps((__m256) t->mask[args->scope].q[0]) &
		t->grf[args->src].ud[7];

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

struct dp1_atomic_dword_message_descriptor {
	uint32_t			binding_table_index;
	enum mdc_aop			atomic_operation;
	enum mdc_sm2r			simd_mode;
	uint32_t			return_data_control;
	enum dp1_message_type		message_type;
	bool				header_present;
	uint32_t			response_length;
	uint32_t			message_length;
	uint32_t			return_format;
	bool				eot;
};

static inline struct dp1_atomic_dword_message_descriptor
unpack_dp1_atomic_dword_message_descriptor(uint32_t function_control)
{
	return (struct dp1_atomic_dword_message_descriptor) {
		.binding_table_index	= field(function_control,   0,    7),
		.atomic_operation	= field(function_control,   8,   11),
		.simd_mode		= field(function_control,  12,   12),
		.return_data_control	= field(function_control,  13,   13),
		.message_type		= field(function_control,  14,   18),
		.header_present		= field(function_control,  19,   19),
		.response_length	= field(function_control,  20,   24),
		.message_length		= field(function_control,  25,   28),
		.return_format		= field(function_control,  30,   30),
		.eot			= field(function_control,  31,   31),
	};
}

static void
sfid_dataport1_integer_atomic_inc(struct thread *t, struct sfid_dataport1_args *args)
{
	/* Header is MH1_BTS_PSM. Dword 7, bits 0-15 are channel masks. */
	/* Payload is MAP32B_USU_SIMD8, per-channel u */

	void *buffer = args->buffer;
	uint32_t c;

	uint32_t mask =
		_mm256_movemask_ps((__m256) t->mask[args->scope].q[0]) &
		t->grf[args->src].ud[7];
	uint32_t *u = t->grf[args->src + 1].ud;
	for_each_bit (c, mask) {
		uint32_t *dst = buffer + u[c];
		__atomic_add_fetch(dst, 1, __ATOMIC_RELAXED);
	}
	
	if (args->simd_mode == MDC_SM2R_SIMD8)
		return;

	mask =
		_mm256_movemask_ps((__m256) t->mask[args->scope].q[1]) &
		(t->grf[args->src].ud[7] >> 8);
	u = t->grf[args->src + 2].ud;
	for_each_bit (c, mask) {
		uint32_t *dst = buffer + u[c];
		__atomic_add_fetch(dst, 1, __ATOMIC_RELAXED);
	}
}

static void
sfid_dataport1_integer_atomic_predec(struct thread *t, struct sfid_dataport1_args *args)
{
	/* Header is MH1_BTS_PSM. Dword 7, bits 0-15 are channel masks. */
	/* Payload is MAP32B_USU_SIMD8, per-channel u */

	void *buffer = args->buffer;
	uint32_t c;

	uint32_t mask =
		_mm256_movemask_ps((__m256) t->mask[args->scope].q[0]) &
		t->grf[args->src].ud[7];
	uint32_t *u = t->grf[args->src + 1].ud;
	for_each_bit (c, mask) {
		uint32_t *dst = buffer + u[c];
		__atomic_add_fetch(dst, -1, __ATOMIC_RELAXED);
	}
	
	if (args->simd_mode == MDC_SM2R_SIMD8)
		return;

	mask =
		_mm256_movemask_ps((__m256) t->mask[args->scope].q[1]) &
		(t->grf[args->src].ud[7] >> 8);
	u = t->grf[args->src + 2].ud;
	for_each_bit (c, mask) {
		uint32_t *dst = buffer + u[c];
		__atomic_add_fetch(dst, -1, __ATOMIC_RELAXED);
	}
}

static void
emit_dword_atomic_integer(struct kir_program *prog, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	struct dp1_atomic_dword_message_descriptor m =
		unpack_dp1_atomic_dword_message_descriptor(send.function_control);
	struct sfid_dataport1_args *args;
	struct surface s;
	void *func;

	switch (m.atomic_operation) {
	case MDC_AOP_INC:
		func = sfid_dataport1_integer_atomic_inc;
		break;
	case MDC_AOP_PREDEC:
		func = sfid_dataport1_integer_atomic_predec;
		break;
	default:
		stub("AOP");
		break;
	}

	ksim_assert(m.header_present);
	args = get_const_data(sizeof *args, 8);
	bool valid = get_surface(prog->binding_table_address,
				 m.binding_table_index, &s);
	ksim_assert(valid);
	args->src = unpack_inst_2src_src0(inst).num;
	args->buffer = s.pixels;
	args->simd_mode = m.simd_mode;
	args->scope = prog->scope;
	kir_program_send(prog, inst, func, args);
}

void
builder_emit_sfid_dataport1(struct kir_program *prog, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	struct surface buffer;
	void *func;
	
	uint32_t bti = field(send.function_control, 0, 7);
	uint32_t mask = field(send.function_control, 8, 11);
	uint32_t simd_mode = field(send.function_control, 12, 13);
	uint32_t opcode = field(send.function_control, 14, 18);
	//uint32_t header_present = field(send.function_control, 19, 19);

	struct sfid_dataport1_args *args;
	args = get_const_data(sizeof *args, 8);

	switch (opcode) {
	case MSD1R_DWAI2:
		emit_dword_atomic_integer(prog, inst);
		break;
		
	case MSD1W_US:
		/* Command reference: MSD1W_US */
		ksim_assert(simd_mode == 2); /* SIMD8 */
		args->src = unpack_inst_2src_src0(inst).num;
		args->mask = mask;
		bool valid = get_surface(prog->binding_table_address,
					 bti, &buffer);
		ksim_assert(valid);
		args->buffer = buffer.pixels;

		func = sfid_dataport1_untyped_write;
		kir_program_send(prog, inst, func, args);
		break;

	default:
		stub("dataport1 opcode");
		func = NULL;
		break;
	}
}

enum dp_ro_message_type {
	/* Vol 2a, MSD_CC_* */
	MT_CC_OWB	= 0x00, /* [Default] Oword Block Read Constant Cache message */
	MT_CC_OWUB	= 0x01, /* Unaligned Oword Block Read Constant Cache message */
	MT_CC_OWDB	= 0x02, /* Oword Dual Block Read Constant Cache message */
	MT_CC_DWS	= 0x03, /* Dword Scattered Read Constant Cache message */
	MT_SC_OWUB	= 0x04, /* Unaligned Oword Block Read Sampler Cache message */
	MT_SC_MB	= 0x05,	/* Media Block Read Sampler Cache message */
	MT_RSI		= 0x06,	/* Read Surface Info message */
};

enum dp_ro_data_elements {
	OW1L = 0x00, /* 1 Oword, read into or written from the low 128 bits of the destination register */
	OW1U = 0x01, /* 1 Oword, read into or written from the high 128 bits of the destination register */
	OW2 = 0x02, /* 2 Owords */
	OW4 = 0x03, /* 4 Owords */
	OW8 = 0x04, /* 8 Owords */
};

struct dp_ro_message_descriptor {
	uint32_t			binding_table_index;
	enum dp_ro_data_elements	data_elements;
	bool				legacy_simd_mode;
	bool				simd_mode;
	uint32_t			invalidate_after_read;
	enum dp_ro_message_type		message_type;
	bool				legacy_message;
	bool				header_present;
	uint32_t			response_length;
	uint32_t			message_length;
	uint32_t			return_format;
	bool				eot;
};

static inline struct dp_ro_message_descriptor
unpack_dp_ro_message_descriptor(uint32_t function_control)
{
	return (struct dp_ro_message_descriptor) {
		.binding_table_index	= field(function_control,   0,    7),
		.data_elements		= field(function_control,   8,   10),
		.simd_mode		= field(function_control,   8,    8),
		.legacy_simd_mode	= field(function_control,   9,    9),
		.invalidate_after_read	= field(function_control,  13,   13),
		.message_type		= field(function_control,  14,   17),
		.legacy_message		= field(function_control,  18,   18),
		.header_present		= field(function_control,  19,   19),
		.response_length	= field(function_control,  20,   24),
		.message_length		= field(function_control,  25,   28),
		.return_format		= field(function_control,  30,   30),
		.eot			= field(function_control,  31,   31),
	};
}

void
builder_emit_sfid_dataport_ro(struct kir_program *prog, struct inst *inst)
{
	const struct inst_send send = unpack_inst_send(inst);
	const struct dp_ro_message_descriptor md =
		unpack_dp_ro_message_descriptor(send.function_control);
	struct inst_src src = unpack_inst_2src_src0(inst);
	struct inst_dst dst = unpack_inst_2src_dst(inst);
	struct surface buffer;
	bool valid;
	struct kir_reg v, offset, base;

	switch (md.message_type) {
	case MT_CC_OWB:
		valid = get_surface(prog->binding_table_address,
				    md.binding_table_index,
				    &buffer);
		ksim_assert(valid);
		switch (md.data_elements) {
		case OW4:
			kir_program_comment(prog, "ro dp read 4 ow from bti %d",
					    md.binding_table_index);

			/* FIXME: We need constant propagation at this
			 * point to recognize that r72.2 (for example)
			 * is constant and we can compute the exact
			 * address at comppile time. Something like,
			 *
			 *     if (is_constant(grf, 72, 2, &value) {
			 *         base = load_base_imm(buffer.pixels);
			 *         load(prog, base, value * 16 + 0);
			 *         load(prog, base, value * 16 + 32);
			 *     } else {
			 *         what we have below now...
			 *     }
			 *
			 * and then ideally multiple ubo loads from
			 * the same ubo will use the same
			 * load_base_imm.
			 */

			offset = kir_program_load_v8(prog, offsetof(struct thread, grf[src.num]));
			/* Offset is in owords; multiply by 16. */
			offset = kir_program_alu(prog, kir_shli, offset, 4);
			base = kir_program_set_load_base_imm_offset(prog, buffer.pixels, offset);

			v = kir_program_load(prog, base, 0);
			kir_program_store_v8(prog, offsetof(struct thread, grf[dst.num]), v);
			v = kir_program_load(prog, base, 32);
			kir_program_store_v8(prog, offsetof(struct thread, grf[dst.num + 1]), v);
			break;
		default:
			stub("unhandled md.data_elements");
			break;
		}
		break;
	default:
		stub("dp_ro message type %d", md.message_type);
		break;
	}
}

