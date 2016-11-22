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

#include <stdint.h>
#include <stdbool.h>

#include "ksim.h"

enum brw_eu_type {
	BRW_HW_REG_TYPE_UD		= 0,
	BRW_HW_REG_TYPE_D		= 1,
	BRW_HW_REG_TYPE_UW		= 2,
	BRW_HW_REG_TYPE_W		= 3,
	BRW_HW_REG_TYPE_F		= 7,
	GEN8_HW_REG_TYPE_UQ		= 8,
	GEN8_HW_REG_TYPE_Q		= 9,

	BRW_HW_REG_NON_IMM_TYPE_UB	= 4,
	BRW_HW_REG_NON_IMM_TYPE_B	= 5,
	GEN7_HW_REG_NON_IMM_TYPE_DF	= 6,
	GEN8_HW_REG_NON_IMM_TYPE_HF	= 10,

	BRW_HW_REG_IMM_TYPE_UV		= 4 /* Gen6+ packed unsigned immediate vector */,
	BRW_HW_REG_IMM_TYPE_VF		= 5 /* packed float immediate vector */,
	BRW_HW_REG_IMM_TYPE_V		= 6 /* packed int imm. vector; uword dest only */,
	GEN8_HW_REG_IMM_TYPE_DF		= 10,
	GEN8_HW_REG_IMM_TYPE_HF		= 11,
};

static inline const char *
eu_type_to_string(uint32_t type)
{
	static const char * const reg_encoding[] = {
		[BRW_HW_REG_TYPE_UD]          = "UD",
		[BRW_HW_REG_TYPE_D]           = "D",
		[BRW_HW_REG_TYPE_UW]          = "UW",
		[BRW_HW_REG_TYPE_W]           = "W",
		[BRW_HW_REG_NON_IMM_TYPE_UB]  = "UB",
		[BRW_HW_REG_NON_IMM_TYPE_B]   = "B",
		[GEN7_HW_REG_NON_IMM_TYPE_DF] = "DF",
		[BRW_HW_REG_TYPE_F]           = "F",
		[GEN8_HW_REG_TYPE_UQ]         = "UQ",
		[GEN8_HW_REG_TYPE_Q]          = "Q",
		[GEN8_HW_REG_NON_IMM_TYPE_HF] = "HF",
	};

	return reg_encoding[type];
}

enum brw_opcode {
	BRW_OPCODE_MOV			= 1,
	BRW_OPCODE_SEL			= 2,
	BRW_OPCODE_NOT			= 4,
	BRW_OPCODE_AND			= 5,
	BRW_OPCODE_OR			= 6,
	BRW_OPCODE_XOR			= 7,
	BRW_OPCODE_SHR			= 8,
	BRW_OPCODE_SHL			= 9,
	BRW_OPCODE_ASR			= 12,
	BRW_OPCODE_CMP			= 16,
	BRW_OPCODE_CMPN			= 17,
	BRW_OPCODE_CSEL			= 18,  /**< Gen8+ */
	BRW_OPCODE_F32TO16		= 19,  /**< Gen7 only */
	BRW_OPCODE_F16TO32		= 20,  /**< Gen7 only */
	BRW_OPCODE_BFREV		= 23,  /**< Gen7+ */
	BRW_OPCODE_BFE			= 24,  /**< Gen7+ */
	BRW_OPCODE_BFI1			= 25,  /**< Gen7+ */
	BRW_OPCODE_BFI2			= 26,  /**< Gen7+ */
	BRW_OPCODE_JMPI			= 32,
	BRW_OPCODE_IF			= 34,
	BRW_OPCODE_IFF			= 35,  /**< Pre-Gen6 */
	BRW_OPCODE_ELSE			= 36,
	BRW_OPCODE_ENDIF		= 37,
	BRW_OPCODE_DO			= 38,
	BRW_OPCODE_WHILE		= 39,
	BRW_OPCODE_BREAK		= 40,
	BRW_OPCODE_CONTINUE		= 41,
	BRW_OPCODE_HALT			= 42,
	BRW_OPCODE_MSAVE		= 44,  /**< Pre-Gen6 */
	BRW_OPCODE_MRESTORE		= 45, /**< Pre-Gen6 */
	BRW_OPCODE_PUSH			= 46,  /**< Pre-Gen6 */
	BRW_OPCODE_GOTO			= 46,  /**< Gen8+    */
	BRW_OPCODE_POP			= 47,  /**< Pre-Gen6 */
	BRW_OPCODE_WAIT			= 48,
	BRW_OPCODE_SEND			= 49,
	BRW_OPCODE_SENDC		= 50,
	BRW_OPCODE_MATH			= 56,  /**< Gen6+ */
	BRW_OPCODE_ADD			= 64,
	BRW_OPCODE_MUL			= 65,
	BRW_OPCODE_AVG			= 66,
	BRW_OPCODE_FRC			= 67,
	BRW_OPCODE_RNDU			= 68,
	BRW_OPCODE_RNDD			= 69,
	BRW_OPCODE_RNDE			= 70,
	BRW_OPCODE_RNDZ			= 71,
	BRW_OPCODE_MAC			= 72,
	BRW_OPCODE_MACH			= 73,
	BRW_OPCODE_LZD			= 74,
	BRW_OPCODE_FBH			= 75,  /**< Gen7+ */
	BRW_OPCODE_FBL			= 76,  /**< Gen7+ */
	BRW_OPCODE_CBIT			= 77,  /**< Gen7+ */
	BRW_OPCODE_ADDC			= 78,  /**< Gen7+ */
	BRW_OPCODE_SUBB			= 79,  /**< Gen7+ */
	BRW_OPCODE_SAD2			= 80,
	BRW_OPCODE_SADA2		= 81,
	BRW_OPCODE_DP4			= 84,
	BRW_OPCODE_DPH			= 85,
	BRW_OPCODE_DP3			= 86,
	BRW_OPCODE_DP2			= 87,
	BRW_OPCODE_LINE			= 89,
	BRW_OPCODE_PLN			= 90,  /**< G45+ */
	BRW_OPCODE_MAD			= 91,  /**< Gen6+ */
	BRW_OPCODE_LRP			= 92,  /**< Gen6+ */
	BRW_OPCODE_NENOP		= 125, /**< G45 only */
	BRW_OPCODE_NOP			= 126,
};

enum brw_message_target {
	BRW_SFID_NULL                     = 0,
	BRW_SFID_MATH                     = 1, /* Only valid on Gen4-5 */
	BRW_SFID_SAMPLER                  = 2,
	BRW_SFID_MESSAGE_GATEWAY          = 3,
	BRW_SFID_DATAPORT_READ            = 4,
	BRW_SFID_DATAPORT_WRITE           = 5,
	BRW_SFID_URB                      = 6,
	BRW_SFID_THREAD_SPAWNER           = 7,
	BRW_SFID_VME                      = 8,

	GEN6_SFID_DATAPORT_SAMPLER_CACHE  = 4,
	GEN6_SFID_DATAPORT_RENDER_CACHE   = 5,
	GEN6_SFID_DATAPORT_CONSTANT_CACHE = 9,

	GEN7_SFID_DATAPORT_DATA_CACHE     = 10,
	GEN7_SFID_PIXEL_INTERPOLATOR      = 11,
	HSW_SFID_DATAPORT_DATA_CACHE_1    = 12,
	HSW_SFID_CRE                      = 13,
};

enum brw_math_function {
	BRW_MATH_FUNCTION_INV			= 1,
	BRW_MATH_FUNCTION_LOG			= 2,
	BRW_MATH_FUNCTION_EXP			= 3,
	BRW_MATH_FUNCTION_SQRT			= 4,
	BRW_MATH_FUNCTION_RSQ			= 5,
	BRW_MATH_FUNCTION_SIN			= 6,
	BRW_MATH_FUNCTION_COS			= 7,
	BRW_MATH_FUNCTION_SINCOS		= 8 /* gen4, gen5 */,
	BRW_MATH_FUNCTION_FDIV			= 9 /* gen6+ */,
	BRW_MATH_FUNCTION_POW			= 10,
	BRW_MATH_FUNCTION_INT_DIV_QUOTIENT_AND_REMAINDER = 11,
	BRW_MATH_FUNCTION_INT_DIV_QUOTIENT	= 12,
	BRW_MATH_FUNCTION_INT_DIV_REMAINDER	= 13,
	GEN8_MATH_FUNCTION_INVM			= 14,
	GEN8_MATH_FUNCTION_RSQRTM		= 15,
};

#define BRW_ARCHITECTURE_REGISTER_FILE    0
#define BRW_GENERAL_REGISTER_FILE         1
#define BRW_MESSAGE_REGISTER_FILE         2
#define BRW_IMMEDIATE_VALUE               3

#define BRW_ARF_NULL                  0x00
#define BRW_ARF_ADDRESS               0x10
#define BRW_ARF_ACCUMULATOR           0x20
#define BRW_ARF_FLAG                  0x30
#define BRW_ARF_MASK                  0x40
#define BRW_ARF_MASK_STACK            0x50
#define BRW_ARF_MASK_STACK_DEPTH      0x60
#define BRW_ARF_STATE                 0x70
#define BRW_ARF_CONTROL               0x80
#define BRW_ARF_NOTIFICATION_COUNT    0x90
#define BRW_ARF_IP                    0xA0
#define BRW_ARF_TDR                   0xB0
#define BRW_ARF_TIMESTAMP             0xC0

#define BRW_ALIGN_1   0
#define BRW_ALIGN_16  1

#define BRW_ADDRESS_DIRECT                        0
#define BRW_ADDRESS_REGISTER_INDIRECT_REGISTER    1

#define BRW_3SRC_TYPE_F  0
#define BRW_3SRC_TYPE_D  1
#define BRW_3SRC_TYPE_UD 2
#define BRW_3SRC_TYPE_DF 3

enum {
	BRW_CONDITIONAL_NONE	= 0,
	BRW_CONDITIONAL_Z	= 1,
	BRW_CONDITIONAL_NZ	= 2,
	BRW_CONDITIONAL_EQ	= 1,	/* Z */
	BRW_CONDITIONAL_NEQ	= 2,	/* NZ */
	BRW_CONDITIONAL_G	= 3,
	BRW_CONDITIONAL_GE	= 4,
	BRW_CONDITIONAL_L	= 5,
	BRW_CONDITIONAL_LE	= 6,
	BRW_CONDITIONAL_R	= 7,    /* Gen <= 5 */
	BRW_CONDITIONAL_O	= 8,
	BRW_CONDITIONAL_U	= 9,
};

struct inst_common {
	uint32_t opcode;
	uint32_t access_mode;
	uint32_t no_dd_clear;
	uint32_t no_dd_check;
	uint32_t nib_control;
	uint32_t qtr_control;
	uint32_t thread_control;
	uint32_t pred_control;
	uint32_t pred_inv;
	uint32_t exec_size;
	uint32_t math_function;
	uint32_t cond_modifier;
	uint32_t acc_wr_control;
	uint32_t branch_control;
	uint32_t cmpt_control;
	uint32_t debug_control;
	uint32_t saturate;
	uint32_t flag_subreg_nr;
	uint32_t flag_nr;
	uint32_t mask_control;
};

struct inst_dst {
	uint32_t type;
	uint32_t file;
	uint32_t num;
	uint32_t da1_subnum;
	uint32_t da16_subnum;
	uint32_t ia_subnum;
	uint32_t hstride;
	uint32_t address_mode;
	uint32_t writemask;
};

struct inst_src {
	uint32_t vstride;
	uint32_t width;
	uint32_t swiz_w;
	uint32_t swiz_z;
	uint32_t hstride;
	uint32_t address_mode;
	uint32_t negate;
	uint32_t abs;

	uint32_t ia_subnum;
	uint32_t num;
	uint32_t da16_subnum;
	uint32_t da1_subnum;

	uint32_t swiz_x;
	uint32_t swiz_y;

	uint32_t type;
	uint32_t file;
};

struct inst_send {
	uint32_t sfid;
	uint32_t function_control;
	uint32_t header_present;
	uint32_t rlen;
	uint32_t mlen;
	uint32_t eot;
};

struct inst {
	uint64_t qw[2];
};

static inline bool
is_integer(int file, int type)
{
	if (file == BRW_IMMEDIATE_VALUE) {
		switch (type) {
		case BRW_HW_REG_TYPE_UD:
		case BRW_HW_REG_TYPE_D:
		case BRW_HW_REG_TYPE_UW:
		case BRW_HW_REG_TYPE_W:
		case GEN8_HW_REG_TYPE_UQ:
		case GEN8_HW_REG_TYPE_Q:
		case BRW_HW_REG_IMM_TYPE_UV:
		case BRW_HW_REG_IMM_TYPE_V:
			return true;
		case BRW_HW_REG_TYPE_F:
		case BRW_HW_REG_IMM_TYPE_VF:
		case GEN8_HW_REG_IMM_TYPE_DF:
		case GEN8_HW_REG_IMM_TYPE_HF:
			return false;
		}
	} else {
		switch (type) {
		case BRW_HW_REG_TYPE_UD:
		case BRW_HW_REG_TYPE_D:
		case BRW_HW_REG_TYPE_UW:
		case BRW_HW_REG_TYPE_W:
		case BRW_HW_REG_NON_IMM_TYPE_UB:
		case BRW_HW_REG_NON_IMM_TYPE_B:
		case GEN8_HW_REG_TYPE_UQ:
		case GEN8_HW_REG_TYPE_Q:
			return true;
		case BRW_HW_REG_TYPE_F:
		case GEN8_HW_REG_NON_IMM_TYPE_HF:
		case GEN7_HW_REG_NON_IMM_TYPE_DF:
			return false;
		}
	}

	ksim_unreachable("unknown type\n");

	return 0;
}

static inline bool
is_float(int file, int type)
{
	return !is_integer(file, type);
};

static int
type_size(int type)
{
	switch (type) {
	case BRW_HW_REG_TYPE_UD:
	case BRW_HW_REG_TYPE_D:
	case BRW_HW_REG_TYPE_F:
		return 4;
	case BRW_HW_REG_TYPE_UW:
	case BRW_HW_REG_TYPE_W:
	case GEN8_HW_REG_NON_IMM_TYPE_HF:
		return 2;
	case BRW_HW_REG_NON_IMM_TYPE_UB:
	case BRW_HW_REG_NON_IMM_TYPE_B:
		return 1;
	case GEN7_HW_REG_NON_IMM_TYPE_DF:
	case GEN8_HW_REG_TYPE_UQ:
	case GEN8_HW_REG_TYPE_Q:
		return 8;
	default:
		return -1; /* ksim_assert */
	}
}

static uint32_t
_3src_type_to_type(uint32_t _3src_type)
{
	switch (_3src_type) {
	case BRW_3SRC_TYPE_F:	return BRW_HW_REG_TYPE_F;
	case BRW_3SRC_TYPE_D:	return BRW_HW_REG_TYPE_D;
	case BRW_3SRC_TYPE_UD:	return BRW_HW_REG_TYPE_UD;
	case BRW_3SRC_TYPE_DF:	return GEN7_HW_REG_NON_IMM_TYPE_DF;
	default:		assert(0);
	}
}

static inline uint32_t
get_inst_bits(struct inst *inst, int start, int end)
{
	uint32_t mask;

	assert(end + 1 - start < 64);
	mask = ~0U >> (31 - end + start);
	if (start < 64) {
		return (inst->qw[0] >> start) & mask;
	} else {
		return (inst->qw[1] >> (start - 64)) & mask;
	}
}

static inline struct inst_common
unpack_inst_common(struct inst *packed)
{
	return (struct inst_common) {
		.opcode                   = get_inst_bits(packed,  0,   6),
		/* EU_INSTRUCTION_CONTROLS_A */
		.access_mode              = get_inst_bits(packed,  8,   8),
		.no_dd_clear              = get_inst_bits(packed,  9,   9),
		.no_dd_check              = get_inst_bits(packed, 10,  10),
		.nib_control              = get_inst_bits(packed, 11,  11),
		.qtr_control              = get_inst_bits(packed, 12,  13),
		.thread_control           = get_inst_bits(packed, 14,  15),
		.pred_control             = get_inst_bits(packed, 16,  19),
		.pred_inv                 = get_inst_bits(packed, 20,  20),
		.exec_size                = get_inst_bits(packed, 21,  23),
		.math_function            = get_inst_bits(packed, 24,  27),
		.cond_modifier            = get_inst_bits(packed, 24,  27),
		.acc_wr_control           = get_inst_bits(packed, 28,  28),
		.branch_control           = get_inst_bits(packed, 28,  28),
		.cmpt_control             = get_inst_bits(packed, 29,  29),
		.debug_control            = get_inst_bits(packed, 30,  30),
		.saturate                 = get_inst_bits(packed, 31,  31),
		.flag_subreg_nr           = get_inst_bits(packed, 32,  32),
		.flag_nr                  = get_inst_bits(packed, 32,  32),
		.mask_control             = get_inst_bits(packed, 34,  34)
	};
}

static inline struct inst_send
unpack_inst_send(struct inst *packed)
{
	return (struct inst_send) {
		.sfid                     = get_inst_bits(packed,  24,  27),
		.function_control         = get_inst_bits(packed,  96,  127),
		.header_present           = get_inst_bits(packed,  115,  115),
		.rlen                     = get_inst_bits(packed,  116,  120),
		.mlen                     = get_inst_bits(packed,  121,  124),
		.eot                      = get_inst_bits(packed,  127,  127),
	};
}

static inline struct inst_dst
unpack_inst_2src_dst(struct inst *packed)
{
	return (struct inst_dst) {
		.file                     = get_inst_bits(packed,   35,   36),
		.type                     = get_inst_bits(packed,   37,   40),
		.da1_subnum               = get_inst_bits(packed,   48,   52),
		.writemask                = get_inst_bits(packed,   48,   51),
		.da16_subnum              = get_inst_bits(packed,   52,   52),
		.num                      = get_inst_bits(packed,   53,   60),
		.ia_subnum                = get_inst_bits(packed,   57,   60),
		.hstride                  = get_inst_bits(packed,   61,   63),
		.address_mode             = get_inst_bits(packed,   63,   63),
	};
}

static inline struct inst_src
unpack_inst_2src_src0(struct inst *packed)
{
	return (struct inst_src) {
		.vstride                  = (1 << get_inst_bits(packed, 85, 88)) >> 1,
		.width                    = 1 << get_inst_bits(packed, 82, 84),
		.swiz_w                   = get_inst_bits(packed, 82, 83),
		.swiz_z                   = get_inst_bits(packed, 80, 81),
		.hstride                  = (1 << get_inst_bits(packed, 80, 81)) >> 1,
		.address_mode             = get_inst_bits(packed, 79, 79),
		.negate                   = get_inst_bits(packed, 78, 78),
		.abs                      = get_inst_bits(packed, 77, 77),
		.ia_subnum                = get_inst_bits(packed, 73, 76),
		.num                      = get_inst_bits(packed, 69, 76),
		.da16_subnum              = get_inst_bits(packed, 68, 68),
		.da1_subnum               = get_inst_bits(packed, 64, 68),
		.swiz_x                   = get_inst_bits(packed, 66, 67),
		.swiz_y                   = get_inst_bits(packed, 64, 65),
		.type                     = get_inst_bits(packed, 43, 46),
		.file                     = get_inst_bits(packed, 41, 42)
	};
}

static inline struct inst_src
unpack_inst_2src_src1(struct inst *packed)
{
	return (struct inst_src) {
		.file                     = get_inst_bits(packed,  89,  90),
		.type                     = get_inst_bits(packed,  91,  94),
		.da1_subnum               = get_inst_bits(packed,  96, 100),
		.da16_subnum              = get_inst_bits(packed, 100, 100),
		.num                      = get_inst_bits(packed, 101, 108),
		.ia_subnum                = get_inst_bits(packed, 105, 108),
		.abs                      = get_inst_bits(packed, 109, 109),
		.negate                   = get_inst_bits(packed, 110, 110),
		.address_mode             = get_inst_bits(packed, 111, 111),
		.hstride                  = (1 << get_inst_bits(packed, 112, 113)) >> 1,
		.swiz_z                   = get_inst_bits(packed, 112, 113),
		.swiz_w                   = get_inst_bits(packed, 114, 115),
		.width                    = 1 << get_inst_bits(packed, 114, 116),
		.vstride                  = (1 << get_inst_bits(packed, 117, 120)) >> 1
	};
}

static inline struct inst_dst
unpack_inst_3src_dst(struct inst *packed)
{
	uint32_t type = _3src_type_to_type(get_inst_bits(packed, 46, 48));

	return (struct inst_dst) {
		.file		= BRW_GENERAL_REGISTER_FILE,
		.type		= type,
		.da1_subnum	= 0,
		.writemask	= get_inst_bits(packed,   49,   52),
		.da16_subnum	= get_inst_bits(packed,   53,   55) * type_size(type),
		.num		= get_inst_bits(packed,   56,   63),
		.ia_subnum	= 0,
		.hstride	= 1,
		.address_mode	= BRW_ADDRESS_DIRECT
	};
}

static inline struct inst_src
unpack_inst_3src_src0(struct inst *packed)
{
	uint32_t type = _3src_type_to_type(get_inst_bits(packed,  43,  45));

	return (struct inst_src) {
		.file                     = BRW_GENERAL_REGISTER_FILE,
		.type                     = type,
		.abs                      = get_inst_bits(packed,  37,  37),
		.negate                   = get_inst_bits(packed,  38,  38),
		.hstride                  = get_inst_bits(packed,  64,  64) ? 0 : 1,
		.width                    = get_inst_bits(packed,  64,  64) ? 1 : 4,
		.vstride                  = get_inst_bits(packed,  64,  64) ? 0 : 4,
		.swiz_x                   = get_inst_bits(packed,  65,  66),
		.swiz_y                   = get_inst_bits(packed,  67,  68),
		.swiz_z                   = get_inst_bits(packed,  69,  70),
		.swiz_w                   = get_inst_bits(packed,  71,  72),
		.da16_subnum              = get_inst_bits(packed,  73,  75) * type_size(type),
		.num                      = get_inst_bits(packed,  76,  83),
		.da1_subnum               = 0,
		.ia_subnum                = 0,
		.address_mode             = BRW_ADDRESS_DIRECT,
	};
}

static inline struct inst_src
unpack_inst_3src_src1(struct inst *packed)
{
	uint32_t type = _3src_type_to_type(get_inst_bits(packed,  43,  45));

	return (struct inst_src) {
		.file                     = BRW_GENERAL_REGISTER_FILE,
		.type                     = type,
		.abs                      = get_inst_bits(packed,  39,  39),
		.negate                   = get_inst_bits(packed,  40,  40),
		.hstride                  = get_inst_bits(packed,  85,  85) ? 0 : 1,
		.width                    = get_inst_bits(packed,  85,  85) ? 1 : 4,
		.vstride                  = get_inst_bits(packed,  85,  85) ? 0 : 4,
		.swiz_x                   = get_inst_bits(packed,  86,  87),
		.swiz_y                   = get_inst_bits(packed,  88,  89),
		.swiz_z                   = get_inst_bits(packed,  90,  91),
		.swiz_w                   = get_inst_bits(packed,  92,  93),
		.da16_subnum              = get_inst_bits(packed,  94,  96) * type_size(type),
		.num                      = get_inst_bits(packed,  97, 104),
		.da1_subnum               = 0,
		.ia_subnum                = 0,
		.address_mode             = BRW_ADDRESS_DIRECT,
	};
}

static inline struct inst_src
unpack_inst_3src_src2(struct inst *packed)
{
	uint32_t type = _3src_type_to_type(get_inst_bits(packed,  43,  45));

	return (struct inst_src) {
		.file                     = BRW_GENERAL_REGISTER_FILE,
		.type                     = type,
		.abs                      = get_inst_bits(packed,  41,  41),
		.negate                   = get_inst_bits(packed,  42,  42),
		.hstride                  = get_inst_bits(packed, 106, 106) ? 0 : 1,
		.width                    = get_inst_bits(packed, 106, 106) ? 1 : 4,
		.vstride                  = get_inst_bits(packed, 106, 106) ? 0 : 4,
		.swiz_x                   = get_inst_bits(packed, 107, 108),
		.swiz_y                   = get_inst_bits(packed, 109, 110),
		.swiz_z                   = get_inst_bits(packed, 111, 112),
		.swiz_w                   = get_inst_bits(packed, 113, 114),
		.da16_subnum              = get_inst_bits(packed, 115, 117) * type_size(type),
		.num                      = get_inst_bits(packed, 118, 125),
		.da1_subnum               = 0,
		.ia_subnum                = 0,
		.address_mode             = BRW_ADDRESS_DIRECT,
	};
}

struct inst_imm {
	int32_t d;
	uint32_t ud;
	float f;
	float vf[4];
	uint16_t uv[8];
	int16_t v[8];
};

static inline float
u32_to_float(uint32_t ud)
{
	return ((union { float f; uint32_t ud; }) { .ud = ud }).f;
}

static inline uint32_t
float_to_u32(float f)
{
	return ((union { float f; uint32_t ud; }) { .f = f }).ud;
}

static inline float
vf_to_float(unsigned char vf)
{
	/* ±0.0f is special cased. */
	if (vf == 0x00 || vf == 0x80)
		return u32_to_float(vf << 24);
	else
		return u32_to_float(((vf & 0x80) << 24) | ((vf & 0x7f) << (23 - 4)));
}

static inline struct inst_imm
unpack_inst_imm(struct inst *packed)
{
	return (struct inst_imm) {
		.d	= get_inst_bits(packed,  96,  127),
		.ud	= get_inst_bits(packed,  96,  127),
		.f	= u32_to_float(get_inst_bits(packed,  96,  127)),
		.vf = {
			vf_to_float(get_inst_bits(packed,   96,  103)),
			vf_to_float(get_inst_bits(packed,  104,  111)),
			vf_to_float(get_inst_bits(packed,  112,  119)),
			vf_to_float(get_inst_bits(packed,  120,  127)),
		},
		.uv = {
			get_inst_bits(packed,  96,  99),
			get_inst_bits(packed, 100, 103),
			get_inst_bits(packed, 104, 107),
			get_inst_bits(packed, 108, 111),
			get_inst_bits(packed, 112, 115),
			get_inst_bits(packed, 116, 119),
			get_inst_bits(packed, 120, 123),
			get_inst_bits(packed, 124, 127)
		},
		.v = { /* FIXME: Sign extend */
			get_inst_bits(packed,  96,  99),
			get_inst_bits(packed, 100, 103),
			get_inst_bits(packed, 104, 107),
			get_inst_bits(packed, 108, 111),
			get_inst_bits(packed, 112, 115),
			get_inst_bits(packed, 116, 119),
			get_inst_bits(packed, 120, 123),
			get_inst_bits(packed, 124, 127)
		}
	};
}

static inline bool
is_logic_instruction(struct inst *inst)
{
	switch (unpack_inst_common(inst).opcode) {
	case BRW_OPCODE_AND:
	case BRW_OPCODE_NOT:
	case BRW_OPCODE_OR:
	case BRW_OPCODE_XOR:
		return true;
	default:
		return false;
	}
}
