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

#include "ksim.h"
#include "avx-builder.h"

#define BRW_HW_REG_TYPE_UD  0
#define BRW_HW_REG_TYPE_D   1
#define BRW_HW_REG_TYPE_UW  2
#define BRW_HW_REG_TYPE_W   3
#define BRW_HW_REG_TYPE_F   7
#define GEN8_HW_REG_TYPE_UQ 8
#define GEN8_HW_REG_TYPE_Q  9

#define BRW_HW_REG_NON_IMM_TYPE_UB  4
#define BRW_HW_REG_NON_IMM_TYPE_B   5
#define GEN7_HW_REG_NON_IMM_TYPE_DF 6
#define GEN8_HW_REG_NON_IMM_TYPE_HF 10

#define BRW_HW_REG_IMM_TYPE_UV  4 /* Gen6+ packed unsigned immediate vector */
#define BRW_HW_REG_IMM_TYPE_VF  5 /* packed float immediate vector */
#define BRW_HW_REG_IMM_TYPE_V   6 /* packed int imm. vector; uword dest only */
#define GEN8_HW_REG_IMM_TYPE_DF 10
#define GEN8_HW_REG_IMM_TYPE_HF 11

enum opcode {
   /* These are the actual hardware opcodes. */
   BRW_OPCODE_MOV =	1,
   BRW_OPCODE_SEL =	2,
   BRW_OPCODE_NOT =	4,
   BRW_OPCODE_AND =	5,
   BRW_OPCODE_OR =	6,
   BRW_OPCODE_XOR =	7,
   BRW_OPCODE_SHR =	8,
   BRW_OPCODE_SHL =	9,
   BRW_OPCODE_ASR =	12,
   BRW_OPCODE_CMP =	16,
   BRW_OPCODE_CMPN =	17,
   BRW_OPCODE_CSEL =	18,  /**< Gen8+ */
   BRW_OPCODE_F32TO16 = 19,  /**< Gen7 only */
   BRW_OPCODE_F16TO32 = 20,  /**< Gen7 only */
   BRW_OPCODE_BFREV =	23,  /**< Gen7+ */
   BRW_OPCODE_BFE =	24,  /**< Gen7+ */
   BRW_OPCODE_BFI1 =	25,  /**< Gen7+ */
   BRW_OPCODE_BFI2 =	26,  /**< Gen7+ */
   BRW_OPCODE_JMPI =	32,
   BRW_OPCODE_IF =	34,
   BRW_OPCODE_IFF =	35,  /**< Pre-Gen6 */
   BRW_OPCODE_ELSE =	36,
   BRW_OPCODE_ENDIF =	37,
   BRW_OPCODE_DO =	38,
   BRW_OPCODE_WHILE =	39,
   BRW_OPCODE_BREAK =	40,
   BRW_OPCODE_CONTINUE = 41,
   BRW_OPCODE_HALT =	42,
   BRW_OPCODE_MSAVE =	44,  /**< Pre-Gen6 */
   BRW_OPCODE_MRESTORE = 45, /**< Pre-Gen6 */
   BRW_OPCODE_PUSH =	46,  /**< Pre-Gen6 */
   BRW_OPCODE_GOTO =	46,  /**< Gen8+    */
   BRW_OPCODE_POP =	47,  /**< Pre-Gen6 */
   BRW_OPCODE_WAIT =	48,
   BRW_OPCODE_SEND =	49,
   BRW_OPCODE_SENDC =	50,
   BRW_OPCODE_MATH =	56,  /**< Gen6+ */
   BRW_OPCODE_ADD =	64,
   BRW_OPCODE_MUL =	65,
   BRW_OPCODE_AVG =	66,
   BRW_OPCODE_FRC =	67,
   BRW_OPCODE_RNDU =	68,
   BRW_OPCODE_RNDD =	69,
   BRW_OPCODE_RNDE =	70,
   BRW_OPCODE_RNDZ =	71,
   BRW_OPCODE_MAC =	72,
   BRW_OPCODE_MACH =	73,
   BRW_OPCODE_LZD =	74,
   BRW_OPCODE_FBH =	75,  /**< Gen7+ */
   BRW_OPCODE_FBL =	76,  /**< Gen7+ */
   BRW_OPCODE_CBIT =	77,  /**< Gen7+ */
   BRW_OPCODE_ADDC =	78,  /**< Gen7+ */
   BRW_OPCODE_SUBB =	79,  /**< Gen7+ */
   BRW_OPCODE_SAD2 =	80,
   BRW_OPCODE_SADA2 =	81,
   BRW_OPCODE_DP4 =	84,
   BRW_OPCODE_DPH =	85,
   BRW_OPCODE_DP3 =	86,
   BRW_OPCODE_DP2 =	87,
   BRW_OPCODE_LINE =	89,
   BRW_OPCODE_PLN =	90,  /**< G45+ */
   BRW_OPCODE_MAD =	91,  /**< Gen6+ */
   BRW_OPCODE_LRP =	92,  /**< Gen6+ */
   BRW_OPCODE_NENOP =	125, /**< G45 only */
   BRW_OPCODE_NOP =	126,
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

#define BRW_MATH_FUNCTION_INV                              1
#define BRW_MATH_FUNCTION_LOG                              2
#define BRW_MATH_FUNCTION_EXP                              3
#define BRW_MATH_FUNCTION_SQRT                             4
#define BRW_MATH_FUNCTION_RSQ                              5
#define BRW_MATH_FUNCTION_SIN                              6
#define BRW_MATH_FUNCTION_COS                              7
#define BRW_MATH_FUNCTION_SINCOS                           8 /* gen4, gen5 */
#define BRW_MATH_FUNCTION_FDIV                             9 /* gen6+ */
#define BRW_MATH_FUNCTION_POW                              10
#define BRW_MATH_FUNCTION_INT_DIV_QUOTIENT_AND_REMAINDER   11
#define BRW_MATH_FUNCTION_INT_DIV_QUOTIENT                 12
#define BRW_MATH_FUNCTION_INT_DIV_REMAINDER                13
#define GEN8_MATH_FUNCTION_INVM                            14
#define GEN8_MATH_FUNCTION_RSQRTM                          15

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
   BRW_CONDITIONAL_NONE = 0,
   BRW_CONDITIONAL_Z    = 1,
   BRW_CONDITIONAL_NZ   = 2,
   BRW_CONDITIONAL_EQ   = 1,	/* Z */
   BRW_CONDITIONAL_NEQ  = 2,	/* NZ */
   BRW_CONDITIONAL_G    = 3,
   BRW_CONDITIONAL_GE   = 4,
   BRW_CONDITIONAL_L    = 5,
   BRW_CONDITIONAL_LE   = 6,
   BRW_CONDITIONAL_R    = 7,    /* Gen <= 5 */
   BRW_CONDITIONAL_O    = 8,
   BRW_CONDITIONAL_U    = 9,
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
   case BRW_3SRC_TYPE_F: return BRW_HW_REG_TYPE_F;
   case BRW_3SRC_TYPE_D: return BRW_HW_REG_TYPE_D;
   case BRW_3SRC_TYPE_UD: return BRW_HW_REG_TYPE_UD;
   case BRW_3SRC_TYPE_DF: return GEN7_HW_REG_NON_IMM_TYPE_DF;
   default: assert(0);
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
      .file                     = BRW_GENERAL_REGISTER_FILE,
      .type                     = type,
      .da1_subnum               = 0,
      .writemask                = get_inst_bits(packed,   49,   52),
      .da16_subnum              = get_inst_bits(packed,   53,   55) * type_size(type),
      .num                      = get_inst_bits(packed,   56,   63),
      .ia_subnum                = 0,
      .hstride                  = 1,
      .address_mode             = BRW_ADDRESS_DIRECT
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
      .d                        = get_inst_bits(packed,  96,  127),
      .ud                       = get_inst_bits(packed,  96,  127),
      .f                        = u32_to_float(get_inst_bits(packed,  96,  127)),
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

union alu_reg {
   __m256i d;
   __m256 f;
   uint32_t u32[8];
};

static bool
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

static const struct {
   int num_srcs;
   bool store_dst;
} opcode_info[] = {
   [BRW_OPCODE_MOV]             = { .num_srcs = 1, .store_dst = true },
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
		for (int y = 0; y < bld->exec_size / src->width; y++) {
			builder_emit_load_rsi(bld, offsetof(struct thread, grf[src->num].uw[subnum + y * src->vstride]));
			builder_emit_vpinsrq(bld, reg, reg, y);
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

static void
builder_emit_type_conversion(struct builder *bld, int reg, int dst_type, int src_type)
{
	switch (dst_type) {
	case BRW_HW_REG_TYPE_UD:
	case BRW_HW_REG_TYPE_D:
		if (src_type == BRW_HW_REG_TYPE_UD || src_type == BRW_HW_REG_TYPE_D)
			break;

		if (src_type == BRW_HW_REG_TYPE_UW)
			builder_emit_vpmovzxwd(bld, reg, reg);
		else if (src_type == BRW_HW_REG_TYPE_W)
			builder_emit_vpmovsxwd(bld, reg, reg);
		else if (src_type == BRW_HW_REG_TYPE_F)
			builder_emit_vcvtps2dq(bld, reg, reg);
		else
			stub("src type %d for ud/d dst type\n", src_type);
		break;

	case BRW_HW_REG_TYPE_UW:
	case BRW_HW_REG_TYPE_W: {
		int tmp_reg = builder_get_reg(bld);
		if (src_type == BRW_HW_REG_TYPE_UW || src_type == BRW_HW_REG_TYPE_W)
			break;

		if (src_type == BRW_HW_REG_TYPE_UD) {
			stub("not sure this pack is right");
			builder_emit_vextractf128(bld, tmp_reg, reg, 1);
			builder_emit_vpackusdw(bld, reg, tmp_reg, reg);
		} else if (src_type == BRW_HW_REG_TYPE_D) {
			stub("not sure this pack is right");
			builder_emit_vextractf128(bld, tmp_reg, reg, 1);
			builder_emit_vpackssdw(bld, reg, tmp_reg, reg);
		} else
			stub("src type %d for uw/w dst type\n", src_type);
		break;
	}
	case BRW_HW_REG_TYPE_F:
		if (src_type == BRW_HW_REG_TYPE_F)
			break;

		if (src_type == BRW_HW_REG_TYPE_UW) {
			builder_emit_vpmovzxwd(bld, reg, reg);
			builder_emit_vcvtdq2ps(bld, reg, reg);
		} else if (src_type == BRW_HW_REG_TYPE_W) {
			builder_emit_vpmovsxwd(bld, reg, reg);
			builder_emit_vcvtdq2ps(bld, reg, reg);
		} else if (src_type == BRW_HW_REG_TYPE_UD) {
			/* FIXME: Need to convert to int64 and then
			 * convert to floats as there is no uint32 to
			 * float cvt. */
			builder_emit_vcvtdq2ps(bld, reg, reg);
		} else if (src_type == BRW_HW_REG_TYPE_D) {
			builder_emit_vcvtdq2ps(bld, reg, reg);
		} else {
			stub("src type %d for float dst\n", src_type);
		}
		break;

	case GEN8_HW_REG_TYPE_UQ:
	case GEN8_HW_REG_TYPE_Q:
	default:
		stub("dst type\n", dst_type);
		break;
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
		reg = builder_get_reg(bld);
		switch (src->type) {
		case BRW_HW_REG_TYPE_UD:
		case BRW_HW_REG_TYPE_D:
			p = builder_get_const_ud(bld, unpack_inst_imm(inst).ud);
			builder_emit_vpbroadcastd_rip_relative(bld, reg, builder_offset(bld, p));
			break;

		case BRW_HW_REG_TYPE_UW:
		case BRW_HW_REG_TYPE_W:
			stub("unhandled imm type in src load");
			break;

		case BRW_HW_REG_IMM_TYPE_UV:
			/* Gen6+ packed unsigned immediate vector */
			p = builder_get_const_data(bld, 8 * 2, 16);
			memcpy(p, unpack_inst_imm(inst).uv, 8 * 2);
			builder_emit_vbroadcasti128_rip_relative(bld, reg, builder_offset(bld, p));
			src_type = BRW_HW_REG_TYPE_UW;
			break;

		case BRW_HW_REG_IMM_TYPE_VF:
			/* packed float immediate vector */
			stub("float imm vector");
			src_type = BRW_HW_REG_TYPE_F;
			break;

		case BRW_HW_REG_IMM_TYPE_V:
			/* packed int imm. vector; uword dest only */
			p = builder_get_const_data(bld, 8 * 2, 16);
			memcpy(p, unpack_inst_imm(inst).v, 8 * 2);
			builder_emit_vbroadcasti128_rip_relative(bld, reg, builder_offset(bld, p));
			src_type = BRW_HW_REG_TYPE_W;
			break;

		case BRW_HW_REG_TYPE_F:
			p = builder_get_const_ud(bld, unpack_inst_imm(inst).ud);
			builder_emit_vpbroadcastd_rip_relative(bld, reg, builder_offset(bld, p));
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

	builder_emit_type_conversion(bld, reg, dst.type, src_type);

	/* FIXME: Build the load above into the source modifier when possible, eg:
	 *
	 *     vpabsd 0x456(%rdi), %ymm1
	 */

	if (src->abs) {
		if (src->type == BRW_HW_REG_TYPE_F) {
			uint32_t *ud = builder_get_const_ud(bld, 0x7fffffff);
			int tmp_reg = builder_get_reg(bld);

			builder_emit_vpbroadcastd_rip_relative(bld, tmp_reg,
							       builder_offset(bld, ud));
			builder_emit_vpand(bld, reg, reg, tmp_reg);
		} else {
			builder_emit_vpabsd(bld, reg, reg);
		}
	}

	if (src->negate) {
		uint32_t *ud = builder_get_const_ud(bld, 0);
		int tmp_reg = builder_get_reg(bld);
		reg = builder_get_reg(bld);

		builder_emit_vpbroadcastd_rip_relative(bld, tmp_reg,
						       builder_offset(bld, ud));

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
		int tmp_reg = builder_get_reg(bld);
		ksim_assert(is_float(dst->file, dst->type));
		uint32_t *zero = builder_get_const_ud(bld, 0);
		builder_emit_vpbroadcastd_rip_relative(bld, tmp_reg,
						       builder_offset(bld, zero));

		builder_emit_vmaxps(bld, avx_reg, avx_reg, tmp_reg);


		uint32_t *one = builder_get_const_ud(bld, float_to_u32(1.0f));
		builder_emit_vpbroadcastd_rip_relative(bld, tmp_reg,
						       builder_offset(bld, one));
		builder_emit_vminps(bld, avx_reg, avx_reg, tmp_reg);
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

enum simd_mode {
	SIMD_MODE_SIMD8D_SIMD4x2,
	SIMD_MODE_SIMD8,
	SIMD_MODE_SIMD16,
	SIMD_MODE_SIMD32,
};

struct message_descriptor {
	uint32_t	binding_table_index;
	uint32_t	sampler_index;
	uint32_t	message_type;
	enum simd_mode	simd_mode;
	bool		header_present;
	uint32_t	response_length;
	uint32_t	message_length;
	uint32_t	return_format;
	bool		eot;
};

static inline struct message_descriptor
unpack_message_descriptor(uint32_t function_control)
{
	/* Vol 2d, "Message Descriptor - Sampling Engine" (p328) */
	return (struct message_descriptor) {
		.binding_table_index	= field(function_control,   0,    7),
		.sampler_index		= field(function_control,   8,   11),
		.message_type		= field(function_control,  12,   16),
		.simd_mode		= field(function_control,  17,   18),
		.header_present		= field(function_control,  19,   19),
		.response_length	= field(function_control,  20,   24),
		.message_length		= field(function_control,  25,   28),
		.return_format		= field(function_control,  30,   30),
		.eot			= field(function_control,  31,   31),
	};
}

enum simd_mode_extension {
	SIMD_MODE_EXTENSION_SIMD8D,
	SIMD_MODE_EXTENSION_SIMD4x2
};

struct message_header {
	uint32_t			r_offset;
	uint32_t			v_offset;
	uint32_t			u_offset;
	uint32_t			red_channel_mask;
	uint32_t			green_channel_mask;
	uint32_t			blue_channel_mask;
	uint32_t			alpha_channel_mask;
	uint32_t			gather4_source_channel_select;
	uint32_t			simd3264_output_format_control;
	enum simd_mode_extension	simd_mode_extension;
	uint32_t			pixel_null_mask_enable;
	uint32_t			render_target_index;
	uint32_t			sampler_state_pointer;
	uint32_t			destination_x_address;
	uint32_t			destination_y_address;
	uint32_t			output_format;
};

/* Vol 7, p 362 */
enum sample_message_type {
	SAMPLE_MESSAGE_SAMPLE		= 0b00000,
	SAMPLE_MESSAGE_SAMPLE_B		= 0b00001,
	SAMPLE_MESSAGE_SAMPLE_L		= 0b00010,
	SAMPLE_MESSAGE_SAMPLE_C		= 0b00011,
	SAMPLE_MESSAGE_SAMPLE_D		= 0b00100,
	SAMPLE_MESSAGE_SAMPLE_B_C	= 0b00101,
	SAMPLE_MESSAGE_SAMPLE_L_C	= 0b00110,
	SAMPLE_MESSAGE_LD		= 0b00111,
	SAMPLE_MESSAGE_GATHER4		= 0b01000,
	SAMPLE_MESSAGE_LOD		= 0b01001,
	SAMPLE_MESSAGE_RESINFO		= 0b01010,
	SAMPLE_MESSAGE_SAMPLEINFO	= 0b01011,
	SAMPLE_MESSAGE_GATHER4_C	= 0b10000,
	SAMPLE_MESSAGE_GATHER4_PO	= 0b10001,
	SAMPLE_MESSAGE_GATHER4_PO_C	= 0b10010,
	SAMPLE_MESSAGE_D_C		= 0b10100,
	SAMPLE_MESSAGE_MIN		= 0b10110,
	SAMPLE_MESSAGE_MAX		= 0b10111,
	SAMPLE_MESSAGE_LZ		= 0b11000,
	SAMPLE_MESSAGE_C_LZ		= 0b11001,
	SAMPLE_MESSAGE_LD_LZ		= 0b11010, /* Not in docs */
	SAMPLE_MESSAGE_LD2DMS_W		= 0b11100,
	SAMPLE_MESSAGE_LD_MCS		= 0b11101,
	SAMPLE_MESSAGE_LD2DMS		= 0b11110,
};

static inline struct message_header
unpack_message_header(const struct reg h)
{
	return (struct message_header) {
		.r_offset			= field(h.ud[2],  0,  3),
		.v_offset			= field(h.ud[2],  4,  7),
		.u_offset			= field(h.ud[2],  8, 11),
		.red_channel_mask		= field(h.ud[2], 12, 12),
		.green_channel_mask		= field(h.ud[2], 13, 13),
		.blue_channel_mask		= field(h.ud[2], 14, 14),
		.alpha_channel_mask		= field(h.ud[2], 15, 15),
		.gather4_source_channel_select	= field(h.ud[2], 16, 17),
		.simd3264_output_format_control	= field(h.ud[2], 18, 19),
		.simd_mode_extension		= field(h.ud[2], 22, 22),
		.pixel_null_mask_enable		= field(h.ud[2], 23, 23),
		.render_target_index		= field(h.ud[2], 24, 31),
		.sampler_state_pointer		= field(h.ud[3],  0, 31),
		.destination_x_address		= field(h.ud[4],  0, 15),
		.destination_y_address		= field(h.ud[4], 16, 31),
		.output_format			= field(h.ud[5],  0,  4),
	};
}

struct sfid_sampler_args {
	int src;
	int dst;
	int header;
	int rlen;
	struct surface tex;
};

static void
sfid_sampler_ld_simd4x2_linear(struct thread *t, const struct sfid_sampler_args *args)
{
	const struct message_header h = unpack_message_header(t->grf[args->header]);
	struct reg u, sample;
	uint32_t *p;
	/* Payload struct MAP32B_TS_SIMD4X2 */

	ksim_assert(h.simd_mode_extension == SIMD_MODE_EXTENSION_SIMD4x2);

	u.ireg = t->grf[args->src].ireg;
	switch (args->tex.format) {
	case SF_R32G32B32A32_FLOAT:
	case SF_R32G32B32A32_SINT:
	case SF_R32G32B32A32_UINT:
		p = args->tex.pixels + u.ud[0] * args->tex.stride;
		sample.ud[0] = p[0];
		sample.ud[1] = p[1];
		sample.ud[2] = p[2];
		sample.ud[3] = p[3];
		t->grf[args->dst] = sample;
		break;
	default:
		stub("unhandled ld format");
		break;
	}
}

static void
load_format_simd8(void *p, uint32_t format, __m256i offsets, __m256i emask, struct reg *dst)
{
	const __m256 zero = _mm256_set1_ps(0.0f);

	/* Grr, _mm256_mask_i32gather_epi32() is a macro that doesn't
	 * properly protect its arguments and casts the base pointer
	 * arg to an int pointer. We have to put () around where we
	 * add the offset. */

	switch (format) {
	case SF_R32G32B32A32_FLOAT:
	case SF_R32G32B32A32_SINT:
	case SF_R32G32B32A32_UINT:
		dst[0].ireg = _mm256_mask_i32gather_epi32(zero, (p + 0), offsets, emask, 1);
		dst[1].ireg = _mm256_mask_i32gather_epi32(zero, (p + 4), offsets, emask, 1);
		dst[2].ireg = _mm256_mask_i32gather_epi32(zero, (p + 8), offsets, emask, 1);
		dst[3].ireg = _mm256_mask_i32gather_epi32(zero, (p + 12), offsets, emask, 1);
		break;
	case SF_R16G16B16A16_UNORM: {
		const __m256i mask = _mm256_set1_epi32(0xffff);
		const __m256 scale = _mm256_set1_ps(1.0f / 65535.0f);
		struct reg rg, ba;

		rg.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rg.ireg, mask)), scale);
		rg.ireg = _mm256_srli_epi32(rg.ireg, 16);
		dst[1].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rg.ireg, mask)), scale);
		ba.ireg = _mm256_mask_i32gather_epi32(zero, (p + 4), offsets, emask, 1);
		dst[2].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(ba.ireg, mask)), scale);
		ba.ireg = _mm256_srli_epi32(ba.ireg, 16);
		dst[3].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(ba.ireg, mask)), scale);
		break;
	}

	case SF_R8G8B8X8_UNORM: {
		const __m256i mask = _mm256_set1_epi32(0xff);
		const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
		struct reg rgbx;

		rgbx.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgbx.ireg, mask)), scale);
		rgbx.ireg = _mm256_srli_epi32(rgbx.ireg, 8);
		dst[1].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgbx.ireg, mask)), scale);
		rgbx.ireg = _mm256_srli_epi32(rgbx.ireg, 8);
		dst[2].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgbx.ireg, mask)), scale);
		dst[3].reg = _mm256_set1_ps(1.0f);
		break;
	}

	case SF_R8G8B8A8_UNORM: {
		const __m256i mask = _mm256_set1_epi32(0xff);
		const __m256 scale = _mm256_set1_ps(1.0f / 255.0f);
		struct reg rgba;

		rgba.ireg = _mm256_mask_i32gather_epi32(zero, p, offsets, emask, 1);
		dst[0].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgba.ireg, mask)), scale);
		rgba.ireg = _mm256_srli_epi32(rgba.ireg, 8);
		dst[1].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgba.ireg, mask)), scale);
		rgba.ireg = _mm256_srli_epi32(rgba.ireg, 8);
		dst[2].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgba.ireg, mask)), scale);
		rgba.ireg = _mm256_srli_epi32(rgba.ireg, 8);
		dst[3].reg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(rgba.ireg, mask)), scale);
		break;
	}

	default:
		stub("sampler ld format %d", format);
		break;
	}
}

static void
sfid_sampler_ld_simd8_linear(struct thread *t, const struct sfid_sampler_args *args)
{
	struct reg u, v;

	u.ireg = t->grf[args->src].ireg;
	v.ireg = t->grf[args->src + 1].ireg;
	void *p = args->tex.pixels;

	struct reg offsets;
	offsets.ireg =
		_mm256_add_epi32(_mm256_mullo_epi32(u.ireg, _mm256_set1_epi32(args->tex.cpp)),
				 _mm256_mullo_epi32(v.ireg, _mm256_set1_epi32(args->tex.stride)));

	load_format_simd8(p, args->tex.format,
			  offsets.ireg, t->mask_full, &t->grf[args->dst]);
}

static void
sfid_sampler_sample_simd8_linear(struct thread *t, const struct sfid_sampler_args *args)
{
#if 0
	/* Clamp */
	struct reg u;
	u.reg = _mm256_min_ps(t->grf[args->src].reg, _mm256_set1_ps(1.0f));
	u.reg = _mm256_max_ps(u.reg, _mm256_setzero_ps());

	struct reg v;
	v.reg = _mm256_min_ps(t->grf[args->src + 1].reg, _mm256_set1_ps(1.0f));
	v.reg = _mm256_max_ps(v.reg, _mm256_setzero_ps());
#endif

	/* Wrap */
	struct reg u;
	u.reg = _mm256_floor_ps(t->grf[args->src].reg);
	u.reg = _mm256_sub_ps(t->grf[args->src].reg, u.reg);

	struct reg v;
	v.reg = _mm256_floor_ps(t->grf[args->src + 1].reg);
	v.reg = _mm256_sub_ps(t->grf[args->src + 1].reg, v.reg);

	u.reg = _mm256_mul_ps(u.reg, _mm256_set1_ps(args->tex.width - 1));
	v.reg = _mm256_mul_ps(v.reg, _mm256_set1_ps(args->tex.height - 1));

	u.ireg = _mm256_cvttps_epi32(u.reg);
	v.ireg = _mm256_cvttps_epi32(v.reg);

	struct reg offsets;
	offsets.ireg =
		_mm256_add_epi32(_mm256_mullo_epi32(u.ireg, _mm256_set1_epi32(args->tex.cpp)),
				 _mm256_mullo_epi32(v.ireg, _mm256_set1_epi32(args->tex.stride)));

	load_format_simd8(args->tex.pixels, args->tex.format,
			  offsets.ireg, t->mask_full, &t->grf[args->dst]);
}

static void *
builder_emit_sfid_render_cache_helper(struct builder *bld,
				      uint32_t opcode, uint32_t type, uint32_t src, uint32_t surface);

static void *
builder_emit_sfid_sampler(struct builder *bld, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	const int exec_size = 1 << unpack_inst_common(inst).exec_size;
	struct sfid_sampler_args *args;
	void *func;
	int num;

	const struct message_descriptor d =
		unpack_message_descriptor(send.function_control);

	args = builder_get_const_data(bld, sizeof *args, 8);

	args->dst = unpack_inst_2src_dst(inst).num;
	num = unpack_inst_2src_src0(inst).num;
	if (d.header_present)
		args->header = num++;
	else
		args->header = -1;
	args->src = num;

	bool tex_valid = get_surface(bld->binding_table_address,
				     d.binding_table_index, &args->tex);
	ksim_assert(tex_valid);
	switch (args->tex.format) {
	case SF_R32G32B32A32_FLOAT:
	case SF_R32G32B32A32_SINT:
	case SF_R32G32B32A32_UINT:
	case SF_R8G8B8X8_UNORM:
	case SF_R8G8B8A8_UNORM:
	case SF_R16G16B16A16_UNORM:
	case SF_R16G16B16A16_SNORM:
	case SF_R16G16B16A16_SINT:
	case SF_R16G16B16A16_UINT:
		break;
	default:
		stub("surface format: %d", args->tex.format);
		break;
	}

	builder_emit_load_rsi_rip_relative(bld, builder_offset(bld, args));

	switch (d.message_type) {
	case SAMPLE_MESSAGE_LD:
	case SAMPLE_MESSAGE_LD_LZ:
		if (d.simd_mode == SIMD_MODE_SIMD8D_SIMD4x2) {
			/* We only handle 4x2, which on SKL requires
			 * the simd mode extension bit in the header
			 * to be set. Assert we have a header. */
			ksim_assert(d.header_present);
			ksim_assert(exec_size == 4);
			func = sfid_sampler_ld_simd4x2_linear;
		} else if (d.simd_mode == SIMD_MODE_SIMD8) {
			func = sfid_sampler_ld_simd8_linear;
		} else {
			stub("unhandled ld simd mode");
		}
		break;
	default:
		func = sfid_sampler_sample_simd8_linear;
		break;
	}

	args->rlen = send.rlen;
	if (args->rlen == 0) {
		const uint32_t bti = 0; /* Should be M0.2 from header */
		const uint32_t opcode = 12;
		const uint32_t type = 4;

		/* dst is the null reg for rlen 0 messages, and so
		 * we'll end up overwriting grf0 - grf3.  We need the
		 * fragment x and y from grf1. so move it up. */
		args->dst = 2;
		builder_emit_call(bld, func);

		return builder_emit_sfid_render_cache_helper(bld, opcode, type, args->dst, bti);
	} else {
		return func;
	}
}

static void *
builder_emit_sfid_render_cache_helper(struct builder *bld,
				      uint32_t opcode, uint32_t type, uint32_t src, uint32_t surface)
{
	struct sfid_render_cache_args *args;
	bool rt_valid;

	args = builder_get_const_data(bld, sizeof *args, 32);
	args->src = src;

	rt_valid = get_surface(bld->binding_table_address, surface, &args->rt);
	ksim_assert(rt_valid);
	if (!rt_valid)
		return NULL;

	static const struct reg sx = { .d = {  0, 1, 0, 1, 2, 3, 2, 3 } };
	static const struct reg sy = { .d = {  0, 0, 1, 1, 0, 0, 1, 1 } };

	args->offsets.ireg =
		_mm256_add_epi32(_mm256_mullo_epi32(sx.ireg, _mm256_set1_epi32(args->rt.cpp)),
				 _mm256_mullo_epi32(sy.ireg, _mm256_set1_epi32(args->rt.stride)));

	builder_emit_load_rsi_rip_relative(bld, builder_offset(bld, args));

	/* vol 2d, p445 */
	switch (opcode) {
	case 12: /* rt write */
		switch (type) {
		case 0:
			return sfid_render_cache_rt_write_simd16;
		case 1:
			stub("rep16 rt write");
			return sfid_render_cache_rt_write_simd8_bgra_unorm8_xtiled;

		case 4: /* simd8 */
			if (args->rt.format == SF_R16G16B16A16_UNORM &&
			    args->rt.tile_mode == LINEAR)
				return sfid_render_cache_rt_write_simd8_rgba_unorm16_linear;
			else if (args->rt.format == SF_R8G8B8A8_UNORM &&
				 args->rt.tile_mode == LINEAR)
				return sfid_render_cache_rt_write_simd8_rgba_unorm8_linear;
			else if (args->rt.format == SF_B8G8R8A8_UNORM &&
				 args->rt.tile_mode == XMAJOR)
				return sfid_render_cache_rt_write_simd8_bgra_unorm8_xtiled;
			else if (args->rt.format == SF_B8G8R8X8_UNORM &&
				 args->rt.tile_mode == XMAJOR)
				return sfid_render_cache_rt_write_simd8_bgra_unorm8_xtiled;
			else if (args->rt.format == SF_B8G8R8A8_UNORM_SRGB &&
				 args->rt.tile_mode == XMAJOR)
				return sfid_render_cache_rt_write_simd8_bgra_unorm8_xtiled;
			else
				stub("simd8 rt write format/tile_mode: %d %d",
				     args->rt.format, args->rt.tile_mode);
		default:
			stub("rt write type %d", type);
			return NULL;
		}
	default:
		stub("render cache message opcode %d", opcode);
		return NULL;
	}
}

static void *
builder_emit_sfid_render_cache(struct builder *bld, struct inst *inst)
{
	struct inst_send send = unpack_inst_send(inst);
	uint32_t opcode = field(send.function_control, 14, 17);
	uint32_t type = field(send.function_control, 8, 10);
	uint32_t surface = field(send.function_control, 0, 7);
	uint32_t src = unpack_inst_2src_src0(inst).num;

	return builder_emit_sfid_render_cache_helper(bld, opcode, type, src, surface);
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
		dst_reg = src0_reg;
		break;
	case BRW_OPCODE_SEL: {
		int modifier = unpack_inst_common(inst).cond_modifier;
		int tmp_reg = builder_get_reg(bld);
		builder_emit_cmp(bld, modifier, tmp_reg, src0_reg, src1_reg);
		/* AVX2 blendv is opposite of the EU sel order, so we
		 * swap src0 and src1 operands. */
		builder_emit_vpblendvb(bld, dst_reg, tmp_reg, src1_reg, src0_reg);
		break;
	}
	case BRW_OPCODE_NOT: {
		uint32_t *ud = builder_get_const_ud(bld, 0);
		int tmp_reg = builder_get_reg(bld);
		builder_emit_vpbroadcastd_rip_relative(bld, tmp_reg,
						       builder_offset(bld, ud));
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
		int tmp0_reg = builder_get_reg(bld);
		int tmp1_reg = builder_get_reg(bld);

		src0_reg = builder_emit_src_load(bld, inst, &src1);
		builder_emit_vpbroadcastd(bld, tmp0_reg, reg_offset(src0.num, subnum));
		builder_emit_vpbroadcastd(bld, tmp1_reg, reg_offset(src0.num, subnum + 3));
		builder_emit_vfmadd132ps(bld, src0_reg, tmp0_reg, tmp1_reg);
		dst_reg = src0_reg;
		break;
	}
	case BRW_OPCODE_PLN: {
		src0 = unpack_inst_2src_src0(inst);
		src1 = unpack_inst_2src_src1(inst);
		ksim_assert(src0.type == BRW_HW_REG_TYPE_F);
		ksim_assert(src1.type == BRW_HW_REG_TYPE_F);
		int tmp0_reg = builder_get_reg(bld);
		int tmp1_reg = builder_get_reg(bld);

		src2 = unpack_inst_2src_src1(inst);
		src2.num++;

		int subnum = src0.da1_subnum / 4;
		src1_reg = builder_emit_src_load(bld, inst, &src1);
		builder_emit_vpbroadcastd(bld, tmp0_reg, reg_offset(src0.num, subnum));
		builder_emit_vpbroadcastd(bld, tmp1_reg, reg_offset(src0.num, subnum + 3));
		builder_emit_vfmadd132ps(bld, src1_reg, tmp0_reg, tmp1_reg);
		builder_emit_vpbroadcastd(bld, tmp0_reg, reg_offset(src0.num, subnum + 1));
		src0_reg = builder_emit_src_load(bld, inst, &src2);
		builder_emit_vfmadd132ps(bld, src0_reg, tmp0_reg, src1_reg);
		dst_reg = src0_reg;
		break;
	}
	case BRW_OPCODE_MAD:
		if (is_integer(dst.file, dst.type)) {
			builder_emit_vpmulld(bld, src1_reg, src1_reg, src2_reg);
			builder_emit_vpaddd(bld, dst_reg, src0_reg, src1_reg);
		} else {
			builder_emit_vfmadd231ps(bld, src0_reg, src2_reg, src1_reg);
			dst_reg = src0_reg;
		}
		break;
	case BRW_OPCODE_LRP:
		stub("BRW_OPCODE_LRP");
		break;
	case BRW_OPCODE_NENOP:
		stub("BRW_OPCODE_NENOP");
		break;
	case BRW_OPCODE_NOP:
		stub("BRW_OPCODE_NOP");
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

#include <common/gen_device_info.h>

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
