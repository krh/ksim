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
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <stdarg.h>
#include <math.h>
#include <immintrin.h>

#include "brw_context.h"
#include "brw_defines.h"
#include "brw_reg.h"
#include "brw_inst.h"
#include "brw_eu.h"

#include "gen_disasm.h"

union alu_reg {
   __m256i d;
   __m256 f;
   uint32_t u32[8];
};

static int
type_size(int type)
{
   switch (type) {
   case BRW_HW_REG_TYPE_UD:
   case BRW_HW_REG_TYPE_D:
   case BRW_HW_REG_TYPE_F: return 4;
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

static void
store_type(struct thread *t, uint32_t v, int type, int offset)
{
   void *address = ((void *) t->grf + offset);
   switch (type) {
   case BRW_HW_REG_TYPE_UD:
   case BRW_HW_REG_TYPE_D:
   case BRW_HW_REG_TYPE_F:
      *(uint32_t *) address = v;
      break;
   case BRW_HW_REG_TYPE_UW:
   case BRW_HW_REG_TYPE_W:
   case GEN8_HW_REG_NON_IMM_TYPE_HF:
      *(uint16_t *) address = v;
      break;
   case BRW_HW_REG_NON_IMM_TYPE_UB:
   case BRW_HW_REG_NON_IMM_TYPE_B:
      *(uint8_t *) address = v;
      break;
   case GEN7_HW_REG_NON_IMM_TYPE_DF:
   case GEN8_HW_REG_TYPE_UQ:
   case GEN8_HW_REG_TYPE_Q:
      *(uint64_t *) address = v;
      break;
   }
}

static int
load_imm(const struct brw_device_info *devinfo,
         union alu_reg *reg, unsigned type, brw_inst *inst)
{
   switch (type) {
   case BRW_HW_REG_TYPE_UD:
   case BRW_HW_REG_TYPE_D:
   case BRW_HW_REG_IMM_TYPE_UV:
   case BRW_HW_REG_TYPE_F:
   case BRW_HW_REG_TYPE_UW:
   case BRW_HW_REG_TYPE_W:
      reg->d = _mm256_set1_epi32(brw_inst_imm_ud(devinfo, inst));
      break;
   case BRW_HW_REG_IMM_TYPE_VF:
      reg->f = _mm256_set_ps(
         brw_vf_to_float(brw_inst_imm_ud(devinfo, inst)),
         brw_vf_to_float(brw_inst_imm_ud(devinfo, inst) >> 8),
         brw_vf_to_float(brw_inst_imm_ud(devinfo, inst) >> 16),
         brw_vf_to_float(brw_inst_imm_ud(devinfo, inst) >> 24),
         brw_vf_to_float(brw_inst_imm_ud(devinfo, inst)),
         brw_vf_to_float(brw_inst_imm_ud(devinfo, inst) >> 8),
         brw_vf_to_float(brw_inst_imm_ud(devinfo, inst) >> 16),
         brw_vf_to_float(brw_inst_imm_ud(devinfo, inst) >> 24));
      break;
   case BRW_HW_REG_IMM_TYPE_V:
      /* FIXME: What is this? */
      break;
   case GEN8_HW_REG_IMM_TYPE_DF:
   case GEN8_HW_REG_IMM_TYPE_HF:
      /* FIXME: */
      break;
   }
   return 0;
}

static int
load_reg(const struct brw_device_info *devinfo, struct thread *t,
         union alu_reg *r, brw_inst *inst,
         unsigned type, unsigned _reg_file, unsigned _reg_nr,
         unsigned sub_reg_num, unsigned vstride, unsigned width, unsigned hstride)
{
   /* Clear the Compr4 instruction compression bit. */
   if (_reg_file == BRW_MESSAGE_REGISTER_FILE)
      _reg_nr &= ~(1 << 7);

   switch (_reg_file) {
   case BRW_ARCHITECTURE_REGISTER_FILE:
      switch (_reg_nr & 0xf0) {
      case BRW_ARF_NULL:
         return -1;
      case BRW_ARF_ADDRESS:
         break;
      case BRW_ARF_ACCUMULATOR:
         break;
      case BRW_ARF_FLAG:
         break;
      case BRW_ARF_MASK:
         break;
      case BRW_ARF_MASK_STACK:
         break;
      case BRW_ARF_STATE:
         break;
      case BRW_ARF_CONTROL:
         break;
      case BRW_ARF_NOTIFICATION_COUNT:
         break;
      case BRW_ARF_IP:
         return -1;
      case BRW_ARF_TDR:
         return -1;
      case BRW_ARF_TIMESTAMP:
         break;
      default:
         break;
      }
      break;
   case BRW_GENERAL_REGISTER_FILE: {
      assert(is_power_of_two(width));

      uint32_t shift = __builtin_ffs(width) - 1;
      __m256i base = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
      __m256i hoffsets = _mm256_and_si256(base, _mm256_set1_epi32(width - 1));
      __m256i voffsets = _mm256_srlv_epi32(base, _mm256_set1_epi32(shift));
      __m256i offsets = _mm256_add_epi32(
         _mm256_mullo_epi32(hoffsets, _mm256_set1_epi32(hstride)),
         _mm256_mullo_epi32(voffsets, _mm256_set1_epi32(vstride)));

      offsets = _mm256_mullo_epi32(offsets, _mm256_set1_epi32(type_size(type)));
      void *grf_base = (void *) t->grf + _reg_nr * 32 + sub_reg_num;
      r->d = _mm256_i32gather_epi32(grf_base, offsets, 1);

      /* FIXME: Mask out bits above type size? Shouldn't matter... */

      break;
   }
   case BRW_MESSAGE_REGISTER_FILE:
      break;
   case BRW_IMMEDIATE_VALUE:
      break;
   }

   return 0;
}

static bool
is_logic_instruction(unsigned opcode)
{
   return opcode == BRW_OPCODE_AND ||
          opcode == BRW_OPCODE_NOT ||
          opcode == BRW_OPCODE_OR ||
          opcode == BRW_OPCODE_XOR;
}

static void
apply_mods(const struct brw_device_info *devinfo,
           union alu_reg *r, unsigned type, unsigned opcode, unsigned _negate, unsigned __abs)
{
   if (__abs) {
      if (type == BRW_HW_REG_TYPE_F) {
         r->d = _mm256_and_si256(r->d, _mm256_set1_epi32(0x7fffffff));
      } else {
         r->d = _mm256_abs_epi32(r->d);
      }
   }

   if (_negate) {
      if (devinfo->gen >= 8 && is_logic_instruction(opcode)) {
         r->d = _mm256_xor_si256(_mm256_setzero_si256(), r->d);
      } else if (type == BRW_HW_REG_TYPE_F) {
         r->f = _mm256_sub_ps(_mm256_setzero_ps(), r->f);
      } else {
         r->d = _mm256_sub_epi32(_mm256_setzero_si256(), r->d);
      }
   }
}

static int
load_src_da1(const struct brw_device_info *devinfo,
             struct thread *t,
             union alu_reg *r, brw_inst *inst,
             unsigned type, unsigned _reg_file,
             unsigned _vert_stride, unsigned _width, unsigned _horiz_stride,
             unsigned reg_num, unsigned sub_reg_num, unsigned __abs,
             unsigned _negate)
{
   uint32_t vstride = (1 << _vert_stride) >> 1;
   uint32_t width = 1 << _width;
   uint32_t hstride = (1 << _horiz_stride) >> 1;

   load_reg(devinfo, t, r, inst, type,
            _reg_file, reg_num, sub_reg_num, vstride, width, hstride);

   apply_mods(devinfo, r, type, brw_inst_opcode(devinfo, inst), _negate, __abs);

   return 0;
}

static int
load_src_ia1(const struct brw_device_info *devinfo,
             union alu_reg *r,
             unsigned opcode,
             unsigned type,
             unsigned _reg_file,
             int _addr_imm,
             unsigned _addr_subreg_nr,
             unsigned _negate,
             unsigned __abs,
             unsigned _addr_mode,
             unsigned _horiz_stride, unsigned _width, unsigned _vert_stride)
{
   /* FIXME ia load */

   apply_mods(devinfo, r, type, opcode, _negate, __abs);

   return 0;
}

static int
load_src_da16(const struct brw_device_info *devinfo,
              union alu_reg *r,
              unsigned opcode,
              unsigned _reg_type,
              unsigned _reg_file,
              unsigned _vert_stride,
              unsigned _reg_nr,
              unsigned _subreg_nr,
              unsigned __abs,
              unsigned _negate,
              unsigned swz_x, unsigned swz_y, unsigned swz_z, unsigned swz_w)
{
   return 0;
}

static int
load_src0(const struct brw_device_info *devinfo,
          struct thread *t, union alu_reg *reg, brw_inst *inst)
{
   if (brw_inst_src0_reg_file(devinfo, inst) == BRW_IMMEDIATE_VALUE) {
      return load_imm(devinfo, reg, brw_inst_src0_reg_type(devinfo, inst), inst);
   } else if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
      if (brw_inst_src0_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         return load_src_da1(devinfo, t, reg, inst,
                             brw_inst_src0_reg_type(devinfo, inst),
                             brw_inst_src0_reg_file(devinfo, inst),
                             brw_inst_src0_vstride(devinfo, inst),
                             brw_inst_src0_width(devinfo, inst),
                             brw_inst_src0_hstride(devinfo, inst),
                             brw_inst_src0_da_reg_nr(devinfo, inst),
                             brw_inst_src0_da1_subreg_nr(devinfo, inst),
                             brw_inst_src0_abs(devinfo, inst),
                             brw_inst_src0_negate(devinfo, inst));
      } else {
         return load_src_ia1(devinfo, reg,
                             brw_inst_opcode(devinfo, inst),
                             brw_inst_src0_reg_type(devinfo, inst),
                             brw_inst_src0_reg_file(devinfo, inst),
                             brw_inst_src0_ia1_addr_imm(devinfo, inst),
                             brw_inst_src0_ia_subreg_nr(devinfo, inst),
                             brw_inst_src0_negate(devinfo, inst),
                             brw_inst_src0_abs(devinfo, inst),
                             brw_inst_src0_address_mode(devinfo, inst),
                             brw_inst_src0_hstride(devinfo, inst),
                             brw_inst_src0_width(devinfo, inst),
                             brw_inst_src0_vstride(devinfo, inst));
      }
   } else {
      if (brw_inst_src0_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         return load_src_da16(devinfo, reg,
                              brw_inst_opcode(devinfo, inst),
                              brw_inst_src0_reg_type(devinfo, inst),
                              brw_inst_src0_reg_file(devinfo, inst),
                              brw_inst_src0_vstride(devinfo, inst),
                              brw_inst_src0_da_reg_nr(devinfo, inst),
                              brw_inst_src0_da16_subreg_nr(devinfo, inst),
                              brw_inst_src0_abs(devinfo, inst),
                              brw_inst_src0_negate(devinfo, inst),
                              brw_inst_src0_da16_swiz_x(devinfo, inst),
                              brw_inst_src0_da16_swiz_y(devinfo, inst),
                              brw_inst_src0_da16_swiz_z(devinfo, inst),
                              brw_inst_src0_da16_swiz_w(devinfo, inst));
      } else {
         return 1;
      }
   }
}

static int
load_src1(const struct brw_device_info *devinfo,
          struct thread *t, union alu_reg *reg, brw_inst *inst)
{
   if (brw_inst_src1_reg_file(devinfo, inst) == BRW_IMMEDIATE_VALUE) {
      return load_imm(devinfo, reg, brw_inst_src1_reg_type(devinfo, inst), inst);
   } else if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
      if (brw_inst_src1_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         return load_src_da1(devinfo, t, reg, inst,
                             brw_inst_src1_reg_type(devinfo, inst),
                             brw_inst_src1_reg_file(devinfo, inst),
                             brw_inst_src1_vstride(devinfo, inst),
                             brw_inst_src1_width(devinfo, inst),
                             brw_inst_src1_hstride(devinfo, inst),
                             brw_inst_src1_da_reg_nr(devinfo, inst),
                             brw_inst_src1_da1_subreg_nr(devinfo, inst),
                             brw_inst_src1_abs(devinfo, inst),
                             brw_inst_src1_negate(devinfo, inst));
      } else {
         return load_src_ia1(devinfo, reg,
                             brw_inst_opcode(devinfo, inst),
                             brw_inst_src1_reg_type(devinfo, inst),
                             brw_inst_src1_reg_file(devinfo, inst),
                             brw_inst_src1_ia1_addr_imm(devinfo, inst),
                             brw_inst_src1_ia_subreg_nr(devinfo, inst),
                             brw_inst_src1_negate(devinfo, inst),
                             brw_inst_src1_abs(devinfo, inst),
                             brw_inst_src1_address_mode(devinfo, inst),
                             brw_inst_src1_hstride(devinfo, inst),
                             brw_inst_src1_width(devinfo, inst),
                             brw_inst_src1_vstride(devinfo, inst));
      }
   } else {
      if (brw_inst_src1_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         return load_src_da16(devinfo, reg,
                              brw_inst_opcode(devinfo, inst),
                              brw_inst_src1_reg_type(devinfo, inst),
                              brw_inst_src1_reg_file(devinfo, inst),
                              brw_inst_src1_vstride(devinfo, inst),
                              brw_inst_src1_da_reg_nr(devinfo, inst),
                              brw_inst_src1_da16_subreg_nr(devinfo, inst),
                              brw_inst_src1_abs(devinfo, inst),
                              brw_inst_src1_negate(devinfo, inst),
                              brw_inst_src1_da16_swiz_x(devinfo, inst),
                              brw_inst_src1_da16_swiz_y(devinfo, inst),
                              brw_inst_src1_da16_swiz_z(devinfo, inst),
                              brw_inst_src1_da16_swiz_w(devinfo, inst));
      } else {
         return 1;
      }
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

static void
load_src0_3src(const struct brw_device_info *devinfo,
               struct thread *t, union alu_reg *r, brw_inst *inst)
{
   uint32_t type = _3src_type_to_type(brw_inst_3src_src_type(devinfo, inst));
   uint32_t vstride, width, hstride;

   if (brw_inst_3src_src0_rep_ctrl(devinfo, inst)) {
      vstride = 0;
      width = 1;
      hstride = 0;
   } else {
      vstride = 4;
      width = 4;
      hstride = 1;
   }

   load_reg(devinfo, t, r, inst, type,
            BRW_GENERAL_REGISTER_FILE,
            brw_inst_3src_src0_reg_nr(devinfo, inst),
            brw_inst_3src_src0_subreg_nr(devinfo, inst) * 4,
            vstride, width, hstride);

   apply_mods(devinfo, r, type, brw_inst_opcode(devinfo, inst),
              brw_inst_3src_src0_negate(devinfo, inst),
              brw_inst_3src_src0_abs(devinfo, inst));

   /* brw_inst_3src_src0_swizzle(devinfo, inst)); */
}

static void
load_src1_3src(const struct brw_device_info *devinfo,
               struct thread *t, union alu_reg *r, brw_inst *inst)
{
   uint32_t type = _3src_type_to_type(brw_inst_3src_src_type(devinfo, inst));
   uint32_t vstride, width, hstride;

   if (brw_inst_3src_src1_rep_ctrl(devinfo, inst)) {
      vstride = 0;
      width = 1;
      hstride = 0;
   } else {
      vstride = 4;
      width = 4;
      hstride = 1;
   }

   load_reg(devinfo, t, r, inst, type,
            BRW_GENERAL_REGISTER_FILE,
            brw_inst_3src_src1_reg_nr(devinfo, inst),
            brw_inst_3src_src1_subreg_nr(devinfo, inst) * 4,
            vstride, width, hstride);

   apply_mods(devinfo, r, type, brw_inst_opcode(devinfo, inst),
              brw_inst_3src_src1_negate(devinfo, inst),
              brw_inst_3src_src1_abs(devinfo, inst));

   /* brw_inst_3src_src1_swizzle(devinfo, inst)); */
}

static void
load_src2_3src(const struct brw_device_info *devinfo,
               struct thread *t, union alu_reg *r, brw_inst *inst)
{
   uint32_t type = _3src_type_to_type(brw_inst_3src_src_type(devinfo, inst));
   uint32_t vstride, width, hstride;

   if (brw_inst_3src_src2_rep_ctrl(devinfo, inst)) {
      vstride = 0;
      width = 1;
      hstride = 0;
   } else {
      vstride = 4;
      width = 4;
      hstride = 1;
   }

   load_reg(devinfo, t, r, inst, type,
            BRW_GENERAL_REGISTER_FILE,
            brw_inst_3src_src2_reg_nr(devinfo, inst),
            brw_inst_3src_src2_subreg_nr(devinfo, inst) * 4,
            vstride, width, hstride);

   apply_mods(devinfo, r, type, brw_inst_opcode(devinfo, inst),
              brw_inst_3src_src2_negate(devinfo, inst),
              brw_inst_3src_src2_abs(devinfo, inst));

   /* brw_inst_3src_src2_swizzle(devinfo, inst)); */
}

static int
store_reg(const struct brw_device_info *devinfo, struct thread *t,
          union alu_reg *r, brw_inst *inst,
          unsigned type, unsigned _reg_file, unsigned _reg_nr,
          unsigned sub_reg_num, unsigned _horiz_stride)
{
   /* Clear the Compr4 instruction compression bit. */
   if (_reg_file == BRW_MESSAGE_REGISTER_FILE)
      _reg_nr &= ~(1 << 7);

   switch (_reg_file) {
   case BRW_ARCHITECTURE_REGISTER_FILE:
      switch (_reg_nr & 0xf0) {
      case BRW_ARF_NULL:
         return -1;
      case BRW_ARF_ADDRESS:
         break;
      case BRW_ARF_ACCUMULATOR:
         break;
      case BRW_ARF_FLAG:
         break;
      case BRW_ARF_MASK:
         break;
      case BRW_ARF_MASK_STACK:
         break;
      case BRW_ARF_STATE:
         break;
      case BRW_ARF_CONTROL:
         break;
      case BRW_ARF_NOTIFICATION_COUNT:
         break;
      case BRW_ARF_IP:
         return -1;
      case BRW_ARF_TDR:
         return -1;
      case BRW_ARF_TIMESTAMP:
         break;
      default:
         break;
      }
      break;
   case BRW_GENERAL_REGISTER_FILE: {
      int exec_size = 1 << brw_inst_exec_size(devinfo, inst);
      int size = type_size(type);
      int offset = _reg_nr * 32 + sub_reg_num * size;

      uint32_t out[8];
      _mm256_storeu_si256((void *) out, r->d);

      if (size == 4 && _horiz_stride == 1) {
         memcpy((void *) t->grf + offset, out, 4 * exec_size);
      } else {

         for (int i = 0; i < exec_size; i++) {
            store_type(t, out[i], type, offset);
            offset += ((1 << _horiz_stride) >> 1) * size;
         }
      }
      break;
   }
   case BRW_MESSAGE_REGISTER_FILE:
      break;
   case BRW_IMMEDIATE_VALUE:
      break;
   }

   return 0;
}


static int
store_dst(const struct brw_device_info *devinfo,
          struct thread *t, union alu_reg *r, brw_inst *inst)
{
   /* FIXME: write masks */

   if (brw_inst_saturate(devinfo, inst) &&
       brw_inst_dst_reg_type(devinfo, inst) == BRW_HW_REG_TYPE_F) {
      r->f = _mm256_max_ps(r->f, _mm256_set1_ps(1.0f));
      r->f = _mm256_max_ps(r->f, _mm256_set1_ps(0.0f));
   }

   if (brw_inst_access_mode(devinfo, inst) == BRW_ALIGN_1) {
      if (brw_inst_dst_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {

         store_reg(devinfo, t, r, inst,
                   brw_inst_dst_reg_type(devinfo, inst),
                   brw_inst_dst_reg_file(devinfo, inst),
                   brw_inst_dst_da_reg_nr(devinfo, inst),
                   brw_inst_dst_da1_subreg_nr(devinfo, inst),
                   brw_inst_dst_hstride(devinfo, inst));
      } else {
            /* FIXME: indirect align1 */
      }
   } else {
      if (brw_inst_dst_address_mode(devinfo, inst) == BRW_ADDRESS_DIRECT) {
         /* FIXME: align16 */
      } else {
         /* Indirect align16 address mode not supported */
      }
   }

   return 0;
}

static void
store_dst_3src(const struct brw_device_info *devinfo,
               struct thread *t, union alu_reg *r, brw_inst *inst)
{
   store_reg(devinfo, t, r, inst,
             brw_inst_3src_dst_type(devinfo, inst),
             BRW_GENERAL_REGISTER_FILE,
             brw_inst_3src_dst_reg_nr(devinfo, inst),
             brw_inst_3src_dst_subreg_nr(devinfo, inst),
             1);
}

void
sfid_urb_simd8_write(struct thread *t, int reg, int offset, int mlem);

static void
sfid_urb(const struct brw_device_info *devinfo,
         brw_inst *inst, struct thread *t)
{
   switch (brw_inst_urb_opcode(devinfo, inst)) {
   case 0: /* write HWord */
   case 1: /* write OWord */
   case 2: /* read HWord */
   case 3: /* read OWord */
   case 4: /* atomic mov */
   case 5: /* atomic inc */
   case 6: /* atomic add */
      break;
   case 7: /* SIMD8 write */
      sfid_urb_simd8_write(t,
                           brw_inst_src0_da_reg_nr(devinfo, inst),
                           brw_inst_urb_global_offset(devinfo, inst),
                           brw_inst_mlen(devinfo, inst));
      break;
   }
}

static const struct {
   int num_srcs;
   bool store_dst;
} opcode_info[] = {
   [BRW_OPCODE_MOV]             = { .num_srcs = 1, .store_dst = true },
   [BRW_OPCODE_SEL]             = { },
   [BRW_OPCODE_NOT]             = { .num_srcs = 1, .store_dst = true },
   [BRW_OPCODE_AND]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_OR]              = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_XOR]             = { .num_srcs = 2,. store_dst = true },
   [BRW_OPCODE_SHR]             = { },
   [BRW_OPCODE_SHL]             = { },
   [BRW_OPCODE_ASR]             = { },
   [BRW_OPCODE_CMP]             = { },
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
   [BRW_OPCODE_FRC]             = { },
   [BRW_OPCODE_RNDU]            = { },
   [BRW_OPCODE_RNDD]            = { },
   [BRW_OPCODE_RNDE]            = { },
   [BRW_OPCODE_RNDZ]            = { },
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
   [BRW_OPCODE_DP4]             = { },
   [BRW_OPCODE_DPH]             = { },
   [BRW_OPCODE_DP3]             = { },
   [BRW_OPCODE_DP2]             = { },
   [BRW_OPCODE_LINE]            = { },
   [BRW_OPCODE_PLN]             = { },
   [BRW_OPCODE_MAD]             = { .num_srcs = 3, .store_dst = true },
   [BRW_OPCODE_LRP]             = { .num_srcs = 3, .store_dst = true },
   [BRW_OPCODE_NENOP]           = { .num_srcs = 0, .store_dst = false },
   [BRW_OPCODE_NOP]             = { .num_srcs = 0, .store_dst = false },
};

static void
dump_reg(const char *name, union alu_reg reg, int type)
{
   printf("%s: ", name);
   if (is_float(type)) {
      for (int c = 0; c < 8; c++)
         printf("  %6.2f", reg.f[c]);
   } else {
      for (int c = 0; c < 8; c++)
         printf("  %6d", reg.u32[c]);
   }
   printf("\n");
}

int
brw_execute_inst(const struct brw_device_info *devinfo,
                 brw_inst *inst, bool is_compacted,
                 struct thread *t)
{
   const enum opcode opcode = brw_inst_opcode(devinfo, inst);

   union alu_reg dst, src0, src1, src2;

   switch (opcode_info[opcode].num_srcs) {
   case 3:
      load_src0_3src(devinfo, t, &src0, inst);
      load_src1_3src(devinfo, t, &src1, inst);
      load_src2_3src(devinfo, t, &src2, inst);

      dump_reg("src0", src0,
               _3src_type_to_type(brw_inst_3src_src_type(devinfo, inst)));
      dump_reg("src1", src1,
               _3src_type_to_type(brw_inst_3src_src_type(devinfo, inst)));
      dump_reg("src2", src2,
               _3src_type_to_type(brw_inst_3src_src_type(devinfo, inst)));;

      break;
   case 2:
      load_src1(devinfo, t, &src1, inst);
   case 1:
      load_src0(devinfo, t, &src0, inst);
      break;
   case 0:
      break;
   }

   if (opcode_info[opcode].num_srcs == 2) {
      dump_reg("src0", src0, brw_inst_src0_reg_type(devinfo, inst));
      dump_reg("src1", src1, brw_inst_src1_reg_type(devinfo, inst));
   } else {
      dump_reg("src0", src0, brw_inst_src0_reg_type(devinfo, inst));
   }

   switch ((unsigned) opcode) {
   case BRW_OPCODE_MOV:
      dst = src0;
      break;
   case BRW_OPCODE_SEL:
      break;
   case BRW_OPCODE_NOT:
      dst.d = _mm256_xor_si256(_mm256_setzero_si256(), src0.d);
      break;
   case BRW_OPCODE_AND:
      dst.d = _mm256_and_si256(src0.d, src1.d);
      break;
   case BRW_OPCODE_OR:
      dst.d = _mm256_or_si256(src0.d, src1.d);
      break;
   case BRW_OPCODE_XOR:
      dst.d = _mm256_xor_si256(src0.d, src1.d);
      break;
   case BRW_OPCODE_SHR:
      dst.d = _mm256_srlv_epi32(src0.d, src1.d);
      break;
   case BRW_OPCODE_SHL:
      dst.d = _mm256_sllv_epi32(src0.d, src1.d);
      break;
   case BRW_OPCODE_ASR:
      dst.d = _mm256_srav_epi32(src0.d, src1.d);
      break;
   case BRW_OPCODE_CMP:
      break;
   case BRW_OPCODE_CMPN:
      break;
   case BRW_OPCODE_CSEL:
      break;
   case BRW_OPCODE_F32TO16:
      break;
   case BRW_OPCODE_F16TO32:
      break;
   case BRW_OPCODE_BFREV:
      break;
   case BRW_OPCODE_BFE:
      break;
   case BRW_OPCODE_BFI1:
      break;
   case BRW_OPCODE_BFI2:
      break;
   case BRW_OPCODE_JMPI:
      break;
   case BRW_OPCODE_IF:
      break;
   case BRW_OPCODE_IFF:
      break;
   case BRW_OPCODE_ELSE:
      break;
   case BRW_OPCODE_ENDIF:
      break;
   case BRW_OPCODE_DO:
      break;
   case BRW_OPCODE_WHILE:
      break;
   case BRW_OPCODE_BREAK:
      break;
   case BRW_OPCODE_CONTINUE:
      break;
   case BRW_OPCODE_HALT:
      break;
   case BRW_OPCODE_MSAVE:
      break;
   case BRW_OPCODE_MRESTORE:
      break;
   case BRW_OPCODE_GOTO:
      break;
   case BRW_OPCODE_POP:
      break;
   case BRW_OPCODE_WAIT:
      break;
   case BRW_OPCODE_SEND:
   case BRW_OPCODE_SENDC:
      switch (brw_inst_sfid(devinfo, inst)) {
      case BRW_SFID_NULL:
      case BRW_SFID_MATH:
      case BRW_SFID_SAMPLER:
      case BRW_SFID_MESSAGE_GATEWAY:
         break;
      case BRW_SFID_URB:
         sfid_urb(devinfo, inst, t);
         break;
      case BRW_SFID_THREAD_SPAWNER:
      case GEN6_SFID_DATAPORT_SAMPLER_CACHE:
      case GEN6_SFID_DATAPORT_RENDER_CACHE:
      case GEN6_SFID_DATAPORT_CONSTANT_CACHE:
      case GEN7_SFID_DATAPORT_DATA_CACHE:
      case GEN7_SFID_PIXEL_INTERPOLATOR:
      case HSW_SFID_DATAPORT_DATA_CACHE_1:
      case HSW_SFID_CRE:
         break;
      }
      break;
   case BRW_OPCODE_MATH:
      switch (brw_inst_math_function(devinfo, inst)) {

      case BRW_MATH_FUNCTION_INV:
         break;
      case BRW_MATH_FUNCTION_LOG:
         break;
      case BRW_MATH_FUNCTION_EXP:
         break;
      case BRW_MATH_FUNCTION_SQRT:
         break;
      case BRW_MATH_FUNCTION_RSQ:
         dst.f = _mm256_rsqrt_ps(src0.f);
         break;
      case BRW_MATH_FUNCTION_SIN:
         break;
      case BRW_MATH_FUNCTION_COS:
         break;
      case BRW_MATH_FUNCTION_SINCOS:
         break;
      case BRW_MATH_FUNCTION_FDIV:
         break;
      case BRW_MATH_FUNCTION_POW:
         break;
      case BRW_MATH_FUNCTION_INT_DIV_QUOTIENT_AND_REMAINDER:
         break;
      case BRW_MATH_FUNCTION_INT_DIV_QUOTIENT:
         break;
      case BRW_MATH_FUNCTION_INT_DIV_REMAINDER:
         break;
      case GEN8_MATH_FUNCTION_INVM:
         break;
      case GEN8_MATH_FUNCTION_RSQRTM:
         break;
      default:
         printf("some math function\n");
         break;
      }
      break;
   case BRW_OPCODE_ADD:
      dst.d = _mm256_add_epi32(src0.d, src1.d);
      break;
   case BRW_OPCODE_MUL:
      dst.d = _mm256_mullo_epi32(src0.d, src1.d);
      break;
   case BRW_OPCODE_AVG:
      break;
   case BRW_OPCODE_FRC:
      break;
   case BRW_OPCODE_RNDU:
      break;
   case BRW_OPCODE_RNDD:
      break;
   case BRW_OPCODE_RNDE:
      break;
   case BRW_OPCODE_RNDZ:
      break;
   case BRW_OPCODE_MAC:
      break;
   case BRW_OPCODE_MACH:
      break;
   case BRW_OPCODE_LZD:
      break;
   case BRW_OPCODE_FBH:
      break;
   case BRW_OPCODE_FBL:
      break;
   case BRW_OPCODE_CBIT:
      break;
   case BRW_OPCODE_ADDC:
      break;
   case BRW_OPCODE_SUBB:
      break;
   case BRW_OPCODE_SAD2:
      break;
   case BRW_OPCODE_SADA2:
      break;
   case BRW_OPCODE_DP4:
      break;
   case BRW_OPCODE_DPH:
      break;
   case BRW_OPCODE_DP3:
      break;
   case BRW_OPCODE_DP2:
      break;
   case BRW_OPCODE_LINE: {
      __m256 p = _mm256_set1_ps(src0.f[0]);
      __m256 q = _mm256_set1_ps(src0.f[3]);

      dst.f = _mm256_add_ps(_mm256_mul_ps(src1.f, p), q);
      break;
   }
   case BRW_OPCODE_PLN: {
      __m256 p = _mm256_set1_ps(src0.f[0]);
      __m256 q = _mm256_set1_ps(src0.f[1]);
      __m256 r = _mm256_set1_ps(src0.f[3]);

      load_src_da1(devinfo, t, &src2, inst,
                   brw_inst_src1_reg_type(devinfo, inst),
                   brw_inst_src1_reg_file(devinfo, inst),
                   brw_inst_src1_vstride(devinfo, inst),
                   brw_inst_src1_width(devinfo, inst),
                   brw_inst_src1_hstride(devinfo, inst),
                   brw_inst_src1_da_reg_nr(devinfo, inst) + 1,
                   brw_inst_src1_da1_subreg_nr(devinfo, inst),
                   brw_inst_src1_abs(devinfo, inst),
                   brw_inst_src1_negate(devinfo, inst));

      dst.f = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(src1.f, p),
                                          _mm256_mul_ps(src2.f, q)), r);
      break;
   }
   case BRW_OPCODE_MAD:
      dst.f = _mm256_add_ps(_mm256_mul_ps(src0.f, src1.f), src2.f);
      break;
   case BRW_OPCODE_LRP:
      break;
   case BRW_OPCODE_NENOP:
      break;
   case BRW_OPCODE_NOP:
      break;
   }

   dump_reg("dst", dst, brw_inst_dst_reg_type(devinfo, inst));

   if (opcode_info[opcode].store_dst) {
      if (opcode_info[opcode].num_srcs == 3)
         store_dst_3src(devinfo, t, &dst, inst);
      else
         store_dst(devinfo, t, &dst, inst);
   }

   return 0;
}
