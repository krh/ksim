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

#include "brw_context.h"
#include "brw_defines.h"
#include "brw_reg.h"
#include "brw_inst.h"
#include "brw_eu.h"

#include "gen_disasm.h"

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
store_type(struct thread *t, struct reg *r, int channel, int type, int offset)
{
   void *address = ((void *) t->grf + offset);
   switch (type) {
   case BRW_HW_REG_TYPE_UD:
   case BRW_HW_REG_TYPE_D:
   case BRW_HW_REG_TYPE_F:
      *(uint32_t *) address = r->ud[channel];
      break;
   case BRW_HW_REG_TYPE_UW:
   case BRW_HW_REG_TYPE_W:
   case GEN8_HW_REG_NON_IMM_TYPE_HF:
   *(uint16_t *) address = r->ud[channel];
      break;
   case BRW_HW_REG_NON_IMM_TYPE_UB:
   case BRW_HW_REG_NON_IMM_TYPE_B:
       *(uint8_t *) address = r->ud[channel];
      break;
   case GEN7_HW_REG_NON_IMM_TYPE_DF:
   case GEN8_HW_REG_TYPE_UQ:
   case GEN8_HW_REG_TYPE_Q:
      *(uint64_t *) address = r->ud[channel];
      break;
   }
}

static void
load_type(struct thread *t, struct reg *r, int channel, int type, int offset)
{
   void *address = ((void *) t->grf + offset);
   switch (type) {
   case BRW_HW_REG_TYPE_UD:
   case BRW_HW_REG_TYPE_D:
   case BRW_HW_REG_TYPE_F:
      r->ud[channel] = *(uint32_t *) address;
      break;
   case BRW_HW_REG_TYPE_UW:
   case BRW_HW_REG_TYPE_W:
   case GEN8_HW_REG_NON_IMM_TYPE_HF:
      r->uw[channel] = *(uint16_t *) address;
      break;
   case BRW_HW_REG_NON_IMM_TYPE_UB:
   case BRW_HW_REG_NON_IMM_TYPE_B:
      r->ub[channel] = *(uint8_t *) address;
      break;
   case GEN7_HW_REG_NON_IMM_TYPE_DF:
   case GEN8_HW_REG_TYPE_UQ:
   case GEN8_HW_REG_TYPE_Q:
      r->uq[channel] = *(uint64_t *) address;
      break;
   }
}

static int
load_imm(const struct brw_device_info *devinfo, struct reg *reg, unsigned type, brw_inst *inst)
{
   int exec_size = 1 << brw_inst_exec_size(devinfo, inst);

   switch (type) {
   case BRW_HW_REG_TYPE_UD:
   case BRW_HW_REG_TYPE_D:
   case BRW_HW_REG_IMM_TYPE_UV:
   case BRW_HW_REG_TYPE_F:
      for (int i = 0; i < exec_size; i++)
         reg->ud[i] = brw_inst_imm_ud(devinfo, inst);
      break;
   case BRW_HW_REG_TYPE_UW:
   case BRW_HW_REG_TYPE_W:
      for (int i = 0; i < exec_size; i++)
         reg->uw[0] = brw_inst_imm_ud(devinfo, inst);
      break;
   case BRW_HW_REG_IMM_TYPE_VF:
      for (int i = 0; i < exec_size; i += 4) {
         reg->f[i + 0] = brw_vf_to_float(brw_inst_imm_ud(devinfo, inst));
         reg->f[i + 1] = brw_vf_to_float(brw_inst_imm_ud(devinfo, inst) >> 8);
         reg->f[i + 2] = brw_vf_to_float(brw_inst_imm_ud(devinfo, inst) >> 16);
         reg->f[i + 3] = brw_vf_to_float(brw_inst_imm_ud(devinfo, inst) >> 24);
      }
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
         struct reg *r, brw_inst *inst,
         unsigned type, unsigned _reg_file, unsigned _reg_nr,
         unsigned sub_reg_num, unsigned _vert_stride, unsigned _width, unsigned _horiz_stride)
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
      int height = exec_size / (1 << _width);
      int row = _reg_nr * 32 + sub_reg_num * size;
      int address[16];
      int channel = 0;
      int offset;
      for (int i = 0; i < height; i++) {
         offset = row;
         row += ((1 << _vert_stride) >> 1) * size;
         for (int j = 0; j < (1 << _width); j++) {
            address[channel++] = offset;
            offset += ((1 << _horiz_stride) >> 1) * size;
         }
      }
         
      for (int i = 0; i < exec_size; i++)
         load_type(t, r, i, type, address[i]);
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
           struct reg *r, unsigned type, unsigned opcode, unsigned _negate, unsigned __abs)
{
   if (__abs) {
      if (type == BRW_HW_REG_TYPE_F) {
         for (int i = 0; i < 8; i++)
            r->f[i] = fabsf(r->f[i]);
      } else {
         for (int i = 0; i < 8; i++)
            r->d[i] = abs(r->d[i]);
      }
   }

   if (_negate) {
      if (devinfo->gen >= 8 && is_logic_instruction(opcode)) {
         for (int i = 0; i < 8; i++)
            r->ud[i] = ~r->ud[i];
      } else if (type == BRW_HW_REG_TYPE_F) {
         for (int i = 0; i < 8; i++)
            r->f[i] = -r->f[i];
      } else {
         for (int i = 0; i < 8; i++)
            r->d[i] = -r->d[i];
      }
   }
}

static int
load_src_da1(const struct brw_device_info *devinfo,
             struct thread *t,
             struct reg *r, brw_inst *inst,
             unsigned type, unsigned _reg_file,
             unsigned _vert_stride, unsigned _width, unsigned _horiz_stride,
             unsigned reg_num, unsigned sub_reg_num, unsigned __abs,
             unsigned _negate)
{
   load_reg(devinfo, t, r, inst, type,
            _reg_file, reg_num, sub_reg_num, _vert_stride, _width, _horiz_stride);

   apply_mods(devinfo, r, type, brw_inst_opcode(devinfo, inst), _negate, __abs);

   return 0;
}

static int
load_src_ia1(const struct brw_device_info *devinfo,
             struct reg *r,
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
              struct reg *r,
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
load_src0(const struct brw_device_info *devinfo, struct thread *t, struct reg *reg, brw_inst *inst)
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
store_reg(const struct brw_device_info *devinfo, struct thread *t,
          struct reg *r, brw_inst *inst,
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

      for (int i = 0; i < exec_size; i++) {
         store_type(t, r, i, type, offset);
         offset += ((1 << _horiz_stride) >> 1) * size;
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
          struct thread *t, struct reg *r, brw_inst *inst)
{
   if (brw_inst_saturate(devinfo, inst) &&
       brw_inst_dst_reg_type(devinfo, inst) == BRW_HW_REG_TYPE_F) {
      for (int i = 0; i < 8; i++) {
         if (r->f[i] > 1.0f)
            r->f[i] = 1.0f;
         else if (r->f[i] < 0.0f)
            r->f[i] = 0.0f;
      }
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

int
brw_execute_inst(const struct brw_device_info *devinfo,
		 brw_inst *inst, bool is_compacted,
                 struct thread *t)
{
   const enum opcode opcode = brw_inst_opcode(devinfo, inst);

   struct reg dst, src[3];

   int exec_size = 1 << brw_inst_exec_size(devinfo, inst);

   switch ((unsigned) opcode) {
   case BRW_OPCODE_MOV:
      load_src0(devinfo, t, &src[0], inst);
      store_dst(devinfo, t, &src[0], inst);
      break;
   case BRW_OPCODE_SEL:
      break;
   case BRW_OPCODE_NOT:
      load_src0(devinfo, t, &src[0], inst);
      for (int i = 0; i < exec_size; i++)
         dst.ud[i] = ~src[0].ud[i];
      store_dst(devinfo, t, &dst, inst);
      break;
   case BRW_OPCODE_AND:
      load_src0(devinfo, t, &src[0], inst);
      load_src1(devinfo, t, &src[1], inst);
      for (int i = 0; i < exec_size; i++)
         dst.ud[i] = src[0].ud[i] & src[1].ud[i];
      store_dst(devinfo, t, &dst, inst);
      break;
   case BRW_OPCODE_OR:
      load_src0(devinfo, t, &src[0], inst);
      load_src1(devinfo, t, &src[1], inst);
      for (int i = 0; i < exec_size; i++)
         dst.ud[i] = src[0].ud[i] | src[1].ud[i];
      store_dst(devinfo, t, &dst, inst);
      break;
   case BRW_OPCODE_XOR:
      load_src0(devinfo, t, &src[0], inst);
      load_src1(devinfo, t, &src[1], inst);
      for (int i = 0; i < exec_size; i++)
         dst.ud[i] = src[0].ud[i] & src[1].ud[i];
      store_dst(devinfo, t, &dst, inst);
      break;
   case BRW_OPCODE_SHR:
      break;
   case BRW_OPCODE_SHL:
      break;
   case BRW_OPCODE_ASR:
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
      break;
   case BRW_OPCODE_SENDC:
      break;
   case BRW_OPCODE_MATH:
      break;
   case BRW_OPCODE_ADD:
      load_src0(devinfo, t, &src[0], inst);
      load_src1(devinfo, t, &src[1], inst);
      for (int i = 0; i < exec_size; i++)
         dst.f[i] = src[0].f[i] * src[1].f[i];
      store_dst(devinfo, t, &dst, inst);
      break;
   case BRW_OPCODE_MUL:
      load_src0(devinfo, t, &src[0], inst);
      load_src1(devinfo, t, &src[1], inst);
      for (int i = 0; i < exec_size; i++)
         dst.f[i] = src[0].f[i] * src[1].f[i];
      store_dst(devinfo, t, &dst, inst);
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
   case BRW_OPCODE_LINE:
      break;
   case BRW_OPCODE_PLN:
      break;
   case BRW_OPCODE_MAD:
      break;
   case BRW_OPCODE_LRP:
      break;
   case BRW_OPCODE_NENOP:
      break;
   case BRW_OPCODE_NOP:
      break;
   }

   return 0;
}
