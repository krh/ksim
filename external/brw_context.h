#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#define PACKED

#include "gen_device_info.h"


typedef union { float f; int32_t i; uint32_t u; } fi_type;

struct brw_inst;
struct annotation { int offset; };
struct annotation_info { };

#define p_atomic_cmpxchg(v, old, _new) \
   __sync_val_compare_and_swap((v), (old), (_new))

#define PRINTFLIKE(a, b)
#define ARRAY_SIZE(x) (sizeof(x) / sizeof(*(x)))


int brw_disassemble_inst(FILE *file, const struct gen_device_info *devinfo,
                         struct brw_inst *inst, bool is_compacted);

#define unreachable(msg) __builtin_unreachable()

#define unlikely(cond) (cond)

static const uint32_t DEBUG_NO_COMPACTION = 0;
static const uint32_t DEBUG_HEX = 0;
static const uint32_t INTEL_DEBUG = 0;

#define WRITEMASK_X     0x1
#define WRITEMASK_XY    0x3
#define WRITEMASK_XYZ   0x7
#define WRITEMASK_XYW   0xb
#define WRITEMASK_XYZW  0xf


#define rzalloc_array(a, b, c) NULL
#define _mesa_is_pow_two(s) 0
