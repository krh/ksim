#ifndef BRW_CONTEXT_H
#define BRW_CONTEXT_H

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "brw_device_info.h"

#define PACKED
#define unreachable(x)
#define ARRAY_SIZE(a) (sizeof (a) / sizeof (a)[0])

static inline bool
is_power_of_two(unsigned value)
{
   return (value & (value - 1)) == 0;
}

#define ffs __builtin_ffs

#define PRINTFLIKE(f, a) __attribute__ ((format(__printf__, f, a)))

#define p_atomic_cmpxchg(v, old, _new) \
   __sync_val_compare_and_swap((v), (old), (_new))

typedef union { float f; int i; unsigned int u; } fi_type;

struct annotation;

struct brw_context {
	int gen;
	int is_g4x;
	int is_cherryview;
};

struct opcode_desc {
    char    *name;
    int	    nsrc;
    int	    ndst;
};

struct annotation {
   int offset;

   /* Pointers to the basic block in the CFG if the instruction group starts
    * or ends a basic block.
    */
   struct bblock_t *block_start;
   struct bblock_t *block_end;

   /* Annotation for the generated IR.  One of the two can be set. */
   const void *ir;
   const char *annotation;
};

extern const struct opcode_desc opcode_descs[128];

#define BRW_SWIZZLE4(a,b,c,d) (((a)<<0) | ((b)<<2) | ((c)<<4) | ((d)<<6))
#define BRW_GET_SWZ(swz, idx) (((swz) >> ((idx)*2)) & 0x3)

#define BRW_SWIZZLE_NOOP      BRW_SWIZZLE4(0,1,2,3)
#define BRW_SWIZZLE_XYZW      BRW_SWIZZLE4(0,1,2,3)
#define BRW_SWIZZLE_XXXX      BRW_SWIZZLE4(0,0,0,0)
#define BRW_SWIZZLE_YYYY      BRW_SWIZZLE4(1,1,1,1)
#define BRW_SWIZZLE_ZZZZ      BRW_SWIZZLE4(2,2,2,2)
#define BRW_SWIZZLE_WWWW      BRW_SWIZZLE4(3,3,3,3)
#define BRW_SWIZZLE_XYXY      BRW_SWIZZLE4(0,1,0,1)
#define BRW_SWIZZLE_YZXW      BRW_SWIZZLE4(1,2,0,3)
#define BRW_SWIZZLE_ZXYW      BRW_SWIZZLE4(2,0,1,3)
#define BRW_SWIZZLE_ZWZW      BRW_SWIZZLE4(2,3,2,3)

#define WRITEMASK_X     0x1
#define WRITEMASK_Y     0x2
#define WRITEMASK_XY    0x3
#define WRITEMASK_Z     0x4
#define WRITEMASK_XZ    0x5
#define WRITEMASK_YZ    0x6
#define WRITEMASK_XYZ   0x7
#define WRITEMASK_W     0x8
#define WRITEMASK_XW    0x9
#define WRITEMASK_YW    0xa
#define WRITEMASK_XYW   0xb
#define WRITEMASK_ZW    0xc
#define WRITEMASK_XZW   0xd
#define WRITEMASK_YZW   0xe
#define WRITEMASK_XYZW  0xf

extern uint64_t INTEL_DEBUG;

struct brw_inst;

void
brw_disassemble(struct brw_context *brw,
                void *assembly, int start, int end, FILE *out);

/*
 * Returns the floor form of binary logarithm for a 32-bit integer.
 */
static inline unsigned int
_mesa_logbase2(unsigned int n)
{
#if defined(__GNUC__) && \
   ((__GNUC__ * 100 + __GNUC_MINOR__) >= 304) /* gcc 3.4 or later */
   return (31 - __builtin_clz(n | 1));
#else
   unsigned int pos = 0;
   if (n >= 1<<16) { n >>= 16; pos += 16; }
   if (n >= 1<< 8) { n >>=  8; pos +=  8; }
   if (n >= 1<< 4) { n >>=  4; pos +=  4; }
   if (n >= 1<< 2) { n >>=  2; pos +=  2; }
   if (n >= 1<< 1) {           pos +=  1; }
   return pos;
#endif
}

#endif /* BRW_CONTEXT_H */
