#include <bfd.h>
#include <dis-asm.h>

struct avx2_reg {
        struct list link;
};

struct builder {
	struct shader *shader;
	uint8_t *p;
	int pool_index;
	uint64_t binding_table_address;
	uint64_t sampler_state_address;

	struct avx2_reg regs[16];
	struct list regs_lru_list;
	struct list used_regs_list;

	/* Disassembly fields */
	struct disassemble_info info;
	int disasm_tail;
	char disasm_output[128];
	int disasm_length;
};

#define emit(bld, ...)							\
	do {								\
		uint8_t bytes[] = { __VA_ARGS__ };			\
		const uint32_t length = ARRAY_LENGTH(bytes);		\
		for (uint32_t i = 0; i < length; i++)			\
		        bld->p[i] = bytes[i];				\
		bld->p += length;					\
	} while (0)

#define emit_uint32(u)			\
	((u) & 0xff),			\
	(((u) >> 8) & 0xff),		\
	(((u) >> 16) & 0xff),		\
	(((u) >> 24) & 0xff)


static inline void
builder_emit_jmp_rip_relative(struct builder *bld, int32_t offset)
{
	emit(bld, 0xff, 0x25, emit_uint32(offset - 6));
}

static inline void
builder_emit_call_rip_relative(struct builder *bld, int32_t offset)
{
	emit(bld, 0xff, 0x15, emit_uint32(offset - 6));
}

static inline void
builder_emit_m256i_load(struct builder *bld, int dst, int32_t offset)
{
	emit(bld, 0xc5, 0xfd - (dst & 8) * 16, 0x6f, 0x87 + (dst & 7) * 0x08, emit_uint32(offset));
}

static inline void
builder_emit_m128i_load(struct builder *bld, int dst, int32_t offset)
{
	emit(bld, 0xc5, 0xf9 - (dst & 8) * 16, 0x6f, 0x87 + (dst & 7) * 8, emit_uint32(offset));
}

static inline void
builder_emit_m256i_load_rip_relative(struct builder *bld, int dst, int32_t offset)
{
	ksim_assert(dst < 16);

	emit(bld, 0xc5, 0xfd - (dst & 8) * 16, 0x6f, 0x05 + (dst & 7) * 8, emit_uint32(offset - 8));
}

static inline void
builder_emit_m256i_store(struct builder *bld, int src, int32_t offset)
{
	emit(bld, 0xc5, 0xfd - (src & 8) * 16, 0x7f, 0x87 + (src & 7) * 0x08, emit_uint32(offset));
}

static inline void
builder_emit_m128i_store(struct builder *bld, int src, int32_t offset)
{
	emit(bld, 0xc5, 0xf9 - (src & 8) * 16, 0x7f, 0x87 + (src & 7) * 0x08, emit_uint32(offset));
}

static inline void
builder_emit_u32_store(struct builder *bld, int src, int32_t offset)
{
	emit(bld, 0x66, 0x0f, 0x7e, 0x87 + src * 0x08, emit_uint32(offset));
}

static inline void
builder_emit_vpbroadcastd(struct builder *bld, int dst, int32_t offset)
{
	emit(bld, 0xc4, 0xe2 - (dst & 8) * 16, 0x7d, 0x58, 0x87 + (dst & 7) * 8,
	     emit_uint32(offset));
}

static inline void
builder_emit_vpbroadcastd_rip_relative(struct builder *bld, int dst, int32_t offset)
{
	emit(bld, 0xc4, 0xe2 - (dst & 8) * 16, 0x7d, 0x58, 0x05 + (dst & 7) * 8,
	     emit_uint32(offset - 9));
}

static inline void
builder_emit_load_rsi_rip_relative(struct builder *bld, int offset)
{
	emit(bld, 0x48, 0x8d, 0x35, emit_uint32(offset - 7));
}

static inline void
builder_emit_long_alu(struct builder *bld, int opcode0, int opcode1, int dst, int src0, int src1)
{
	ksim_assert(dst < 16 && src0 < 16 && src1 < 16);

	if (src0 < 8)
		emit(bld, 0xc5, (0xf0 | opcode0) - src1 * 8 - (dst & 8) * 16,
		     opcode1, 0xc0 + src0 + (dst & 7) * 8);
	else
		emit(bld, 0xc4, 0xc1 - (dst & 8) * 16, (0x70 | opcode0) - src1 * 8,
		     opcode1, 0xc0 + (src0 & 7) + (dst & 7) * 8);
}

static inline void
builder_emit_short_alu(struct builder *bld, int opcode, int dst, int src0, int src1)
{
	emit(bld, 0xc4, 0xe2 - (src0 & 8) * 4 - (dst & 8) * 16, 0x7d - src1 * 8,
	     opcode, 0xc0 + (src0 & 7) + (dst & 7) * 8);
}

static inline void
builder_emit_vpaddd(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0xfe, dst, src0, src1);
}

static inline void
builder_emit_vpsubd(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0xfa, dst, src0, src1);
}

static inline void
builder_emit_vpmulld(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_short_alu(bld, 0x40, dst, src0, src1);
}

static inline void
builder_emit_vaddps(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0c, 0x58, dst, src0, src1);
}

static inline void
builder_emit_vmulps(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0c, 0x59, dst, src0, src1);
}

static inline void
builder_emit_vdivps(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0c, 0x5e, dst, src0, src1);
}

static inline void
builder_emit_vsubps(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0c, 0x5c, dst, src0, src1);
}

static inline void
builder_emit_vpand(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0xdb, dst, src0, src1);
}

static inline void
builder_emit_vpxor(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0xef, dst, src0, src1);
}

static inline void
builder_emit_vpor(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0xeb, dst, src0, src1);
}

static inline void
builder_emit_vpsrlvd(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_short_alu(bld, 0x45, dst, src0, src1);
}

static inline void
builder_emit_vpsravd(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_short_alu(bld, 0x46, dst, src0, src1);
}

static inline void
builder_emit_vpsllvd(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_short_alu(bld, 0x47, dst, src0, src1);
}

/* For the vfmaddXYZps instructions, X and Y are multiplied, Z is
 * added. 1, 2, 3 and refer to the three ymmX register sources, here
 * dst, src0 and src1 (dst is a src too).
 */

static inline void
builder_emit_vfmadd132ps(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_short_alu(bld, 0x98, dst, src0, src1);
}

static inline void
builder_emit_vfmadd231ps(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_short_alu(bld, 0xb8, dst, src0, src1);
}

static inline void
builder_emit_vpabsd(struct builder *bld, int dst, int src0)
{
	builder_emit_short_alu(bld, 0x1e, dst, src0, 0);
}

static inline void
builder_emit_vrsqrtps(struct builder *bld, int dst, int src0)
{
	builder_emit_long_alu(bld, 0x0c, 0x52, dst, src0, 0);
}

static inline void
builder_emit_vsqrtps(struct builder *bld, int dst, int src0)
{
	builder_emit_long_alu(bld, 0x0c, 0x51, dst, src0, 0);
}

static inline void
builder_emit_vrcpps(struct builder *bld, int dst, int src0)
{
	builder_emit_long_alu(bld, 0x0c, 0x53, dst, src0, 0);
}

static inline void
builder_emit_vcmpps(struct builder *bld, int op, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0c, 0xc2, dst, src0, src1);
	emit(bld, op);
}

static inline void
builder_emit_vmaxps(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0c, 0x5f, dst, src0, src1);
}

static inline void
builder_emit_vminps(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0c, 0x5d, dst, src0, src1);
}

static inline void
builder_emit_vpblendvb(struct builder *bld, int dst, int mask, int src0, int src1)
{
	emit(bld, 0xc4, 0xe3, 0x7d - src1 * 8, 0x4c, 0xc0 + dst * 8 + src0, mask * 16);
}

static inline void
builder_emit_vpackssdw(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x09, 0x6b, dst, src0, src1);
}

static inline void
builder_emit_vpmovsxwd(struct builder *bld, int dst, int src)
{
	emit(bld, 0xc4, 0xe2, 0x7d, 0x23, 0xc0 + dst * 8 + src);
}

static inline void
builder_emit_vpmovzxwd(struct builder *bld, int dst, int src)
{
	emit(bld, 0xc4, 0xe2, 0x7d, 0x33, 0xc0 + dst * 8 + src);
}

static inline void
builder_emit_vextractf128(struct builder *bld, int dst, int src, int sel)
{
	emit(bld, 0xc4, 0xe3, 0x7d, 0x19, 0xc0 + dst + src * 8, sel);
}

static inline uint32_t
builder_offset(struct builder *bld, void *p)
{
	return p - (void *) bld->p;
}

static inline void *
builder_get_const_data(struct builder *bld, size_t size, size_t align)
{
	int offset = align_u64(bld->pool_index, align);

	bld->pool_index += size;
	ksim_assert(bld->pool_index <= sizeof(bld->shader->constant_pool));

	return bld->shader->constant_pool + offset;
}

static inline uint32_t *
builder_get_const_ud(struct builder *bld, uint32_t ud)
{
	uint32_t *p;

	p = builder_get_const_data(bld, sizeof *p, 4);

	*p = ud;

	return p;
}

void
builder_init(struct builder *bld, uint64_t surfaces, uint64_t samplers);

void
builder_finish(struct builder *bld);

int
builder_get_reg(struct builder *bld);

void
builder_release_regs(struct builder *bld);

bool
builder_disasm(struct builder *bld);
