#include <bfd.h>
#include <dis-asm.h>
#include <limits.h>

struct builder {
	shader_t shader;
	uint8_t *p;

	/* Disassembly fields */
	struct disassemble_info info;
	disassembler_ftype disasm_fn;
	int disasm_last;
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

#define emit_uint16(u)			\
	((u) & 0xff),			\
	(((u) >> 8) & 0xff)

#define emit_uint32(u)			\
	((u) & 0xff),			\
	(((u) >> 8) & 0xff),		\
	(((u) >> 16) & 0xff),		\
	(((u) >> 24) & 0xff)

static inline void
builder_emit_push_rdi(struct builder *bld)
{
	emit(bld, 0x57);
}

static inline void
builder_emit_pop_rdi(struct builder *bld)
{
	emit(bld, 0x5f);
}

static inline void
builder_emit_load_rax_from_offset(struct builder *bld, uint32_t offset)
{
	emit(bld, 0x48, 0x8b, 0x87, emit_uint32(offset));
}

static inline void
builder_emit_jmp_relative(struct builder *bld, int32_t offset)
{
	emit(bld, 0xe9, emit_uint32(offset - 5));
}

static inline void
builder_emit_call_relative(struct builder *bld, int32_t offset)
{
	emit(bld, 0xe8, emit_uint32(offset - 5));
}

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
builder_emit_ret(struct builder *bld)
{
	emit(bld, 0xc3);
}

static inline void *
builder_emit_jne(struct builder *bld)
{
	void *p = bld->p;

	emit(bld, 0x75, 0);

	return p;
}

static inline void
builder_set_branch_target(struct builder *bld, uint8_t *branch, uint8_t *target)
{
	int distance = target - (branch + 2);

	ksim_assert((distance & ~0xff) == 0);
	branch[1] = distance;
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
builder_emit_vpmaskmovd(struct builder *bld, int src, int mask, int32_t offset)
{
	emit(bld, 0xc4, 0xe2 - (src & 8) * 16, 0x7d - mask * 8, 0x8e, 0x87 + (src & 7) * 8,
	     emit_uint32(offset));
}

static inline void
builder_emit_vmovdqa(struct builder *bld, int dst, int src)
{
	ksim_assert(dst < 16 && src < 16);

	if (src < 8)
		emit(bld, 0xc5, 0xfd - (dst & 8) * 16, 0x6f, 0xc0 + (src & 7) + (dst & 7) * 8);
	else if (dst < 8)
		emit(bld, 0xc5, 0x7d, 0x7f, 0xc0 + (src & 7) * 8 + (dst & 7));
	else
		emit(bld, 0xc4, 0x41, 0x7d, 0x6f, 0xc0 + (dst & 7) * 8 + (src & 7));
}

enum {
	RAX,
	RCX,
	RDX,
	RBX,
	RSP,
	RBP,
	RSI,
	RDI,
};

enum {
	IMM_BYTE_OFFSET = 0x40,
	IMM_DWORD_OFFSET = 0x80,
};	

static bool
is_byte_range(int offset)
{
	return -128 <= offset && offset <= 127;
}

static inline void
builder_emit_vmovdqa_from_rax(struct builder *bld, int dst, int offset)
{
	/* vmovdqa offset(%rax),%ymm0 */
	if (offset == 0)
		emit(bld, 0xc5, 0xfd - (dst & 8) * 16, 0x6f,
		     (dst & 7) * 8 | RAX);
	else if (is_byte_range(offset))
		emit(bld, 0xc5, 0xfd - (dst & 8) * 16, 0x6f,
		     (dst & 7) * 8 | RAX | IMM_BYTE_OFFSET, offset);
	else
		emit(bld, 0xc5, 0xfd - (dst & 8) * 16, 0x6f,
		     (dst & 7) * 8 | RAX | IMM_DWORD_OFFSET,
		     emit_uint32(offset));
}

static inline void
builder_emit_vpmaskmovd_to_rax(struct builder *bld, int src, int mask, int offset)
{
	/* vpmaskmovd %ymm0,%ymm1,(%rax) */
	if (offset == 0)
		emit(bld, 0xc4, 0xe2 - (src & 8) * 16, 0x7d - mask * 8, 0x8e,
		     (src & 7) * 8 | RAX);
	else if (is_byte_range(offset))
		emit(bld, 0xc4, 0xe2 - (src & 8) * 16, 0x7d - mask * 8, 0x8e,
		     (src & 7) * 8 | RAX | IMM_BYTE_OFFSET,
		     offset);
	else
		emit(bld, 0xc4, 0xe2 - (src & 8) * 16, 0x7d - mask * 8, 0x8e,
		     (src & 7) * 8 | RAX | IMM_DWORD_OFFSET,
		     emit_uint32(offset));
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
	emit(bld, 0xc5, 0xf9 - (src & 8) * 16, 0x7e, 0x87 + (src & 7) * 0x08, emit_uint32(offset));
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
builder_emit_vpbroadcastw(struct builder *bld, int dst, int32_t offset)
{
	emit(bld, 0xc4, 0xe2 - (dst & 8) * 16, 0x7d, 0x79, 0x87 + (dst & 7) * 8,
	     emit_uint32(offset));
}

static inline void
builder_emit_vpbroadcastw_rip_relative(struct builder *bld, int dst, int32_t offset)
{
	emit(bld, 0xc4, 0xe2 - (dst & 8) * 16, 0x7d, 0x79, 0x05 + (dst & 7) * 8,
	     emit_uint32(offset - 9));
}

static inline void
builder_emit_vpbroadcastw_xmm(struct builder *bld, int dst, int32_t offset)
{
	emit(bld, 0xc4, 0xe2 - (dst & 8) * 16, 0x79, 0x79, 0x87 + (dst & 7) * 8,
	     emit_uint32(offset));
}

static inline void
builder_emit_vpbroadcastq(struct builder *bld, int dst, int32_t offset)
{
	emit(bld, 0xc4, 0xe2 - (dst & 8) * 16, 0x7d, 0x59, 0x87 + (dst & 7) * 8,
	     emit_uint16(offset));
}

static inline void
builder_emit_vbroadcasti128(struct builder *bld, int dst, int32_t offset)
{
	emit(bld, 0xc4, 0xe2 - (dst & 8) * 16, 0x7d, 0x5a, 0x87 + (dst & 7) * 8,
	     emit_uint32(offset));
}

static inline void
builder_emit_vbroadcasti128_rip_relative(struct builder *bld, int dst, int32_t offset)
{
	emit(bld, 0xc4, 0xe2 - (dst & 8) * 16, 0x7d, 0x5a, 0x05 + (dst & 7) * 8,
	     emit_uint32(offset - 9));
}

static inline void
builder_emit_load_rsi_rip_relative(struct builder *bld, int offset)
{
	emit(bld, 0x48, 0x8d, 0x35, emit_uint32(offset - 7));
}

static inline void
builder_emit_load_rsi(struct builder *bld, int offset)
{
	emit(bld, 0x48, 0x8b, 0xb7, emit_uint32(offset));
}

static inline void
builder_emit_load_edi(struct builder *bld, uint32_t value)
{
	emit(bld, 0xbf, emit_uint32(value));
}

static inline void
builder_emit_load_rax(struct builder *bld, uint32_t offset)
{
	emit(bld, 0x48, 0x8b, 0x87, emit_uint32(offset));
}

static inline void
builder_emit_load_rax_rip_relative(struct builder *bld, uint32_t offset)
{
	emit(bld, 0x48, 0x8b, 0x05, emit_uint32(offset - 7));
}

static inline void
builder_emit_vmovmskps(struct builder *bld, uint32_t src)
{
	if (src < 8)
		emit(bld, 0xc5, 0xfc, 0x50, 0xc0 + src);
	else
		emit(bld, 0xc4, 0xc1, 0x7c, 0x50, 0xc0 + src);
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
builder_emit_vpgatherdd(struct builder *bld, int dst, int index, int mask, int scale, int offset)
{
	const uint32_t opcode = 0x90;
	const uint32_t scale_log2 = __builtin_ffs(scale) - 1;

	ksim_assert(offset < 128);

	if (offset == 0)
		emit(bld, 0xc4, 0xe2 - (index & 8) * 8 - (dst & 8) * 16, 0x7d - mask * 8,
		     opcode, 0x04 + (dst & 7) * 8, (index & 7) * 8 + scale_log2 * 0x40);
	else
		emit(bld, 0xc4, 0xe2 - (index & 8) * 8 - (dst & 8) * 16, 0x7d - mask * 8,
		     opcode, 0x44 + (dst & 7) * 8, (index & 7) * 8 + scale_log2 * 0x40, offset);
}


static inline void
builder_emit_vpinsrq_rdi_relative(struct builder *bld, int dst, int src1, int offset, int idx)
{
	int src0 = 0;

	if (offset < 128)
		emit(bld, 0xc4,
		     0xe3 - (src0 & 8) * 4 - (dst & 8) * 16,
		     0xf9 - src1 * 8,
		     0x22,
		     0x47 + (src0 & 7) + (dst & 7) * 8,
		     offset,
		     idx);
	else
		emit(bld, 0xc4,
		     0xe3 - (src0 & 8) * 4 - (dst & 8) * 16,
		     0xf9 - src1 * 8,
		     0x22,
		     0x87 + (src0 & 7) + (dst & 7) * 8,
		     emit_uint32(offset),
		     idx);
}

static inline void
builder_emit_vpinsrd_rdi_relative(struct builder *bld, int dst, int src1, int offset, int idx)
{
	int src0 = 0;

	if (offset < 128)
		emit(bld, 0xc4,
		     0xe3 - (src0 & 8) * 4 - (dst & 8) * 16,
		     0x79 - src1 * 8,
		     0x22,
		     0x47 + (src0 & 7) + (dst & 7) * 8,
		     offset,
		     idx);
	else
		emit(bld, 0xc4,
		     0xe3 - (src0 & 8) * 4 - (dst & 8) * 16,
		     0x79 - src1 * 8,
		     0x22,
		     0x87 + (src0 & 7) + (dst & 7) * 8,
		     emit_uint32(offset),
		     idx);
}

static inline void
builder_emit_vinserti128(struct builder *bld, int dst, int src0, int src1, int idx)
{
	emit(bld, 0xc4, 0xe3 - (src0 & 8) * 4 - (dst & 8) * 16, 0x7d - src1 * 8,
	     0x38, 0xc0 + (src0 & 7) + (dst & 7) * 8);
	emit(bld, idx);
}

static inline void
builder_emit_vpaddd(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0xfe, dst, src0, src1);
}

static inline void
builder_emit_vpaddw(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0xfd, dst, src0, src1);
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
builder_emit_vpmullw(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0xd5, dst, src0, src1);
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
builder_emit_vpandn(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0xdf, dst, src0, src1);
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

static inline void
builder_emit_vpsrld(struct builder *bld, int dst, int src0, int shift)
{
	if (src0 < 8)
		emit(bld, 0xc5, 0xfd - dst * 8, 0x72, 0xd0 + src0, shift);
	else
		emit(bld, 0xc4, 0xc1, 0x7d - dst * 8,
		     0x72, 0xd0 + (src0 & 7), shift);
}

static inline void
builder_emit_vpslld(struct builder *bld, int dst, int src0, int shift)
{
	if (src0 < 8)
		emit(bld, 0xc5, 0xfd - dst * 8, 0x72, 0xf0 + src0, shift);
	else
		emit(bld, 0xc4, 0xc1, 0x7d - dst * 8,
		     0x72, 0xf0 + (src0 & 7), shift);
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
builder_emit_vfnmadd132ps(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_short_alu(bld, 0x9c, dst, src0, src1);
}

static inline void
builder_emit_vfnmadd231ps(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_short_alu(bld, 0xbc, dst, src0, src1);
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
builder_emit_vpcmpeqd(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0x76, dst, src0, src1);
}

static inline void
builder_emit_vpcmpgtd(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x0d, 0x66, dst, src0, src1);
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
builder_emit_short_alu_e3(struct builder *bld, int opcode, int dst, int src0, int src1)
{
	emit(bld, 0xc4, 0xe3 - (src0 & 8) * 4 - (dst & 8) * 16, 0x7d - src1 * 8,
	     opcode, 0xc0 + (src0 & 7) + (dst & 7) * 8);
}

static inline void
builder_emit_vpermilps(struct builder *bld, int dst, int imm, int src0)
{
	int src1 = 0;

	builder_emit_short_alu_e3(bld, 0x04, dst, src0, src1);
	emit(bld, imm);
}


static inline void
builder_emit_vroundps(struct builder *bld, int dst, int op, int src1)
{
	int src0 = 0;

	builder_emit_short_alu_e3(bld, 0x08, dst, src0, src1);
	emit(bld, op);
}

static inline void
builder_emit_vpblendvb(struct builder *bld, int dst, int mask, int src0, int src1)
{
	builder_emit_short_alu_e3(bld, 0x4c, dst, src0, src1);
	emit(bld, mask * 16);
}

static inline void
builder_emit_vpblendd(struct builder *bld, int dst, int mask, int src0, int src1)
{
	builder_emit_short_alu_e3(bld, 0x02, dst, src0, src1);
	emit(bld, mask); /* imm mask */
}

static inline void
builder_emit_vpblendvps(struct builder *bld, int dst, int mask, int src0, int src1)
{
	builder_emit_short_alu_e3(bld, 0x4a, dst, src0, src1);
	emit(bld, mask * 16); /* mask register */
}

static inline void
builder_emit_vpackusdw(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_short_alu(bld, 0x2b, dst, src0, src1);
}

static inline void
builder_emit_vpackssdw(struct builder *bld, int dst, int src0, int src1)
{
	builder_emit_long_alu(bld, 0x09, 0x6b, dst, src0, src1);
}

static inline void
builder_emit_vpmovsxwd(struct builder *bld, int dst, int src)
{
	emit(bld, 0xc4, 0xe2 - (dst & 8) * 16 - (src & 8) * 4,
	     0x7d, 0x23, 0xc0 + (dst & 7) * 8 + (src & 7));
}

static inline void
builder_emit_vpmovzxwd(struct builder *bld, int dst, int src)
{
	emit(bld, 0xc4, 0xe2 - (dst & 8) * 16 - (src & 8) * 4,
	     0x7d, 0x33, 0xc0 + (dst & 7) * 8 + (src & 7));
}

static inline void
builder_emit_vextractf128(struct builder *bld, int dst, int src, int sel)
{
	emit(bld, 0xc4, 0xe3, 0x79, 0x16, 0xc0 + dst + src * 8, sel);
}

static inline void
builder_emit_vpextrd(struct builder *bld, int src, int sel)
{
	emit(bld, 0xc4, 0xe3 - (src & 8) * 16, 0x79, 0x16, 0xc0 | (src & 7) * 8 | RAX, sel);
}

static inline void
builder_emit_add_rax_rip_relative(struct builder *bld, uint32_t offset)
{
	emit(bld, 0x48, 0x03, 0x05, emit_uint32(offset - 7));
}

static inline void
builder_emit_vcvtps2dq(struct builder *bld, int dst, int src)
{
	builder_emit_long_alu(bld, 0x0d, 0x5b, dst, src, 0);
}

static inline void
builder_emit_vcvtdq2ps(struct builder *bld, int dst, int src)
{
	builder_emit_long_alu(bld, 0x0c, 0x5b, dst, src, 0);
}

static inline uint32_t
builder_offset(struct builder *bld, void *p)
{
	return p - (void *) bld->p;
}

static inline int
builder_emit_call(struct builder *bld, void *func)
{
	builder_emit_push_rdi(bld);
	builder_emit_call_relative(bld, (uint8_t *) func - bld->p);
	builder_emit_pop_rdi(bld);

	return 0;
}

static inline void
builder_emit_trap(struct builder *bld)
{
	builder_emit_push_rdi(bld);
	builder_emit_load_edi(bld, SIGTRAP);

	const uint64_t offset = (uint8_t *) raise - bld->p;
	ksim_assert(offset < INT_MAX);
	builder_emit_call_relative(bld, offset);

	builder_emit_pop_rdi(bld);
}

void
builder_init(struct builder *bld);

void
builder_align(struct builder *bld);

shader_t
builder_finish(struct builder *bld);

bool
builder_disasm(struct builder *bld);
