/*
 * Copyright © 2016 Kristian H. Kristensen <hoegsberg@gmail.com>
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

struct kir_reg {
	int n;
};

static inline struct kir_reg
kir_reg(int n)
{
	return (struct kir_reg) { .n = n };
}

struct kir_program {
	struct list insns;

	struct kir_reg next_reg;
	uint32_t exec_size;
	uint32_t exec_offset;
	struct kir_reg dst;
	int scope;
	int new_scope;
	int quarter;
	uint32_t *live_ranges;
	uint32_t urb_offset;
	uint32_t urb_length;

	uint64_t binding_table_address;
	uint64_t sampler_state_address;
};

enum kir_opcode {
	kir_comment,

	kir_load_region,
	kir_store_region_mask,
	kir_store_region,
	kir_gather,

	kir_set_load_base_indirect,
	kir_set_load_base_imm,
	kir_set_load_base_imm_offset,
	kir_load,
	kir_mask_store,

	kir_immd,
	kir_immw,
	kir_immv,
	kir_immvf,

	/* send */
	kir_send,
	kir_const_send, /* No side effects: sampler or constant cache loads */
	kir_call,
	kir_const_call, /* No side effects: typiclly math helpers */

	/* alu unop */
	kir_mov,
	kir_zxwd,
	kir_sxwd,
	kir_ps2d,
	kir_d2ps,
	kir_absd,
	kir_rcp,
	kir_sqrt,
	kir_rsqrt,
	kir_rndu,
	kir_rndd,
	kir_rnde,
	kir_rndz,
	kir_shri, /* src1 immediate */
	kir_shli, /* src1 immediate */

	/* alu binop */
	kir_and,
	kir_andn,
	kir_or,
	kir_xor,
	kir_shr,
	kir_shl,
	kir_asr,

	kir_maxd,
	kir_maxud,
	kir_maxw,
	kir_maxuw,
	kir_maxf,
	kir_mind,
	kir_minud,
	kir_minw,
	kir_minuw,
	kir_minf,

	kir_divf,
	kir_int_div_q_and_r,
	kir_int_div_q,
	kir_int_div_r,
	kir_int_invm,
	kir_int_rsqrtm,

	kir_addd,
	kir_addw,
	kir_addf,

	kir_subd,
	kir_subw,
	kir_subf,

	kir_muld,
	kir_mulw,
	kir_mulf,

	kir_cmpf,	/* src2 is an cmp op immediate, not register */
	kir_cmpeqd,
	kir_cmpgtd,

	/* alu triops */
	kir_nmaddf,
	kir_maddf,
	kir_blend,

	kir_eot,
	kir_eot_if_dead
};

struct eu_region {
	uint32_t offset; /* num * 32 + subnum */
	uint32_t type_size;
	uint32_t exec_size;
	uint32_t vstride;
	uint32_t width;
	uint32_t hstride;
};

typedef void (*kir_send_helper_t)(struct thread *t, void *args);

struct kir_insn {
	enum kir_opcode opcode;

	struct kir_reg dst;
	int scope; /* FIXME: Should be part of kir_block */
	int quarter;

	union {
		char *comment;

		struct {
			struct eu_region region;
			uint32_t offset;
			struct kir_reg src;
			struct kir_reg mask;
		} xfer;

		/* The base register for store and load comes from the
		 * set_load_base instructions. It's not a real kir
		 * register and we always allocate rax for it. It's
		 * present as a kir_reg so that we can to liveness
		 * analysis and dependency tracking for it. */
		struct {
			struct kir_reg base;
			uint32_t offset;
			struct kir_reg src;
			struct kir_reg mask;
		} store;
		struct {
			uint32_t offset;
			struct kir_reg base;
		} load;

		struct {
			struct kir_reg src;
			uint32_t offset;
			void *pointer;
		} set_load_base;

		struct {
			/* avx regs, for alu ops */
			struct kir_reg src0;
			union {
				struct kir_reg src1;
				uint32_t imm1;
			};
			union {
				struct kir_reg src2;
				uint32_t imm2;
			};
		} alu;

		struct {
			struct kir_reg base;	/* base address */
			struct kir_reg offset;	/* per-channel offset, register */
			struct kir_reg mask;	/* mask register */
			uint32_t scale;		/* immediate scale, 1, 2 or 4 */
			uint32_t base_offset;	/* immediate offset */
		} gather;

		union {
			int32_t d;
			int16_t v[8];
			float vf[4];
		} imm;

		/* A send instruction is a C function that reads its
		 * arguments from a contiguous number of registers in
		 * the EU register file and writes the result back. */
		struct {
			uint32_t src;
			uint32_t mlen;
			uint32_t dst;
			uint32_t rlen;
			kir_send_helper_t func;
			void *args;
			uint32_t exec_size;
		} send;

		/* A call instruction follows C calling conventions
		 * and expects one or two arguments in ymm0 and ymm1,
		 * returns result in ymm0. */
		struct {
			void *func;
			struct kir_reg src0, src1;
			uint32_t args;
		} call;

		struct {
			struct kir_reg src;
		} eot;
	};

	struct list link;
};

static inline struct kir_insn *
kir_insn_next(struct kir_insn *insn)
{
	return container_of(insn->link.next, insn, link);
}

static inline struct kir_insn *
kir_insn_prev(struct kir_insn *insn)
{
	return container_of(insn->link.prev, insn, link);
}

struct kir_insn *
kir_program_add_insn(struct kir_program *prog, uint32_t opcode);

void
kir_program_comment(struct kir_program *prog, const char *fmt, ...);

void
kir_program_init(struct kir_program *prog, uint64_t surfaces, uint64_t samplers);

struct kir_reg
kir_program_alu(struct kir_program *prog, enum kir_opcode opcode, ...);

struct kir_reg
kir_program_load_region(struct kir_program *prog, const struct eu_region *region);

void
kir_program_store_region_mask(struct kir_program *prog, const struct eu_region *region,
			      struct kir_reg src, struct kir_reg mask);

void
kir_program_store_region(struct kir_program *prog, const struct eu_region *region,
			 struct kir_reg src);

struct kir_reg
kir_program_set_load_base_indirect(struct kir_program *prog, uint32_t offset);

struct kir_reg
kir_program_set_load_base_imm(struct kir_program *prog, void *pointer);

struct kir_reg
kir_program_set_load_base_imm_offset(struct kir_program *prog, void *pointer, struct kir_reg offset);

struct kir_reg
kir_program_load(struct kir_program *prog, struct kir_reg base, uint32_t offset);

void
kir_program_mask_store(struct kir_program *prog,
		       struct kir_reg base, uint32_t offset,
		       struct kir_reg mask, struct kir_reg src);

void
__kir_program_send(struct kir_program *prog, struct inst *inst,
		   enum kir_opcode opcode, void *func, void *args);

#define kir_program_send(prog, inst, func, args)			\
	do {								\
		void (*__func)(struct thread *, typeof(args)) = (func);	\
		__kir_program_send((prog), (inst), kir_send, __func, (args)); \
	} while (0)

#define kir_program_const_send(prog, inst, func, args)			\
	do {								\
		void (*__func)(struct thread *, typeof(args)) = (func);	\
		__kir_program_send((prog), (inst), kir_const_send, __func, (args)); \
	} while (0)

struct kir_reg
kir_program_call(struct kir_program *prog, void *func, uint32_t args, ...);

struct kir_reg
kir_program_const_call(struct kir_program *prog, void *func, uint32_t args, ...);

struct kir_reg
kir_program_gather(struct kir_program *prog,
		   struct kir_reg base,
		   struct kir_reg offset,
		   struct kir_reg mask,
		   uint32_t scale, uint32_t base_offset);

shader_t
kir_program_finish(struct kir_program *prog);

void
kir_program_emit(struct kir_program *prog, struct builder *bld);

void
kir_program_emit_shader(struct kir_program *prog, uint64_t kernel_offset);

static inline struct kir_reg
kir_program_immd(struct kir_program *prog, int32_t d)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_immd);

	insn->imm.d = d;

	return insn->dst;
}

static inline struct kir_reg
kir_program_immf(struct kir_program *prog, float f)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_immd);

	insn->imm.d = float_to_u32(f);

	return insn->dst;
}

static inline struct kir_reg
kir_program_load_uniform(struct kir_program *prog, uint32_t offset)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_load_region);

	insn->xfer.region = (struct eu_region) {
		.offset = offset,
		.type_size = 4,
		.exec_size = 1,
		.vstride = 0,
		.width = 1,
		.hstride = 0
	};

	return insn->dst;
}

static inline struct kir_reg
kir_program_load_v8(struct kir_program *prog, uint32_t offset)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_load_region);

	insn->xfer.region = (struct eu_region) {
		.offset = offset,
		.type_size = 4,
		.exec_size = 8,
		.vstride = 8,
		.width = 8,
		.hstride = 1
	};

	return insn->dst;
}

static inline void
kir_program_store_v8(struct kir_program *prog,
		     uint32_t offset, struct kir_reg src)
{
	struct kir_insn *insn = kir_program_add_insn(prog, kir_store_region);

	insn->xfer.src = src;
	insn->xfer.region = (struct eu_region) {
		.offset = offset,
		.type_size = 4,
		.exec_size = 8,
		.vstride = 8,
		.width = 8,
		.hstride = 1
	};
}
