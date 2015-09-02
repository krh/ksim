#include "ksim.h"

#include "libdisasm/gen_disasm.h"

void
run_thread(struct thread *t, uint64_t ksp)
{
	static struct gen_disasm *disasm;
	const int gen = 8;
	uint64_t range;
	void *kernel;;

	if (disasm == NULL)
		disasm = gen_disasm_create(gen);

	kernel = map_gtt_offset(ksp + gt.instruction_base_address, &range);

	if (trace_mask & TRACE_KERNELS) {
		ksim_trace(TRACE_KERNELS, "disassembled kernel:\n");
		gen_disasm_disassemble(disasm, kernel, 0, range, trace_file);
		ksim_trace(TRACE_KERNELS, "\n");
	}
}

