#include <stdint.h>
#include <stdbool.h>
#include <signal.h>

#include "ksim.h"

static inline uint32_t
field(uint32_t value, int start, int end)
{
	uint32_t mask;

	mask = ~0U >> (31 - end + start);

	return (value >> start) & mask;
}

static inline uint64_t
get_u64(uint32_t *p)
{
	return p[0] | ((uint64_t) p[1] << 32);
}

static void
illegal_opcode(void)
{
	ksim_assert(!"illegal opcode");
}

static void
unhandled_command(uint32_t *p)
{
	printf("unhandled command\n");
}

static void
handle_mi_batch_buffer_end(uint32_t *p)
{
}

static void
handle_mi_atomic(uint32_t *p)
{
}

typedef void (*command_handler_t)(uint32_t *);

static const command_handler_t mi_commands[] = {
	[10] = handle_mi_batch_buffer_end,
	[47] = handle_mi_atomic
};

static void
handle_state_base_address(uint32_t *p)
{
}

static void
handle_state_sip(uint32_t *p)
{
}

static void
handle_swtess_base_address(uint32_t *p)
{
}

static void
handle_gpgpu_csr_base_address(uint32_t *p)
{
}

static const command_handler_t state_commands[] = {
	[ 1] = handle_state_base_address,
	[ 2] = handle_state_sip,
	[ 3] = handle_swtess_base_address,
	[ 4] = handle_gpgpu_csr_base_address,
};

static void
handle_state_command(uint32_t *p)
{
	uint32_t h = p[0];
	uint32_t opcode = field(h, 24, 26);
	uint32_t subopcode = field(h, 16, 23);

	if (opcode == 0) {
		/* STATE_PREFETCH subopcode  3 */
	} else if (opcode == 1) {
		if (state_commands[subopcode])
			state_commands[subopcode](p);
		else
			unhandled_command(p);
	} else {
		illegal_opcode();
	}
}

static void
handle_pipeline_select(uint32_t *p)
{
}

static const command_handler_t dword_commands[] = {
	[ 5] = handle_pipeline_select
};

static void
handle_dword_command(uint32_t *p)
{
	uint32_t h = p[0];
	uint32_t opcode = field(h, 24, 26);
	uint32_t subopcode = field(h, 16, 23);

	if (opcode == 1) {
		if (dword_commands[subopcode])
			dword_commands[subopcode](p);
		else
			unhandled_command(p);
	} else {
		illegal_opcode();
	}

}

static void
handle_3dprimitive(uint32_t *p)
{
	printf("3dprimitive\n");
}

static struct {
	struct {
		struct {
			uint64_t address;
			uint32_t size;
			uint32_t pitch;
		} vb[32];
		struct {
			uint32_t vb;
			bool valid;
			uint32_t format;
			bool edgeflag;
			uint32_t offset;
			uint8_t cc[4];
		} ve[33];
		struct {
			uint32_t format;
			uint64_t address;
			uint32_t size;
		} ib;
	} vf;
} gt;

static void
handle_3dstate_clear_params(uint32_t *p)
{
}

static void
handle_3dstate_depth_buffer(uint32_t *p)
{
}

static void
handle_3dstate_stencil_buffer(uint32_t *p)
{
}

static void
handle_3dstate_hier_depth_buffer(uint32_t *p)
{
}

static void
handle_3dstate_vertex_buffers(uint32_t *p)
{
	uint32_t h = p[0];
	uint32_t length = field(h, 0, 7) + 2;

	ksim_assert((length - 1) % 4 == 0);

	for (uint32_t i = 1; i < length; i += 4) {
		uint32_t vb = field(p[i], 26, 31);
		bool modify_address = field(p[i], 14, 14);
		gt.vf.vb[vb].pitch = field(p[i], 0, 11);
		if (modify_address)
			gt.vf.vb[vb].address = * (uint64_t *) &p[i + 1];
		gt.vf.vb[vb].size = p[i + 3];
	}
}

static void
handle_3dstate_vertex_elements(uint32_t *p)
{
	uint32_t h = p[0];
	uint32_t length = field(h, 0, 7) + 2;

	ksim_assert((length - 1) % 2 == 0);

	for (uint32_t i = 1, n = 0; i < length; i += 2, n++) {
		gt.vf.ve[n].vb = field(p[i], 26, 31);
		gt.vf.ve[n].valid = field(p[i], 25, 25);
		gt.vf.ve[n].format = field(p[i], 16, 24);
		gt.vf.ve[n].edgeflag = field(p[i], 15, 15);
		gt.vf.ve[n].offset = field(p[i], 0, 11);
		gt.vf.ve[n].cc[0] = field(p[i + 1], 28, 30);
		gt.vf.ve[n].cc[1] = field(p[i + 1], 24, 26);
		gt.vf.ve[n].cc[2] = field(p[i + 1], 20, 22);
		gt.vf.ve[n].cc[3] = field(p[i + 1], 16, 18);
	}
}

static void
handle_3dstate_index_buffer(uint32_t *p)
{
	gt.vf.ib.format = field(p[1], 8, 9);
	gt.vf.ib.address = get_u64(&p[2]);
	gt.vf.ib.size = p[4];
}

static void
handle_3dstate_vf_statistics(uint32_t *p)
{
}

static void
handle_3dstate_vf(uint32_t *p)
{
}

static void
handle_3dstate_multisamlpe(uint32_t *p)
{
}

static void
handle_3dstate_cc_state_pointers(uint32_t *p)
{
}

static void
handle_3dstate_scissor_state_pointers(uint32_t *p)
{
}

static void
handle_3dstate_vs(uint32_t *p)
{
}

static void
handle_3dstate_gs(uint32_t *p)
{
}

static void
handle_3dstate_clip(uint32_t *p)
{
}

static void
handle_3dstate_sf(uint32_t *p)
{
}

static void
handle_3dstate_wm(uint32_t *p)
{
}

static void
handle_3dstate_constant_vs(uint32_t *p)
{
}

static void
handle_3dstate_constant_gs(uint32_t *p)
{
}

static void
handle_3dstate_constant_ps(uint32_t *p)
{
}

static void
handle_3dstate_sample_mask(uint32_t *p)
{
}

static void
handle_3dstate_constant_hs(uint32_t *p)
{
}

static void
handle_3dstate_constant_ds(uint32_t *p)
{
}

static void
handle_3dstate_hs(uint32_t *p)
{
}

static void
handle_3dstate_te(uint32_t *p)
{
}

static void
handle_3dstate_ds(uint32_t *p)
{
}

static void
handle_3dstate_steamout(uint32_t *p)
{
}

static void
handle_3dstate_sbe(uint32_t *p)
{
}

static void
handle_3dstate_ps(uint32_t *p)
{
}

static void
handle_3dstate_viewport_state_pointer_sf_clip(uint32_t *p)
{
}

static void
handle_3dstate_viewport_state_pointer_cc(uint32_t *p)
{
}

static void
handle_3dstate_blend_state_pointers(uint32_t *p)
{
}

static void
handle_3dstate_binding_table_pointers_vs(uint32_t *p)
{
}

static void
handle_3dstate_binding_table_pointers_hs(uint32_t *p)
{
}

static void
handle_3dstate_binding_table_pointers_ds(uint32_t *p)
{
}

static void
handle_3dstate_binding_table_pointers_gs(uint32_t *p)
{
}

static void
handle_3dstate_binding_table_pointers_ps(uint32_t *p)
{
}

static void
handle_3dstate_sampler_state_pointers_vs(uint32_t *p)
{
}

static void
handle_3dstate_sampler_state_pointers_hs(uint32_t *p)
{
}

static void
handle_3dstate_sampler_state_pointers_ds(uint32_t *p)
{
}

static void
handle_3dstate_sampler_state_pointers_gs(uint32_t *p)
{
}

static void
handle_3dstate_sampler_state_pointers_ps(uint32_t *p)
{
}

static void
handle_3dstate_urb_vs(uint32_t *p)
{
}

static void
handle_3dstate_urb_hs(uint32_t *p)
{
}

static void
handle_3dstate_urb_ds(uint32_t *p)
{
}

static void
handle_3dstate_urb_gs(uint32_t *p)
{
}

static void
handle_gather_constant_vs(uint32_t *p)
{
}

static void
handle_gather_constant_gs(uint32_t *p)
{
}

static void
handle_gather_constant_hs(uint32_t *p)
{
}

static void
handle_gather_constant_ds(uint32_t *p)
{
}

static void
handle_gather_constant_ps(uint32_t *p)
{
}

static void
handle_3dstate_binding_table_edit_vs(uint32_t *p)
{
}

static void
handle_3dstate_binding_table_edit_gs(uint32_t *p)
{
}

static void
handle_3dstate_binding_table_edit_hs(uint32_t *p)
{
}

static void
handle_3dstate_binding_table_edit_ds(uint32_t *p)
{
}

static void
handle_3dstate_binding_table_edit_ps(uint32_t *p)
{
}

static void
handle_3dstate_vf_instancing(uint32_t *p)
{
}

static void
handle_3dstate_vf_sgvs(uint32_t *p)
{
}

static void
handle_3dstate_vf_topology(uint32_t *p)
{
}

static void
handle_3dstate_wm_chromakey(uint32_t *p)
{
}

static void
handle_3dstate_ps_blend(uint32_t *p)
{
}

static void
handle_3dstate_wm_depth_stencil(uint32_t *p)
{
}

static void
handle_3dstate_ps_extra(uint32_t *p)
{
}

static void
handle_3dstate_raster(uint32_t *p)
{
}

static void
handle_3dstate_sbe_swiz(uint32_t *p)
{
}

static void
handle_3dstate_wm_hz_op(uint32_t *p)
{
}

static const command_handler_t _3dstate_commands[] = {
	[ 4] = handle_3dstate_clear_params,
	[ 5] = handle_3dstate_depth_buffer,
	[ 6] = handle_3dstate_stencil_buffer,
	[ 7] = handle_3dstate_hier_depth_buffer,
	[ 8] = handle_3dstate_vertex_buffers,
	[ 9] = handle_3dstate_vertex_elements,
	[10] = handle_3dstate_index_buffer,
	[11] = handle_3dstate_vf_statistics,
	[12] = handle_3dstate_vf,
	[13] = handle_3dstate_multisamlpe,
	[14] = handle_3dstate_cc_state_pointers,
	[15] = handle_3dstate_scissor_state_pointers,
	[16] = handle_3dstate_vs,
	[17] = handle_3dstate_gs,
	[18] = handle_3dstate_clip,
	[19] = handle_3dstate_sf,
	[20] = handle_3dstate_wm,

	[21] = handle_3dstate_constant_vs,
	[22] = handle_3dstate_constant_gs,
	[23] = handle_3dstate_constant_ps,
	[24] = handle_3dstate_sample_mask,
	[25] = handle_3dstate_constant_hs,
	[26] = handle_3dstate_constant_ds,

	[27] = handle_3dstate_hs,
	[28] = handle_3dstate_te,
	[29] = handle_3dstate_ds,
	[30] = handle_3dstate_steamout,
	[31] = handle_3dstate_sbe,
	[32] = handle_3dstate_ps,

	[33] = handle_3dstate_viewport_state_pointer_sf_clip,
	[35] = handle_3dstate_viewport_state_pointer_cc,
	[36] = handle_3dstate_blend_state_pointers,

	[38] = handle_3dstate_binding_table_pointers_vs,
	[39] = handle_3dstate_binding_table_pointers_hs,
	[40] = handle_3dstate_binding_table_pointers_ds,
	[41] = handle_3dstate_binding_table_pointers_gs,
	[42] = handle_3dstate_binding_table_pointers_ps,

	[43] = handle_3dstate_sampler_state_pointers_vs,
	[44] = handle_3dstate_sampler_state_pointers_hs,
	[45] = handle_3dstate_sampler_state_pointers_ds,
	[46] = handle_3dstate_sampler_state_pointers_gs,
	[47] = handle_3dstate_sampler_state_pointers_ps,

	[48] = handle_3dstate_urb_vs,
	[49] = handle_3dstate_urb_hs,
	[50] = handle_3dstate_urb_ds,
	[51] = handle_3dstate_urb_gs,

	[52] = handle_gather_constant_vs,
	[53] = handle_gather_constant_gs,
	[54] = handle_gather_constant_hs,
	[55] = handle_gather_constant_ds,
	[56] = handle_gather_constant_ps,

	[67] = handle_3dstate_binding_table_edit_vs,
	[68] = handle_3dstate_binding_table_edit_gs,
	[69] = handle_3dstate_binding_table_edit_hs,
	[70] = handle_3dstate_binding_table_edit_ds,
	[71] = handle_3dstate_binding_table_edit_ps,
	[73] = handle_3dstate_vf_instancing,
	[74] = handle_3dstate_vf_sgvs,
	[75] = handle_3dstate_vf_topology,
	[76] = handle_3dstate_wm_chromakey,
	[77] = handle_3dstate_ps_blend,
	[78] = handle_3dstate_wm_depth_stencil,
	[79] = handle_3dstate_ps_extra,
	[80] = handle_3dstate_raster,
	[81] = handle_3dstate_sbe_swiz,
	[82] = handle_3dstate_wm_hz_op,
};

static void
handle_pipe_control(uint32_t *p)
{
}

static void
handle_3dstate_command(uint32_t *p)
{
	uint32_t h = p[0];
	uint32_t opcode = field(h, 24, 26);
	uint32_t subopcode = field(h, 16, 23);

	if (opcode == 0) {
		if (_3dstate_commands[subopcode])
			_3dstate_commands[subopcode](p);
		else
			unhandled_command(p);

	} else if (opcode == 1) {
		/* 3DSTATE_DRAWING_RECTANGLE subopcode 0 */
		/* 3DSTATE_SAMPLER_PALETTE_LOAD0 subopcode 2 */
		/* 3DSTATE_CHROMA_KEY subopcode 4 */
		/* 3DSTATE_POLY_STIPPLE_OFFSET subopcode 6 */
		/* 3DSTATE_POLY_STIPPLE_PATTERN subopcode 7 */
		/* 3DSTATE_LINE_STIPPLE subopcode 8 */
		/* 3DSTATE_AA_LINE_PARAMETERS subopcode 10 */
		/* 3DSTATE_SAMPLER_PALETTE_LOAD1 subopcode 12 */
		/* 3DSTATE_MONOFILTER_SIZE subopcode 17 */
		/* 3DSTATE_PUSH_CONSTANT_ALLOC_VS subopcode 18 */
		/* 3DSTATE_PUSH_CONSTANT_ALLOC_DS subopcode 19 */
		/* 3DSTATE_PUSH_CONSTANT_ALLOC_DS subopcode 20 */
		/* 3DSTATE_PUSH_CONSTANT_ALLOC_GS subopcode 21 */
		/* 3DSTATE_PUSH_CONSTANT_ALLOC_PS subopcode 22 */
		/* 3DSTATE_DECL_LIST, sub 23 */
		/* 3DSTATE_SO_BUFFER, sub 24 */
		/* 3DSTATE_BINDING_TABLE_POOL_ALLOC, sub 25 */
		/* 3DSTATE_GATHER_POOL_ALLOC, sub 26 */
		/* 3DSTATE_SAMPLE_PATTERN, sub 28 */
	} else if (opcode == 2) {
		if (subopcode == 0)
			handle_pipe_control(p);
		else
			unhandled_command(p);
	} else if (opcode == 3) {
		if (subopcode == 0)
			handle_3dprimitive(p);
		else
			illegal_opcode();
	}

}

void
start_batch_buffer(uint32_t *p)
{
	bool done = false;

	while (!done) {
		uint32_t h = p[0];
		uint32_t type = field(h, 29, 31);
		uint32_t length;

		printf("decoding command at %p: %08x: ", p, p[0]);
		switch (type) {
		case 0: /* MI */ {
			uint32_t opcode = field(h, 23, 28);
			if (mi_commands[opcode])
				mi_commands[opcode](p);
			else
				unhandled_command(p);
			if (opcode == 10) /* bb end */
				done = true;
			if (opcode < 16)
				length = 1;
			else
				length = field(h, 0, 7) + 2;
			break;
		}

		case 1:
		case 2: /* ? */
			ksim_assert(!"unknown command type");
			break;

		case 3: /* Render */ {
			uint32_t subtype = field(h, 27, 28);
			switch (subtype) {
			case 0:
				handle_state_command(p);
				length = field(h, 0, 7) + 2;
				break;
			case 1:
				handle_dword_command(p);
				length = 1;
				break;
			case 2:
				ksim_assert(!"unknown render command subtype");
				length = 2;
				break;
			case 3:
				handle_3dstate_command(p);
				length = field(h, 0, 7) + 2;
				printf("3dstate command length %d\n", length);
				break;
			}
			break;
		}
		}

		p += length;
	}
}
