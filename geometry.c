/*
 * Copyright © 2017 Kristian H. Kristensen
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

#include "ksim.h"
#include "kir.h"

struct gs_thread {
	struct thread t;
	struct reg gue_handles;
	struct reg *pue;
};

void
compile_gs(void)
{
	struct kir_program prog;

	if (!gt.gs.enable)
		return;

	ksim_trace(TRACE_EU | TRACE_AVX, "jit gs\n");

	kir_program_init(&prog,
			 gt.gs.binding_table_address,
			 gt.gs.sampler_state_address);

	emit_load_constants(&prog, &gt.gs.curbe, gt.gs.urb_start_grf);

	kir_program_comment(&prog, "eu gs");
	kir_program_emit_shader(&prog, gt.gs.ksp);

	kir_program_add_insn(&prog, kir_eot);

	gt.gs.avx_shader = kir_program_finish(&prog);
}

static void
dump_gue(struct reg *gue, const char *label)
{
	uint32_t count;

	if (gt.gs.static_output)
		count = gt.gs.static_output_vertex_count;
	else
		count = gue[0].ud[0];


	fprintf(trace_file, "%s: static count: %s, vertex_count %d, header_size %d, vertex size %d\n",
		label,
		gt.gs.static_output ? "true" : "false",
		count, gt.gs.control_data_header_size,
		gt.gs.output_vertex_size);
	for (uint32_t i = 0; i < gt.gs.urb.size / 32; i++) {
		fprintf(trace_file, "%2d: ",  i);
		for (uint32_t j = 0; j < 8; j++)
			fprintf(trace_file, "  %6f", gue[i].f[j]);
		fprintf(trace_file, "\n");
	}
}

static void
dump_input_vues(struct value ***vue,
		uint32_t vertex_count, uint32_t primitive_count)
{
	for (uint32_t i = 0; i < primitive_count; i++) {
		fprintf(trace_file, "primitive %d\n", i);
		for (uint32_t j = 0; j < vertex_count; j++) {
			struct value *v = vue[i][j];
			for (uint32_t k = 0; k < gt.vs.urb.size / 16; k++) {
				fprintf(trace_file, "  %f  %f  %f  %f\n",
					v[k].vec4.x, v[k].vec4.y, v[k].vec4.z, v[k].vec4.w);
			}
		}
	}
}


static void
process_primitives(struct reg *gue)
{
	uint32_t count;
	struct reg *control_data;

	if (gt.gs.static_output) {
		count = gt.gs.static_output_vertex_count;
		control_data = &gue[0];
	} else {
		count = gue[0].ud[0];
		control_data = &gue[1];
	}

	if (trace_mask & TRACE_GS)
		dump_gue(gue, "pre transform gue");

	struct value *v[10];
	struct value *first = (struct value *)
		(control_data + gt.gs.control_data_header_size);
	for (uint32_t i = 0; i < count; i++) {
		const float *vp = gt.sf.viewport;

		v[i] = first + gt.gs.output_vertex_size * i;

		/* FIXME: We should do this SIMD8. */
		if (!gt.clip.perspective_divide_disable) {
			v[i][1].vec4.x = v[i][1].vec4.x / v[i][1].vec4.w;
			v[i][1].vec4.y = v[i][1].vec4.y / v[i][1].vec4.w;
			v[i][1].vec4.z = v[i][1].vec4.z / v[i][1].vec4.w;
		}

		if (gt.sf.viewport_transform_enable) {
			v[i][1].vec4.x = v[i][1].vec4.x * vp[0] + vp[3];
			v[i][1].vec4.y = v[i][1].vec4.y * vp[1] + vp[4];
			v[i][1].vec4.z = v[i][1].vec4.z * vp[2] + vp[5];
		}
	}

	if (trace_mask & TRACE_GS)
		dump_gue(gue, "post transform gue");

	ksim_assert(gt.gs.output_topology == _3DPRIM_LINESTRIP);

	//setup_prim(v, gt.gs.output_topology, 0);

	/* FIXME: Needs to use input assembler again. */
	for (uint32_t i = 0; i < count; i += 2) {
		struct value *prim[2] = { v[i], v[i + 1] };
		ksim_trace(TRACE_GS, "line %f,%f - %f,%f\n",
			   v[i][1].vec4.x, v[i][1].vec4.y,
			   v[i + 1][1].vec4.x, v[i + 1][1].vec4.y);
		rasterize_primitive(prim, gt.gs.output_topology);
	}
}

void
dispatch_gs(struct value ***vue,
	    uint32_t vertex_count, uint32_t primitive_count)
{
	struct gs_thread t;

	struct reg *grf = &t.t.grf[0];

	/* FIXME: discard if gt.ia.topology vertices per primitive !=
	 * gt.gs.expected_vertex_count */

	/* Not sure what we should make this. */
	uint32_t fftid = 0;

	static const struct reg range = { .d = {  0, 1, 2, 3, 4, 5, 6, 7 } };
	t.t.mask_q1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(primitive_count), range.ireg);

	/* Fixed function header */
	grf[0] = (struct reg) {
		.ud = {
			0,
			0,
			(gt.ia.topology << 16) | (gt.gs.hint << 22),
			/* R0.3: per-thread scratch space, sampler ptr */
			gt.vs.sampler_state_address |
			gt.vs.scratch_size,
			/* R0.4: binding table pointer */
			gt.vs.binding_table_address,
			/* R0.5: fftid, scratch offset */
			gt.vs.scratch_pointer | fftid,
			/* R0.6: thread id */
			gt.vs.tid++ & 0xffffff,
			/* R0.7: Reserved */
			0,
		}
	};

	for (uint32_t i = 0; i < primitive_count; i++) {
		t.gue_handles.ud[i] = urb_entry_to_handle(alloc_urb_entry(&gt.gs.urb));
		grf[1].ud[i] = t.gue_handles.ud[i];
	}

	uint32_t g = 2;
	if (gt.gs.include_primitive_id) {
		for (uint32_t i = 0; i < primitive_count; i++)
			grf[g].ud[i] = 0;
		g++;
	}

	if (gt.gs.include_vertex_handles) {
		for (uint32_t i = 0; i < vertex_count; i++) {
			for (uint32_t j = 0; j < primitive_count; j++)
				grf[g].ud[j] = urb_entry_to_handle(vue[j][i]);
			g++;
		}
	}

	if (trace_mask & TRACE_GS)
		dump_input_vues(vue, vertex_count, primitive_count);

	g = gt.gs.urb_start_grf + load_constants(&t.t, &gt.gs.curbe);
	for (uint32_t i = 0; i < primitive_count; i++) {
		uint32_t l = g;
		for (uint32_t j = 0; j < vertex_count; j++) {
			uint32_t *v = (uint32_t *) vue[i][j] + gt.gs.vue_read_offset * 8;
			for (uint32_t k = 0; k < gt.gs.vue_read_length * 8; k++) {
				grf[l++].ud[i] = v[k];
			}
		}
	}

	if (gt.gs.statistics)
		gt.gs_invocation_count++;

	gt.gs.avx_shader(&t.t);

	for (uint32_t i = 0; i < primitive_count; i++)
		process_primitives(urb_handle_to_entry(t.gue_handles.ud[i]));

	for (uint32_t i = 0; i < primitive_count; i++)
		free_urb_entry(&gt.gs.urb, urb_handle_to_entry(t.gue_handles.ud[i]));

	/* TODO: vertex post processing, process 8 primitives at a time. */
}
