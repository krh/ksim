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

struct hs_thread {
	struct thread t;
	struct reg vue_handles[4];
	struct reg *pue;
};

static void
emit_load_hs_payload(struct kir_program *prog)
{
	uint32_t n = gt.ia.topology - _3DPRIM_PATCHLIST_1 + 1;
	uint32_t regs = DIV_ROUND_UP(n, 8);

	if (gt.hs.include_vertex_handles) {
		for (uint32_t i = 0; i < regs ; i++) {
			kir_program_load_v8(prog, offsetof(struct hs_thread, vue_handles[i]));
			kir_program_store_v8(prog, offsetof(struct thread, grf[i + 1]), prog->dst);
		}
	}

	emit_load_constants(prog, &gt.hs.curbe, gt.hs.urb_start_grf);

	/* FIXME: Load vertex data. */
}

void
compile_hs(void)
{
	struct kir_program prog;

	if (!gt.hs.enable)
		return;

	ksim_trace(TRACE_EU | TRACE_AVX, "jit hs\n");

	kir_program_init(&prog,
			 gt.hs.binding_table_address,
			 gt.hs.sampler_state_address);

	emit_load_hs_payload(&prog);

	kir_program_comment(&prog, "eu hs");
	kir_program_emit_shader(&prog, gt.hs.ksp);

	kir_program_add_insn(&prog, kir_eot);

	gt.hs.avx_shader = kir_program_finish(&prog);
}

void
dispatch_hs(struct hs_thread *t, uint32_t instance)
{
	struct reg *grf = &t->t.grf[0];

	/* Not sure what we should make this. */
	uint32_t fftid = 0;
	uint32_t primitive_id = 0;
	uint32_t barrier = 0;

	t->t.mask_q1 = _mm256_set1_epi32(-1);

	/* Fixed function header */
	grf[0] = (struct reg) {
		.ud = {
			urb_entry_to_handle(t->pue),
			primitive_id,
			/* R0.2: MBZ */
			(barrier << 13) | (instance << 17),
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

	if (gt.hs.statistics)
		gt.hs_invocation_count++;

	gt.hs.avx_shader(&t->t);
}

struct ds_thread {
	struct thread t;
	struct vue_buffer buffer;

	struct reg u, v;
	int count;	/* Counts u and v values as TE emit vertices */
	uint32_t pue_grf;
	struct reg *pue;

	/* VUE handles for generated vertices. Tess level 63 requires
	 * 3072 total vertices, but we generate triangles as we go, so
	 * we don't need to hold that many. The most vertices we need
	 * to hold onto at any point is the for tess level 64 for
	 * inner all all outer. While tessellating the outer ring we
	 * need 3 * 64 (vertices on outer edges) + 1 (for wraparound)
	 * + 62 (vertices on inner edge) + 1 (wraparound) = 256
	 * vertices. */
	uint32_t vue_queue[4 * 64];
	uint32_t vue_head, vue_tail;
	uint32_t inner_level, outer_level[3];

	struct prim_queue pq;
};

static inline void
add_vue(struct ds_thread *t, uint32_t handle)
{
	const uint32_t mask = ARRAY_LENGTH(t->vue_queue) - 1;
	ksim_assert(t->vue_head - t->vue_tail < ARRAY_LENGTH(t->vue_queue));
	t->vue_queue[t->vue_head++ & mask] = handle;
}

static inline uint32_t
get_vue(struct ds_thread *t, uint32_t i)
{
	const uint32_t mask = ARRAY_LENGTH(t->vue_queue) - 1;

	return t->vue_queue[i & mask];
}

static void
free_vues(struct ds_thread *t, uint32_t tail)
{
	for (uint32_t i = t->vue_tail; i < tail; i++)
		prim_queue_free_vue(&t->pq, urb_handle_to_entry(get_vue(t, i)));

	t->vue_tail = tail;
}

struct point { float x, y; };
static const struct point svg_tri[3] = {
	{ 100, 600 }, { 450, 10 }, { 900, 700}
};

static FILE *svg;

static void
svg_start(struct ds_thread *t)
{
	svg = fopen("tess.html", "w");

	int width = 1000, height = 1000;

	fprintf(svg, "<!DOCTYPE html>\n<html>\n<body>\n\n"
		"<style>\n"
		"  body { background-color: #297373; color: #ffffff; }\n"
		"  .base { fill: #ff8552; stroke: none; }\n"
		"  .point { fill: black; r: 5; }\n"
		"</style>\n\n"
	       "<h1>Tesselation</h1>\n"
	       "<p>Outer levels: %d, %d, %d</p><p>Inner level: %d</p>\n"
		"<svg height='%d' width='%d'>\n",
		t->outer_level[0], t->outer_level[1], t->outer_level[2], t->inner_level,
	       width, height);

	fprintf(svg, "<polygon points='%.2f,%.2f %.2f,%.2f %.2f,%.2f' class='base'/>\n",
		svg_tri[0].x, svg_tri[0].y,
		svg_tri[1].x, svg_tri[1].y,
		svg_tri[2].x, svg_tri[2].y);
}

static struct point
map_point(float u, float v)
{
	float w = 1.0f - u - v;

	struct point m;
	m.x = svg_tri[0].x * u + svg_tri[1].x * v + svg_tri[2].x * w;
	m.y = svg_tri[0].y * u + svg_tri[1].y * v + svg_tri[2].y * w;

	return m;	
}

static void
svg_point(float u, float v)
{
	struct point m = map_point(u, v);

	if (svg)
		fprintf(svg, "<circle cx='%.2f' cy='%.2f' class='point'/>\n", m.x, m.y);
}

static void
svg_end(void)
{
	if (svg) {
		fprintf(svg, "</svg>\n</body>\n</html>\n");
		fclose(svg);
		svg = NULL;
	}
}

static void
emit_load_ds_payload(struct kir_program *prog)
{
	struct kir_reg u = kir_program_load_v8(prog, offsetof(struct ds_thread, u));
	kir_program_store_v8(prog, offsetof(struct thread, grf[1]), u);

	struct kir_reg v = kir_program_load_v8(prog, offsetof(struct ds_thread, v));
	kir_program_store_v8(prog, offsetof(struct thread, grf[2]), v);

	if (gt.ds.compute_w) {
		kir_program_immf(prog, 1.0f);
		kir_program_alu(prog, kir_subf, prog->dst, u);
		kir_program_alu(prog, kir_subf, prog->dst, v);
	} else {
		kir_program_immf(prog, 0.0f);
	}
	kir_program_store_v8(prog, offsetof(struct thread, grf[3]), prog->dst);

	kir_program_load_v8(prog, offsetof(struct ds_thread, buffer.vue_handles));
	kir_program_store_v8(prog, offsetof(struct thread, grf[4]), prog->dst);

	emit_load_constants(prog, &gt.ds.curbe, gt.ds.urb_start_grf);
}

void
compile_ds(void)
{
	struct kir_program prog;

	if (!gt.ds.enable)
		return;

	ksim_assert(gt.ds.dispatch_mode == DISPATCH_MODE_SIMD8_SINGLE_PATCH);

	ksim_trace(TRACE_EU | TRACE_AVX, "jit ds\n");

	kir_program_init(&prog,
			 gt.ds.binding_table_address,
			 gt.ds.sampler_state_address);

	prog.urb_offset = offsetof(struct ds_thread, buffer.data);

	emit_load_ds_payload(&prog);

	kir_program_comment(&prog, "eu ds");
	kir_program_emit_shader(&prog, gt.ds.ksp);

	if (!gt.gs.enable)
		emit_vertex_post_processing(&prog,
					    offsetof(struct ds_thread, buffer));

	kir_program_add_insn(&prog, kir_eot);

	gt.ds.avx_shader = kir_program_finish(&prog);
}

void
dispatch_ds(struct ds_thread *t)
{
	struct reg *grf = &t->t.grf[0];

	/* Not sure what we should make this. */
	uint32_t fftid = 0;
	uint32_t primitive_id = 0;

	static const struct reg range = { .d = {  0, 1, 2, 3, 4, 5, 6, 7 } };
	t->t.mask_q1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(t->count), range.ireg);

	/* Fixed function header */
	grf[0] = (struct reg) {
		.ud = {
			urb_entry_to_handle(t->pue),
			primitive_id,
			/* R0.2: MBZ */
			0,
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

	/* Copy in PUE contents */
	struct reg *r = (struct reg *) t->pue;
	uint32_t g = t->pue_grf;
	for (uint32_t i = 0; i < gt.ds.pue_read_length; i++)
		grf[g++] = r[gt.ds.pue_read_offset + i];

	if (gt.ds.statistics)
		gt.ds_invocation_count++;

	gt.ds.avx_shader(&t->t);

	/* Transpose the SIMD8 ds vue buffer back into individual VUEs */
	for (uint32_t c = 0; c < t->count; c++) {
		uint32_t handle = t->buffer.vue_handles.ud[c];
		__m256i *vue = urb_handle_to_entry(handle);
		__m256i offsets = (__m256i) (__v8si) { 0, 8, 16, 24, 32, 40, 48, 56 };
		for (uint32_t i = 0; i < gt.ds.urb.size / 32; i++)
			vue[i] = _mm256_i32gather_epi32(&t->buffer.data[i * 8].d[c], offsets, 4);
 	}
	t->count = 0;
}

static void
output_vertex(struct ds_thread *t, float u, float v)
{
	t->u.f[t->count] = u;
	t->v.f[t->count] = v;

	void *entry = alloc_urb_entry(&gt.ds.urb);
	t->buffer.vue_handles.ud[t->count++] = urb_entry_to_handle(entry);
	add_vue(t, urb_entry_to_handle(entry));

	svg_point(u, v);

	if (t->count == 8)
		dispatch_ds(t);
}

static float
quantize(float f, int bits)
{
	return u32_to_float(float_to_u32(f) & ~((1 << bits) - 1));
}

static void
generate_edge_vertices(struct ds_thread *t, int level, int edge, float scale)
{
	float p[64];
	const int bits = 5;
	int vertex_count = level + 1;

	/* Quantize the step value to ensure 1 - (1 - n * step) == n * step
	 * for n < 64. */
	float step = quantize(1.0f / level, bits);
	for (uint32_t i = 0; i < vertex_count / 2; i++) {
		p[i] = step * i;
		p[vertex_count - i - 1] = 1.0f - p[i];
	}
	if (vertex_count & 1)
		p[vertex_count / 2] = 0.5f;

	float mid = 1.0f / 3.0f;
	float other = mid * (1.0f - scale);
	for (uint32_t i = 0; i < vertex_count; i++)
		p[i] = p[i] * scale + other;

	switch (edge) {
	case 0:
		for (uint32_t i = 0; i < level; i++)
			output_vertex(t, p[i], other);
		break;
	case 1:
		for (uint32_t i = 0; i < level; i++)
			output_vertex(t, p[level - i], p[i]);
		break;
	case 2:
		for (uint32_t i = 0; i < level; i++)
			output_vertex(t, other, p[level - i]);
		break;
	}
}

static void
generate_vertices(struct ds_thread *t)
{
	generate_edge_vertices(t, t->outer_level[0], 0, 1.0f);
	generate_edge_vertices(t, t->outer_level[1], 1, 1.0f);
	generate_edge_vertices(t, t->outer_level[2], 2, 1.0f);
	add_vue(t, t->vue_queue[0]);

	for (int l = t->inner_level - 2; l > 0; l -= 2) {
		int first = t->vue_head;
		float scale = (float) l / t->inner_level;

		generate_edge_vertices(t, l, 0, scale);
		generate_edge_vertices(t, l, 1, scale);
		generate_edge_vertices(t, l, 2, scale);
		add_vue(t, get_vue(t, first));
	}

	if ((t->inner_level & 1) == 0) {
		float mid = 1.0f / 3.0f;
		output_vertex(t, mid, mid);
	}

	if (t->count > 0)
		dispatch_ds(t);
}

static void
generate_edge_tris(struct ds_thread *t,
		   int base0, int level0, int base1, int level1)
{
	struct value *vue[3];
	int i0 = 0, i1 = 0;

	while (i0 < level0 || i1 < level1) {
		if (i0 == level0)
			goto advance_inner;
		else if (i1 == level1)
			goto advance_outer;
		else if (i0 * (level1 + 2) < (i1 + 1) * level0)
			goto advance_outer;
		else
			goto advance_inner;

	advance_inner:
		vue[0] = urb_handle_to_entry(get_vue(t, base1+ i1));
		vue[1] = urb_handle_to_entry(get_vue(t, base0 + i0));
		vue[2] = urb_handle_to_entry(get_vue(t, base1 + i1 + 1));
		prim_queue_add(&t->pq, vue, 1);
		i1++;
		continue;

	advance_outer:
		vue[0] = urb_handle_to_entry(get_vue(t, base0 + i0));
		vue[1] = urb_handle_to_entry(get_vue(t, base0 + i0 + 1));
		vue[2] = urb_handle_to_entry(get_vue(t, base1 + i1));
		prim_queue_add(&t->pq, vue, 1);
		i0++;
		continue;
	}
}

static void
generate_tris(struct ds_thread *t)
{
	int outer = 0;
	int inner = t->outer_level[0] + t->outer_level[1] + t->outer_level[2] + 1;

	int level[3] = { t->outer_level[0], t->outer_level[1], t->outer_level[2] };
	for (int l = t->inner_level; l > 1; l -= 2) {
		for (int i = 0; i < 3; i++) {
			generate_edge_tris(t, outer, level[i], inner, l - 2);
			outer += level[i];
			inner += l - 2;
			level[i] = l - 2;
		}

		free_vues(t, outer);
		t->vue_tail++;
		outer++;
		inner++;
	}

	if (t->inner_level & 1) {
		struct value *vue[3];
		vue[0] = urb_handle_to_entry(get_vue(t, outer));
		vue[1] = urb_handle_to_entry(get_vue(t, outer + 1));
		vue[2] = urb_handle_to_entry(get_vue(t, outer + 2));
		prim_queue_add(&t->pq, vue, 1);
		free_vues(t, outer + 3);
		t->vue_tail++;
	} else {
		free_vues(t, outer + 1);
	}

	ksim_assert(t->vue_tail == t->vue_head);
}

void
tessellate_patch(struct value **vue)
{
	struct hs_thread ht;
	uint32_t n = gt.ia.topology - _3DPRIM_PATCHLIST_1 + 1;

	uint32_t grf = gt.hs.urb_start_grf + load_constants(&ht.t, &gt.hs.curbe);
	for (uint32_t i = 0; i < n; i++) {
		ht.vue_handles[i / 8].ud[i & 7] = urb_entry_to_handle(vue[i]);
		struct reg *r = (struct reg *) vue[i];
		for (uint32_t j = 0; j < gt.hs.vue_read_length; j++)
			ht.t.grf[grf++] = r[gt.hs.vue_read_offset + j];
	}

	ht.pue = alloc_urb_entry(&gt.hs.urb);

	for (uint32_t i = 0; i < gt.hs.instance_count + 1; i++)
		dispatch_hs(&ht, i);

	ksim_trace(TRACE_TS, "inner %f, outer: %f %f %f\n",
		   ht.pue->f[4], ht.pue->f[5], ht.pue->f[6],ht.pue->f[7]);

	/* Cull patch if any outer level is nan or <= 0 */
	for (uint32_t i = 5; i < 8; i++)
		if (isnan(ht.pue->f[i]) || ht.pue->f[i] <= 0.0f)
			goto cull_patch;

	struct ds_thread dt;

	dt.count = 0;
	dt.vue_head = 0;
	dt.vue_tail = 0;
	dt.pue = ht.pue;

	dt.inner_level = ht.pue->f[4];
	dt.outer_level[0] = ht.pue->f[5];
	dt.outer_level[1] = ht.pue->f[6];
	dt.outer_level[2] = ht.pue->f[7];

	dt.pue_grf = gt.hs.urb_start_grf + load_constants(&dt.t, &gt.ds.curbe);
	init_vue_buffer(&dt.buffer);

	if (TRACE_TS & trace_mask)
		svg_start(&dt);

	generate_vertices(&dt);

	svg_end();

	enum GEN9_3D_Prim_Topo_Type topology;
	switch (gt.te.topology) {
	case OUTPUT_POINT:
		topology = _3DPRIM_POINTLIST;
		break;
	case OUTPUT_LINE:
		topology = _3DPRIM_LINELIST;
		break;
	case OUTPUT_TRI_CW:
	case OUTPUT_TRI_CCW:
		topology = _3DPRIM_TRILIST;
		break;
	default:
		ksim_unreachable();
	}

	prim_queue_init(&dt.pq, topology, &gt.ds.urb);
	generate_tris(&dt);
	prim_queue_flush(&dt.pq);

 cull_patch:
	free_urb_entry(&gt.hs.urb, ht.pue);
}
