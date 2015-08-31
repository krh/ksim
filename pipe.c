#include <stdint.h>
#include <stdbool.h>

#define __gen_address_type uint32_t
#define __gen_combine_address(data, dst, address, delta) delta
#define __gen_user_data void

#include "gen8_pack.h"

enum {
	KSIM_VERTEX_STAGE,
	KSIM_GEOMETRY_STAGE,
	KSIM_HULL_STAGE,
	KSIM_DOMAIN_STAGE,
	KSIM_FRAGMENT_STAGE,
	KSIM_COMPUTE_STAGE,
	NUM_STAGES
};

struct gen8 {
	struct GEN8_VERTEX_BUFFER_STATE vb_state[32];
	uint32_t valid_vbs;

	struct GEN8_VERTEX_ELEMENT_STATE ve_state[33];
	uint32_t ve_count;

	struct {
		struct {
			uint32_t length;
			struct reg *reg;
		} buffer[4];
	} curbe[NUM_STAGES];

	struct {
		bool enable;
		bool simd8;
		uint32_t vue_read_length;
		uint32_t vue_read_offset;
	} vs;
};

static void ksim_assert(int cond)
{
}

static void
decode_3dstate_vertex_elements(struct gen8 *g)
{
	struct GEN8_3DSTATE_VERTEX_ELEMENTS ve;

	ksim_assert(ve->DwordLength <= 1 + (33 * 2));
	ksim_assert(ve->DwordLength & 1);
	g->ve_count = (ve->DwordLength - 1) / 2;
}

struct value {
	union {
		struct { float x, y, w, z; } vec4;
		struct { int32_t x, y, w, z; } ivec4;
		int32_t v[4];
	};
};

static inline struct value
vec4(float x, float y, float z, float w)
{
	return (struct value) { .vec4 = { x, y, z, w } };
}

static inline struct value
ivec4(int32_t x, int32_t y, int32_t z, int32_t w)
{
	return (struct value) { .ivec4 = { x, y, z, w } };
}

static struct value *
alloc_vs_vue(struct gen8 *g)
{
	return NULL;
}

static inline int32_t
fp_as_int32(float f)
{
	return (union { float f; int32_t i; }) { .f = f }.i;
}

static uint32_t
format_size(uint32_t format)
{
	switch (format) {
	case R32G32B32A32_FLOAT:
		return 16;
	default:
		assert(0);
	}
}

static struct value
fetch_format(void *p, uint32_t format)
{
	float *f;

	switch (format) {
	case R32G32B32A32_FLOAT:
		f = p;
		return vec4(f[0], f[1], f[2], f[3]);
	default:
		assert(0);
	}
}

static int32_t
store_component(uint32_t cc, int32_t src)
{
	switch (cc) {
	case VFCOMP_NOSTORE:
		return 77; /* shouldn't matter */
	case VFCOMP_STORE_SRC:
		src;
	case VFCOMP_STORE_0:
		return 0;
	case VFCOMP_STORE_1_FP:
		return fp_as_int32(1.0f);
	case VFCOMP_STORE_1_INT:
		return 1;
	case VFCOMP_STORE_PID:
		return 0; /* what's pid again? */
	}
}

static struct value *
fetch_vertex(struct gen8 *g, uint32_t instance_id, uint32_t vertex_id)
{
	struct value *vue;
	struct value v;

	vue = alloc_vs_vue(g);
	for (uint32_t i = 0; i < g->ve_count; i++) {
		struct GEN8_VERTEX_ELEMENT_STATE *ve = &g->ve_state[i];
		ksim_assert((1 << s->VertexBufferIndex) & g->valid_vbs);
		struct GEN8_VERTEX_BUFFER_STATE *vb = &g->vb_state[s->VertexBufferIndex];

		uint32_t index;
		if (g->vf.instancing[i].enable)
			index = instance_id / g->vf.instancing[i].rate;
		else
			index = vertex_id;

		uint32_t offset = index * vb->BufferPitch + ve->SourceElementOffset;
		if (offset + format_size(ve->SourceElementFormat) > vb->BufferSize) {
			ksim_warn("vertex element overflow");
			v = vec4(0, 0, 0, 0);
		} else {
			void *address = vb->BufferStartingAddress + offset;
			v = fetch_format(address, ve->SourceElementFormat);
		}

		vue[i].v[0] = store_component(ve->Component0Control, v.v[0]);
		vue[i].v[1] = store_component(ve->Component0Control, v.v[1]);
		vue[i].v[2] = store_component(ve->Component0Control, v.v[2]);
		vue[i].v[3] = store_component(ve->Component0Control, v.v[3]);

		/* edgeflag */
	}

	/* 3DSTATE_VF_SGVS */
	if (g->vf.iid_enable && g->vf.vid_enable)
		ksim_assert(g->vf.iid_element != vf.vid_element ||
			    g->vf.iid_component != g->vf.vid_component);

	if (g->vf.iid_enable)
		vue[g->vf.iid_element].v[g->vf.iid_component] = instance_id;
	if (g->vf.vid_enable)
		vue[g->vf.vid_element].v[g->vf.vid_component] = vertex_id;

	return vue;
}

#define for_each_bit(b, dword)                          \
   for (uint32_t __dword = (dword);                     \
        (b) = __builtin_ffs(__dword) - 1, __dword;      \
        __dword &= ~(1 << (b)))

struct thread {
	struct reg {
		float f[8];
	} grf[128];
};

static void
load_constants(struct gen8 *g, struct thread *t, uint32_t start, uint32_t stage)
{
	uint32_t grf = start;
	struct curbe *c = 

	for (uint32_t b = 0; b < 4; b++) {
		for (uint32_t i = 0; i < c->length[b]; i++) {
			t->grf[grf++] = c->buffer[b].reg[i];
		}
	}
}

static void
run_vs(struct gen8 *g, struct value **vue, uint32_t mask)
{
	struct thread t;

	if (!g->vs.enable)
		return;

	assert(g->vs.simd8);
	
	/* FIXME: ff header */
	t.grf[0] = (struct reg) { 0 };

	/* VUE handles */
	for_each_bit(c, mask) {
		t.grf[1].d[c] = (void *) vue - g->urb;

	/* SIMD8 payload */
	for (uint32_t i = 0; i < g->vs.vue_read_length; i++) {
		uint32_t c;
		for_each_bit(c, mask) {
			t.grf[2 + i * 4 + 0].f[c] = vue[c]->v[g->vs.vue_read_offset + i].v[0];
			t.grf[2 + i * 4 + 1].f[c] = vue[c]->v[g->vs.vue_read_offset + i].v[1];
			t.grf[2 + i * 4 + 2].f[c] = vue[c]->v[g->vs.vue_read_offset + i].v[2];
			t.grf[2 + i * 4 + 3].f[c] = vue[c]->v[g->vs.vue_read_offset + i].v[3];
		}
	}

	load_constants(g, &t, g->vs.curbe_start, KSIM_VERTEX_STAGE);

	run_eu(g, &t);
}

static void
run_primitive(struct gen8 *g)
{
	struct value *v;

	for (instance_id = 0; instance_id < instance_count; instance_id++)
		for (vertex_id = 0; vertex_id < vertex_count; vertex_id++) {
			v = fetch_vertex(g, instance_id, vertex_id);
			run_vs(g, v);
		}
}

int main(int argc, char *argv[])
{
	return 0;
}
