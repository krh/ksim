/*
 * Copyright © 2015 Intel Corporation
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

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <signal.h>
#include <linux/memfd.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <immintrin.h>

#define ARRAY_LENGTH(a) ( sizeof(a) / sizeof((a)[0]) )

#define MEMFD_INITIAL_SIZE 4096

static inline int
memfd_create(const char *name, unsigned int flags)
{
   return syscall(SYS_memfd_create, name, flags);
}

enum {
	MSG_GEM_CREATE,
	MSG_GEM_CLOSE,
	MSG_GEM_BIND,
	MSG_GEM_EXEC,
	MSG_GEM_SET_DOMAIN,
	MSG_GEM_REPLY
};

struct message {
	uint32_t type;
	int handle;
	uint64_t offset;
	uint64_t size;
	int buffer_count;
};

static inline void
__ksim_assert(int cond, const char *file, int line, const char *msg)
{
	if (!cond) {
		printf("%s:%d: assert failed: %s\n", file, line, msg);
		raise(SIGTRAP);
		__builtin_unreachable();
	}
}

#define ksim_assert(cond) __ksim_assert((cond), __FILE__, __LINE__, #cond)

static inline void
__ksim_unreachable(const char *file, int line, const char *fmt, ...)
{
	va_list va;

	printf("%s:%d: unreachable: \n", file, line);
	va_start(va, fmt);
	vprintf(fmt, va);
	va_end(va);

	raise(SIGTRAP);
	__builtin_unreachable();
}

#define ksim_unreachable(format, ...) \
	__ksim_unreachable(__FILE__, __LINE__, format, ##__VA_ARGS__)


enum {
	TRACE_DEBUG = 1 << 0,		/* Debug trace messages. */
	TRACE_SPAM = 1 << 1,		/* Intermittent junk messages */
	TRACE_WARN = 1 << 2,		/* Warnings for out-of-bounds/unintended behavior. */
	TRACE_GEM = 1 << 3,		/* gem layer trace messages */
	TRACE_CS = 1 << 4,		/* command streamer trace */
	TRACE_VF = 1 << 5,		/* vertex fetch trace */
	TRACE_VS = 1 << 6,		/* trace vs execution */
	TRACE_PS = 1 << 7,		/* trace ps execution */
	TRACE_EU = 1 << 8,		/* trace eu details */
	TRACE_STUB = 1 << 9,		/* unimplemented functionality */
	TRACE_URB = 1 << 10,		/* urb traffic */
	TRACE_QUEUE = 1 << 11,		/* thread queue */
};

extern uint32_t trace_mask;
extern FILE *trace_file;
extern char *framebuffer_filename;
extern bool use_threads;

static inline void
ksim_trace(uint32_t tag, const char *fmt, ...)
{
	va_list va;

	if ((tag & trace_mask) == 0)
		return;

	va_start(va, fmt);
	vfprintf(trace_file, fmt, va);
	va_end(va);
}

#define spam(format, ...) \
	ksim_trace(TRACE_SPAM, format, ##__VA_ARGS__)

#define ksim_warn(format, ...) \
	ksim_trace(TRACE_WARN, format, ##__VA_ARGS__)

#define stub(format, ...)						\
	ksim_trace(TRACE_STUB, "%s:%d: unimplemented: " format,		\
		   __FILE__, __LINE__, ##__VA_ARGS__)

static inline uint32_t
field(uint32_t value, int start, int end)
{
	uint32_t mask;

	mask = ~0U >> (31 - end + start);

	return (value >> start) & mask;
}

static inline uint64_t
get_u64(const uint32_t *p)
{
	return p[0] | ((uint64_t) p[1] << 32);
}

static inline bool
is_power_of_two(uint64_t v)
{
	return (v & (v - 1)) == 0;
}

static inline uint64_t
align_u64(uint64_t v, uint64_t a)
{
	ksim_assert(is_power_of_two(a));

	return (v + a - 1) & ~(a - 1);
}

static inline uint64_t
max_u64(uint64_t a, uint64_t b)
{
	return a > b ? a : b;
}

void start_batch_buffer(uint64_t address);

enum {
	KSIM_VERTEX_STAGE,
	KSIM_GEOMETRY_STAGE,
	KSIM_HULL_STAGE,
	KSIM_DOMAIN_STAGE,
	KSIM_FRAGMENT_STAGE,
	KSIM_COMPUTE_STAGE,
	NUM_STAGES
};

/* bdw gt3 */
#define URB_SIZE (384 * 1024)

/* Per stage urb allocation info and entry pool. All sizes in bytes */
struct urb {
	uint32_t size;
	uint32_t count, total, free_list;
	void *data;
};

struct curbe {
	uint32_t size;
	struct {
		uint32_t length;
		uint64_t address;
	} buffer[4];
};

struct gt {
	struct {
		struct vb {
			uint64_t address;
			uint32_t size;
			uint32_t pitch;
		} vb[32];
		uint32_t vb_valid;
		struct ve {
			uint32_t vb;
			bool valid;
			uint32_t format;
			bool edgeflag;
			uint32_t offset;
			uint8_t cc[4];

			bool instancing;
			uint32_t step_rate;
		} ve[33];
		uint32_t ve_count;
		struct {
			uint32_t format;
			uint64_t address;
			uint32_t size;
		} ib;

		bool iid_enable;
		uint32_t iid_element;
		uint32_t iid_component;
		bool vid_enable;
		uint32_t vid_element;
		uint32_t vid_component;
		bool statistics;
		uint32_t cut_index;
	} vf;

	struct {
		uint32_t topology;
		struct {
			struct value *vue[16];
			uint32_t head, tail;
		} queue;
		int tristrip_parity;
		struct value *trifan_first_vertex;
	} ia;

	struct {
		uint32_t tid;
		bool single_dispatch;
		bool vector_mask;
		uint32_t binding_table_entry_count;
		bool priority;
		bool alternate_fp;
		bool opcode_exception;
		bool access_uav;
		bool sw_exception;
		uint64_t scratch_pointer;
		uint32_t scratch_size;
		bool enable;
		bool simd8;
		bool statistics;
		uint32_t vue_read_length;
		uint32_t vue_read_offset;
		uint64_t ksp;
		uint32_t urb_start_grf;
		struct urb urb;
		struct curbe curbe;
		uint32_t binding_table_address;
		uint32_t sampler_state_address;
		void *shader;
		struct shader *avx_shader;
	} vs;

	struct {
		struct urb urb;
		struct curbe curbe;
		uint32_t binding_table_address;
		uint32_t sampler_state_address;
	} hs;

	struct {
		struct urb urb;
		struct curbe curbe;
		uint32_t binding_table_address;
		uint32_t sampler_state_address;
	} ds;

	struct {
		struct urb urb;
		struct curbe curbe;
		uint32_t binding_table_address;
		uint32_t sampler_state_address;
	} gs;

	struct {
		uint64_t viewport_pointer;
		bool viewport_transform_enable;
	} sf;

	struct {
		uint32_t min_x;
		uint32_t min_y;
		uint32_t max_x;
		uint32_t max_y;
		uint32_t origin_x;
		uint32_t origin_y;
	}  drawing_rectangle;

	struct {
		uint32_t barycentric_mode;
	} wm;

	struct {
		uint32_t num_attributes;
	} sbe;

	struct {
		uint32_t tid;
		bool single_dispatch;
		bool vector_mask;
		uint32_t binding_table_entry_count;
		bool priority;
		bool alternate_fp;
		bool opcode_exception;
		bool access_uav;
		bool sw_exception;
		uint64_t scratch_pointer;
		uint32_t scratch_size;
		bool enable_simd8;
		bool statistics;
		bool push_constant_enable;
		uint64_t ksp0;
		uint32_t grf_start0;
		struct curbe curbe;
		uint32_t binding_table_address;
		uint32_t sampler_state_address;
		uint32_t position_offset_xy;
		bool uses_source_depth;
		bool uses_source_w;
		bool uses_input_coverage_mask;
		bool attribute_enable;
		bool fast_clear;
		bool resolve;
		void *shader;
		struct shader *avx_shader;
	} ps;

	struct {
		bool perspective_divide_disable;
	} clip;

	struct {
		uint64_t viewport_pointer;
		uint32_t state;
	} cc;

	struct {
		uint64_t address;
		uint32_t width;
		uint32_t height;
		uint32_t stride;
		uint32_t format;
		bool write_enable;
		bool test_enable;
		uint32_t test_function;
		bool hiz_enable;
		uint64_t hiz_address;
		uint32_t hiz_stride;
	} depth;

	char urb[URB_SIZE];

	bool curbe_dynamic_state_base;

	uint64_t general_state_base_address;
	uint64_t surface_state_base_address;
	uint64_t dynamic_state_base_address;
	uint64_t indirect_object_base_address;
	uint64_t instruction_base_address;

	uint32_t general_state_buffer_size;
	uint32_t dynamic_state_buffer_size;
	uint32_t indirect_object_buffer_size;
	uint32_t general_instruction_size;

	uint64_t sip_address;

	struct {
		bool predicate;
		bool end_offset;
		uint32_t access_type;

		uint32_t vertex_count;
		uint32_t start_vertex;
		uint32_t instance_count;
		uint32_t start_instance;
		int32_t base_vertex;
	} prim;

	struct {
		uint32_t dimx;
		uint32_t dimy;
		uint32_t dimz;
	} dispatch;

	uint32_t vs_invocation_count;
	uint32_t ia_vertices_count;
	uint32_t ia_primitives_count;
	uint32_t ps_invocation_count;
};

extern struct gt gt;

void *map_gtt_offset(uint64_t offset, uint64_t *range);

#define for_each_bit(b, dword)                          \
	for (uint32_t __dword = (dword);		\
	     (b) = __builtin_ffs(__dword) - 1, __dword;	\
	     __dword &= ~(1 << (b)))

struct value {
	union {
		struct { float x, y, z, w; } vec4;
		struct { int32_t x, y, z, w; } ivec4;
		int32_t v[4];
		float f[4];
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

void dispatch_primitive(void);

bool valid_vertex_format(uint32_t format);
uint32_t format_size(uint32_t format);
struct value fetch_format(uint64_t offset, uint32_t format);
uint32_t depth_format_size(uint32_t format);

struct primitive {
	struct { float x, y, z, w; } v[3];
	struct value *vue[3];
};

void rasterize_primitive(struct primitive *prim);

/* URB handles are indexes to 64 byte blocks in the URB. */

static inline uint32_t
urb_entry_to_handle(void *entry)
{
	uint32_t handle = (entry - (void *) gt.urb) / 64;

	ksim_assert((void *) gt.urb <= entry &&
		    entry < (void *) gt.urb + sizeof(gt.urb));

	return handle;
}

static inline void *
urb_handle_to_entry(uint32_t handle)
{
	void *entry = (void *) gt.urb + handle * 64;

	ksim_assert(handle < sizeof(gt.urb) / 64);

	return entry;
}

struct surface {
	void *pixels;
	int format;
	int width;
	int height;
	int stride;
	int cpp;
};

bool
get_surface(uint32_t binding_table_offset, int i, struct surface *s);

void wm_stall(void);
void wm_flush(void);
void wm_clear(void);
void hiz_clear(void);

struct reg {
	union {
		__m256 reg;
		__m256i ireg;
		float f[8];
		uint32_t ud[8];
		int32_t d[8];
		uint16_t uw[16];
		int16_t w[16];
		uint8_t ub[16];
		int8_t b[16];
		uint64_t uq[4];
		int64_t q[4];
	};
};

struct thread {
	struct reg grf[128];
	uint32_t mask;
};

struct shader {
	uint8_t constant_pool[1024];
	uint8_t code[1024] __attribute__ ((aligned (64)));
};

struct sfid_sampler_args {
	int src;
	int dst;
	struct surface tex;
};

void sfid_sampler(struct thread *t, const struct sfid_sampler_args *args);

struct sfid_urb_args {
	int src;
	int offset;
	int len;
};

void sfid_urb_simd8_write(struct thread *t, struct sfid_urb_args *args);


struct sfid_render_cache_args {
	struct reg offsets;
	int src;
	struct surface rt;
};

void sfid_render_cache_rt_write_simd8(struct thread *t,
				      const struct sfid_render_cache_args *args);


void prepare_shaders(void);
bool execute_inst(void *inst, struct thread *t);
void print_inst(void *p);
uint32_t load_constants(struct thread *t, struct curbe *c, uint32_t start);
void run_thread(struct thread *t, void *kernel, uint32_t trace_flag);
void *compile_shader(void *kernel, struct shader *shader,
		     uint64_t surfaces, uint64_t samplers);


#define __gen_address_type uint64_t
#define __gen_combine_address(data, dst, address, delta) delta
#define __gen_user_data void
#define __gen_unpack_address(qw, start, end) __gen_unpack_field(qw, start, end)

#include "gen8_pack.h"

#define BIM_PERSPECTIVE_PIXEL		 1
#define BIM_PERSPECTIVE_CENTROID	 2
#define BIM_PERSPECTIVE_SAMPLE		 4
#define BIM_LINEAR_PIXEL		 8
#define BIM_LINEAR_CENTROID		16
#define BIM_LINEAR_SAMPLE		32
