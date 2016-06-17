/*
 * Copyright © 2016 Intel Corporation
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

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <error.h>
#include <assert.h>
#include <i915_drm.h>

#define ARRAY_LENGTH(a) ( sizeof(a) / sizeof((a)[0]) )

static inline bool
is_power_of_two(uint64_t v)
{
	return (v & (v - 1)) == 0;
}

static inline uint64_t
align_u64(uint64_t v, uint64_t a)
{
	assert(is_power_of_two(a));

	return (v + a - 1) & ~(a - 1);
}

struct bo;

static inline uint64_t
__gen_combine_address(struct bo *bo, void *location,
		      const uint64_t address, uint32_t delta)
{
	return address | delta;
}

#define __gen_address_type uint64_t
#define __gen_user_data struct bo
#define __gen_unpack_address __gen_unpack_offset

#include "../gen9_pack.h"

#define GEN9_INTERFACE_DESCRIPTOR_DATA_header
#define GEN9_RENDER_SURFACE_STATE_header

struct device {
	int fd;
	uint32_t context_id;
	uint64_t offset;
	union {
		struct GEN9_MEMORY_OBJECT_CONTROL_STATE mocs;
		uint32_t mocs_as_uint;
	};
};

struct bo {
	uint32_t handle;
	uint32_t size;
	void *map, *end;
	uint32_t cursor;
	uint64_t offset;
};


#define bo_emit(bo, cmd, name)						\
	for (struct cmd name = { cmd##_header },			\
	     *_dst = ({ void *_dst = bo->map + bo->cursor; bo->cursor += cmd##_length * 4; _dst; }); \
	     __builtin_expect(_dst != NULL, 1);				\
	     ({ cmd##_pack(NULL, _dst, &name); _dst = NULL; }))

static int
safe_ioctl(int fd, unsigned long request, void *arg)
{
	int ret;

	do {
		ret = ioctl(fd, request, arg);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	return ret;
}

struct bo *
create_bo(struct device *device, uint32_t size)
{
	struct drm_i915_gem_create gem_create = {
		.size = size,
	};
	struct bo *bo;

	bo = malloc(sizeof(*bo));
	if (bo == NULL)
		goto bail;

	int ret = safe_ioctl(device->fd, DRM_IOCTL_I915_GEM_CREATE, &gem_create);
	if (ret != 0)
		goto bail_bo;

	bo->offset = align_u64(device->offset, 4096);
	device->offset = bo->offset + size;
	bo->handle = gem_create.handle;
	bo->size = size;

	struct drm_i915_gem_mmap gem_mmap = {
		.handle = bo->handle,
		.offset = 0,
		.size = size,
		.flags = 0,
	};

	ret = safe_ioctl(device->fd, DRM_IOCTL_I915_GEM_MMAP, &gem_mmap);
	if (ret != 0)
		goto bail_handle;

	bo->map = (void *)(uintptr_t) gem_mmap.addr_ptr;
	bo->end = bo->map + size;
	bo->cursor = 0;

	return bo;

 bail_handle:;
	struct drm_gem_close close = {
		.handle = bo->handle,
	};

	safe_ioctl(device->fd, DRM_IOCTL_GEM_CLOSE, &close);

 bail_bo:
	free(bo);
 bail:
	return NULL;
}

struct device *
create_device(const char *path)
{
	struct device *device;
	int ret;

	device = malloc(sizeof(*device));
	if (device == NULL)
		goto bail;
	device->fd = open(path, O_RDWR);
	if (device->fd == -1)
		goto bail_device;

	struct drm_i915_gem_context_create create = { 0 };
	ret = safe_ioctl(device->fd, DRM_IOCTL_I915_GEM_CONTEXT_CREATE, &create);
	if (ret == -1)
		goto bail_fd;

	device->context_id = create.ctx_id;
	device->offset = 8192;

	device->mocs = (struct GEN9_MEMORY_OBJECT_CONTROL_STATE) {
		.IndextoMOCSTables = 2
	};

	return device;

 bail_fd:
	close(device->fd);
 bail_device:
	free(device);
 bail:
	return NULL;
}

int
execbuf(struct device *device, struct bo **bos, int nbos)
{
	struct drm_i915_gem_exec_object2 objects[16];
	struct drm_i915_gem_execbuffer2 eb;
	struct bo *batch;

	assert(nbos <= ARRAY_LENGTH(objects));

	for (int i = 0; i < nbos; i++) {
		objects[i].handle = bos[i]->handle;
		objects[i].relocation_count = 0;
		objects[i].relocs_ptr = 0;
		objects[i].alignment = 0;
		objects[i].offset = bos[i]->offset;
		objects[i].flags = EXEC_OBJECT_PINNED;
		objects[i].rsvd1 = 0;
		objects[i].rsvd2 = 0;
	}

	batch = bos[nbos - 1];
	eb.buffers_ptr = (uintptr_t) &objects;
	eb.buffer_count = nbos;
	eb.batch_start_offset = 0;
	eb.batch_len = batch->cursor & ~7;
	eb.cliprects_ptr = 0;
	eb.num_cliprects = 0;
	eb.DR1 = 0;
	eb.DR4 = 0;
	eb.flags = I915_EXEC_HANDLE_LUT | I915_EXEC_RENDER |
		I915_EXEC_CONSTANTS_REL_GENERAL;
	eb.rsvd1 = device->context_id;
	eb.rsvd2 = 0;

	return safe_ioctl(device->fd, DRM_IOCTL_I915_GEM_EXECBUFFER2, &eb);
}

static int
device_wait(struct device *device, struct bo *bo)
{
   struct drm_i915_gem_wait wait = {
	   .bo_handle = bo->handle,
	   .timeout_ns = INT64_MAX,
	   .flags = 0,
   };

   return safe_ioctl(device->fd, DRM_IOCTL_I915_GEM_WAIT, &wait);
}


static uint32_t
load_kernel(struct bo *state, const char *filename)
{
	pid_t pid;
	int p[2], cpp_out;
	int status;
	const uint32_t offset = align_u64(state->cursor, 64);
	uint32_t *dw = state->map + offset;
	char line[256];

	if (pipe(p) == -1)
		error(EXIT_FAILURE, errno, "failed to create pipe");
	pid = fork();
	if (pid == -1)
		error(EXIT_FAILURE, errno, "failed fork");
	if (pid == 0) {
		if (dup2(p[1], STDOUT_FILENO) < 0)
			exit(EXIT_FAILURE);
		close(p[0]);
		close(p[1]);
		execlp("cpp", "cpp", "-P", filename, NULL);
	}

	cpp_out = p[0];
	close(p[1]);

	if (pipe(p) == -1)
		error(EXIT_FAILURE, errno, "failed to create pipe");
	pid = fork();
	if (pid == -1)
		error(EXIT_FAILURE, errno, "failed fork");
	if (pid == 0) {
		if (dup2(cpp_out, STDIN_FILENO) < 0)
			exit(EXIT_FAILURE);
		close(cpp_out);
		if (dup2(p[1], STDOUT_FILENO) < 0)
			exit(EXIT_FAILURE);
		close(p[0]);
		close(p[1]);
		execlp("intel-gen4asm", "intel-gen4asm",
		       "--gen", "9", "-", NULL);
	}

	close(cpp_out);
	close(p[1]);

	FILE *output = fdopen(p[0], "r");
	if (output == NULL)
		error(EXIT_FAILURE, errno, "failed to fdopen asm input");

	while (fgets(line, sizeof(line), output)) {
		if (sscanf(line, " { 0x%x, 0x%x, 0x%x, 0x%x },",
			   &dw[0], &dw[1], &dw[2], &dw[3]) != 4)
			error(EXIT_FAILURE, 0, "invalid asm output: %s", line);
		dw += 4;
	}
	if (wait(&status) == -1 || status != EXIT_SUCCESS)
		error(EXIT_FAILURE, errno, "failed to launch intel-gen4asm");
	if (wait(&status) == -1 || status != EXIT_SUCCESS)
		error(EXIT_FAILURE, errno, "failed to launch intel-gen4asm");
	fclose(output);

	state->cursor = (void *) dw - state->map;

	return offset;
}

static uint32_t
add_buffer(struct device *device, struct bo *state, struct bo *buffer)
{
	/* Tiny hack: keep surface state non-0 so aubinator doesn't
	 * get confused. */
	state->cursor = align_u64(state->cursor + 1, 64);
	uint32_t offset = state->cursor;

	bo_emit(state, GEN9_RENDER_SURFACE_STATE, rss) {
		rss.SurfaceType = SURFTYPE_BUFFER;
		rss.SurfaceArray = false;
		rss.SurfaceFormat = SF_RAW;
		rss.SurfaceVerticalAlignment = 0;
		rss.SurfaceHorizontalAlignment = 0;
		rss.Height = ((buffer->size - 1) >> 7) & 0x3fff;
		rss.Width = (buffer->size - 1) & 0x7f;
		rss.Depth = ((buffer->size - 1) >> 21) & 0x3f;
		rss.SurfacePitch = 0;
		rss.NumberofMultisamples = MULTISAMPLECOUNT_1;
		rss.TileMode = LINEAR;
		rss.SamplerL2BypassModeDisable = true;
		rss.RenderCacheReadWriteMode = true; //WriteOnlyCache;
		rss.MOCS = 4;
		rss.ShaderChannelSelectRed = SCS_RED;
		rss.ShaderChannelSelectGreen = SCS_GREEN;
		rss.ShaderChannelSelectBlue = SCS_BLUE;
		rss.ShaderChannelSelectAlpha = SCS_ALPHA;
		rss.SurfaceBaseAddress = buffer->offset;
	};

	return offset;
}

int main(int argc, char *argv[])
{
	struct device *device;
	struct bo *batch, *state, *ssbo;
	static const char device_path[] = "/dev/dri/renderD128";

	if (argc != 2)
		error(EXIT_FAILURE, 0, "usage: cs-runner INPUT.g4a");

	device = create_device(device_path);
	if (device == NULL)
		error(EXIT_FAILURE, errno, "failed to open %s", device_path);

	batch = create_bo(device, 8192);
	if (batch == NULL)
		error(EXIT_FAILURE, errno, "failed to create batch bo");

	state = create_bo(device, 8192);
	if (state == NULL)
		error(EXIT_FAILURE, errno, "failed to create state batch bo");

	ssbo = create_bo(device, 8192);
	if (ssbo == NULL)
		error(EXIT_FAILURE, errno, "failed to create ssbo");

	bo_emit(batch, GEN9_PIPELINE_SELECT, ps) {
		ps.MaskBits = 3;
		ps.PipelineSelection = GPGPU;
	};

	bo_emit(batch, GEN9_STATE_BASE_ADDRESS, sba) {
		sba.SurfaceStateBaseAddress = state->offset;
		sba.SurfaceStateMemoryObjectControlState = device->mocs;
		sba.SurfaceStateBaseAddressModifyEnable = true;

		sba.DynamicStateBaseAddress = state->offset;
		sba.DynamicStateMemoryObjectControlState = device->mocs;
		sba.DynamicStateBaseAddressModifyEnable = true;

		sba.InstructionBaseAddress = state->offset;
		sba.InstructionMemoryObjectControlState = device->mocs;
		sba.InstructionBaseAddressModifyEnable = true;

		sba.GeneralStateBufferSize = 0xfffff;
		sba.GeneralStateBufferSizeModifyEnable = true;
		sba.DynamicStateBufferSize = 0xfffff;
		sba.DynamicStateBufferSizeModifyEnable = true;
		sba.IndirectObjectBufferSize = 0xfffff;
		sba.IndirectObjectBufferSizeModifyEnable = true;
		sba.InstructionBufferSize = 0xfffff;
		sba.InstructionBuffersizeModifyEnable = true;
	};

	const uint32_t skl_gt2_max_cs_threads = 56;

	const uint32_t curbe_size = 64;

	bo_emit(batch, GEN9_MEDIA_VFE_STATE, mvs) {
		mvs.MaximumNumberofThreads = skl_gt2_max_cs_threads - 1;
		mvs.NumberofURBEntries     = 2;
		mvs.ResetGatewayTimer      = true;
		mvs.URBEntryAllocationSize = 2;
		mvs.CURBEAllocationSize    = curbe_size / 32;
	};

	const uint32_t binding_table_offset = align_u64(state->cursor, 64);
	uint32_t *binding_table = state->map + binding_table_offset;
	state->cursor += 128;

	const uint32_t constant_data_offset = align_u64(state->cursor, 64);
	uint32_t *constant_data = state->map + constant_data_offset;
	state->cursor += curbe_size;
	for (uint32_t i = 0; i < curbe_size / 4; i++)
		constant_data[i] = i;

	bo_emit(batch, GEN9_MEDIA_CURBE_LOAD, mcl) {
		mcl.CURBETotalDataLength = curbe_size;
		mcl.CURBEDataStartAddress = constant_data_offset;
	}

	binding_table[0] = add_buffer(device, state, ssbo);

	state->cursor = align_u64(state->cursor, 64);
	uint32_t desc_offset = state->cursor;
	bo_emit(state, GEN9_INTERFACE_DESCRIPTOR_DATA, idd) {
		idd.KernelStartPointer = load_kernel(state, argv[1]);
		idd.SamplerStatePointer = 0;
		idd.SamplerCount = 0;
		idd.BindingTablePointer = binding_table_offset;
		idd.BindingTableEntryCount = 1;
		idd.ConstantIndirectURBEntryReadLength = curbe_size / 32;
		idd.ConstantURBEntryReadOffset = 0;
		idd.BarrierEnable = false;
		idd.SharedLocalMemorySize = 0;
		idd.GlobalBarrierEnable = false;
		idd.NumberofThreadsinGPGPUThreadGroup = 16;
		idd.CrossThreadConstantDataReadLength = 0;
	}

	bo_emit(batch, GEN9_MEDIA_INTERFACE_DESCRIPTOR_LOAD, midl) {
		midl.InterfaceDescriptorTotalLength =
			GEN9_INTERFACE_DESCRIPTOR_DATA_length * sizeof(uint32_t);
		midl.InterfaceDescriptorDataStartAddress = desc_offset;
	};

	bo_emit(batch, GEN9_GPGPU_WALKER, gw) {
		gw.SIMDSize = SIMD8;

		gw.ThreadDepthCounterMaximum = 0;
		gw.ThreadHeightCounterMaximum = 0;
		gw.ThreadWidthCounterMaximum =  0;

		gw.ThreadGroupIDStartingX = 0;
		gw.ThreadGroupIDXDimension = 1;
		gw.ThreadGroupIDStartingY = 0;
		gw.ThreadGroupIDYDimension = 1;
		gw.ThreadGroupIDStartingResumeZ = 0;
		gw.ThreadGroupIDZDimension = 1;
		gw.RightExecutionMask = 0xffffffff;
		gw.BottomExecutionMask = 0xffffffff;
	}

	bo_emit(batch, GEN9_MEDIA_STATE_FLUSH, msf);

	bo_emit(batch, GEN9_PIPE_CONTROL, pc) {
		pc.RenderTargetCacheFlushEnable = true;
		pc.DCFlushEnable = true;
	}

	bo_emit(batch, GEN9_MI_BATCH_BUFFER_END, bbe);

	memset(ssbo->map, 0x55, 1024);

	struct bo *bos[3] = { state, ssbo, batch };
	if (execbuf(device, bos, ARRAY_LENGTH(bos)) == -1)
		error(EXIT_FAILURE, errno, "execbuf failed");

	if (device_wait(device, batch) == -1)
		error(EXIT_FAILURE, errno, "bo wait failed");

	const uint32_t *map = ssbo->map;
	for (uint32_t i = 0; i < 128; i++) {
		if ((i & 7) == 0)
			printf("%08x:", i * 4);
		printf("  %08x", map[i]);
		if ((i & 7) == 7)
			printf("\n");
	}

	return 0;
}
