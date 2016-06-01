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

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <signal.h>
#include <unistd.h>
#include <error.h>
#include <errno.h>
#include <dlfcn.h>
#include <sys/prctl.h>

#include <i915_drm.h>

#include "ksim.h"

#define DRM_MAJOR 226

static int (*libc_close)(int fd);
static int (*libc_ioctl)(int fd, unsigned long request, void *argp);
static void *(*libc_mmap)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);

static int drm_fd = -1;
static int memfd = -1;
static int memfd_size = MEMFD_INITIAL_SIZE;

#define STUB_BO_USERPTR 1
#define STUB_BO_PRIME 2

struct stub_bo {
	union {
		/* Offset into memfd when in use */
		uint64_t offset;
		/* Next pointer when on free list. */
		void *next;
	};

	uint64_t gtt_offset;
	uint32_t size;
	uint32_t stride; /* tiling in lower 2 bits */
	void *map;
	uint32_t kernel_handle;
};

struct gtt_entry {
	uint32_t handle;
};

static struct stub_bo bos[1024], *bo_free_list;
static int next_handle = 1;
#define gtt_order 20
static const uint64_t gtt_size = 4096ul << gtt_order;
static struct gtt_entry gtt[1 << gtt_order];
static uint64_t next_offset = 0ul;

static uint64_t
alloc_range(size_t size)
{
	uint64_t offset;

	if (memfd == -1) {
		memfd = memfd_create("ksim bo", MFD_CLOEXEC);
		ftruncate(memfd, MEMFD_INITIAL_SIZE);
	}

	offset = memfd_size;
	memfd_size += align_u64(size, 4096);
	ftruncate(memfd, memfd_size);

	return offset;
}

static void
free_range(uint64_t offset, size_t size)
{
}

static struct stub_bo *
create_bo(uint64_t size)
{
	struct stub_bo *bo;

	if (bo_free_list) {
		bo = bo_free_list;
		bo_free_list = bo->next;
	} else {
		bo = &bos[next_handle++];
	}

	bo->gtt_offset = NOT_BOUND;
	bo->size = size;

	return bo;
}

static struct stub_bo *
get_bo(int handle)
{
	struct stub_bo *bo;

	ksim_assert(0 < handle && handle < next_handle);
	bo = &bos[handle];

	return bo;
}

static inline uint32_t
get_handle(struct stub_bo *bo)
{
	return bo - bos;
}

void
bind_bo(struct stub_bo *bo, uint64_t offset)
{
	uint32_t num_pages = (bo->size + 4095) >> 12;
	uint32_t start_page = offset >> 12;

	ksim_assert(bo->size > 0);
	ksim_assert(offset < gtt_size);
	ksim_assert(offset + bo->size < gtt_size);

	bo->gtt_offset = offset;
	for (uint32_t p = 0; p < num_pages; p++) {
		ksim_assert(gtt[start_page + p].handle == 0);
		gtt[start_page + p].handle = get_handle(bo);
	}
}

void *
map_gtt_offset(uint64_t offset, uint64_t *range)
{
	struct gtt_entry entry;
	struct stub_bo *bo;

	ksim_assert(offset < gtt_size);
	entry = gtt[offset >> 12];
	bo = get_bo(entry.handle);

	ksim_assert(bo->gtt_offset != NOT_BOUND && bo->size > 0);
	ksim_assert(bo->gtt_offset <= offset);
	ksim_assert(offset < bo->gtt_offset + bo->size);

	*range = bo->gtt_offset + bo->size - offset;

	return bo->map + (offset - bo->gtt_offset);
}


static void
close_bo(struct stub_bo *bo)
{
	free_range(bo->offset, bo->size);

	if (bo->kernel_handle) {
		struct drm_gem_close gem_close;
		int ret;

		ksim_assert(bo->map != NULL);

		gem_close.handle = bo->kernel_handle;
		ret = libc_ioctl(drm_fd, DRM_IOCTL_GEM_CLOSE, &gem_close);
		ksim_assert(ret == 0);

		bo->map = NULL;
		bo->kernel_handle = 0;
	}

	if (bo->offset != STUB_BO_USERPTR)
		munmap(bo->map, bo->size);

	bo->next = bo_free_list;
	bo_free_list = bo;
}

static void
set_kernel_tiling(struct stub_bo *bo)
{
	int ret;
	struct drm_i915_gem_set_tiling set_tiling = {
		.handle = bo->kernel_handle,
		.tiling_mode = bo->stride & 3,
		.stride = bo->stride & ~3u,
		.swizzle_mode = 0,
	};

	ret = libc_ioctl(drm_fd, DRM_IOCTL_I915_GEM_SET_TILING, &set_tiling);
	ksim_assert(ret != -1);
}

static uint32_t
get_kernel_handle(struct stub_bo *bo)
{
	struct drm_i915_gem_userptr userptr;
	int ret;

	if (bo->kernel_handle)
		return bo->kernel_handle;

	bo->map = mmap(NULL, bo->size, PROT_READ | PROT_WRITE,
		       MAP_SHARED, memfd, bo->offset);
	ksim_assert(bo->map != MAP_FAILED);

	userptr.user_ptr = (uint64_t) bo->map;
	userptr.user_size = bo->size;
	userptr.flags = 0;

	ret = libc_ioctl(drm_fd, DRM_IOCTL_I915_GEM_USERPTR, &userptr);

	ksim_assert(ret != -1);

	bo->kernel_handle = userptr.handle;
	set_kernel_tiling(bo);

	return bo->kernel_handle;
}

__attribute__ ((visibility ("default"))) int
close(int fd)
{
	if (fd == drm_fd)
		drm_fd = -1;

	return libc_close(fd);
}

static int
dispatch_getparam(int fd, unsigned long request,
		  struct drm_i915_getparam *getparam)
{
	switch (getparam->param) {
	case I915_PARAM_IRQ_ACTIVE:
	case I915_PARAM_ALLOW_BATCHBUFFER:
	case I915_PARAM_LAST_DISPATCH:
	case I915_PARAM_NUM_FENCES_AVAIL:
	case I915_PARAM_HAS_OVERLAY:
	case I915_PARAM_HAS_PAGEFLIPPING:
	case I915_PARAM_HAS_PRIME_VMAP_FLUSH:
	case I915_PARAM_HAS_SECURE_BATCHES:
	case I915_PARAM_HAS_PINNED_BATCHES:
		errno = EINVAL;
		return -1;

	case I915_PARAM_CHIPSET_ID:
		*getparam->value = 0x1916;
		return 0;

	case I915_PARAM_HAS_GEM:
	case I915_PARAM_HAS_EXECBUF2:
	case I915_PARAM_HAS_RELAXED_FENCING:
	case I915_PARAM_HAS_LLC:
	case I915_PARAM_HAS_WAIT_TIMEOUT:
	case I915_PARAM_HAS_EXEC_NO_RELOC:
	case I915_PARAM_HAS_EXEC_HANDLE_LUT:
	case I915_PARAM_HAS_COHERENT_RINGS:
	case I915_PARAM_HAS_EXEC_CONSTANTS:
	case I915_PARAM_HAS_RELAXED_DELTA:
	case I915_PARAM_HAS_GEN7_SOL_RESET:
	case I915_PARAM_HAS_ALIASING_PPGTT:
	case I915_PARAM_HAS_SEMAPHORES:
	case I915_PARAM_HAS_WT:
	case I915_PARAM_HAS_COHERENT_PHYS_GTT:
		*getparam->value = 1;
		return 0;

	case I915_PARAM_HAS_BSD:
	case I915_PARAM_HAS_BLT:
	case I915_PARAM_HAS_VEBOX:
	case I915_PARAM_HAS_BSD2:
		*getparam->value = 0;
		return 0;

	case I915_PARAM_CMD_PARSER_VERSION:
		*getparam->value = 0;
		return 0;

	case I915_PARAM_MMAP_VERSION:
		*getparam->value = 0;
		return 0;

	case I915_PARAM_REVISION:
	case I915_PARAM_SUBSLICE_TOTAL:
	case I915_PARAM_EU_TOTAL:
		return 0;

	default:
		trace(TRACE_WARN, "unhandled getparam %d\n",
		      getparam->param);
		errno = EINVAL;
		return -1;
	}
}

static int
dispatch_execbuffer2(int fd, unsigned long request,
		     struct drm_i915_gem_execbuffer2 *execbuffer2)
{
	struct drm_i915_gem_exec_object2 *buffers =
		(void *) (uintptr_t) execbuffer2->buffers_ptr;
	struct stub_bo *bo;

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_EXECBUFFER2:\n");

	ksim_assert(execbuffer2->buffer_count > 0);
	ksim_assert((execbuffer2->batch_len & 7) == 0);
	ksim_assert(execbuffer2->num_cliprects == 0);
	ksim_assert(execbuffer2->DR1 == 0);
	ksim_assert(execbuffer2->DR4 == 0);

	/* FIXME: Do relocs and send bo handle and offset for batch
	 * start.  Maybe add a bind message to bind a bo at a given
	 * gtt offset.  Send a bunch of those as we bind bos in the
	 * gtt, then send the exec msg. */

	bool all_matches = true, all_bound = true;
	for (uint32_t i = 0; i < execbuffer2->buffer_count; i++) {
		bo = get_bo(buffers[i].handle);
		trace(TRACE_GEM, "    bo %d, size %ld, ",
		      buffers[i].handle, bo->size);

		if (bo->gtt_offset == NOT_BOUND &&
		    next_offset + bo->size <= gtt_size) {
			uint64_t alignment = max_u64(buffers[i].alignment, 4096);

			bo->gtt_offset = align_u64(next_offset, alignment);
			next_offset = bo->gtt_offset + bo->size;

			bind_bo(bo, bo->gtt_offset);

			trace(TRACE_GEM, "binding to %08x\n", bo->gtt_offset);
		} else {
			trace(TRACE_GEM, "keeping at %08x\n", bo->gtt_offset);
		}

		if (bo->gtt_offset == NOT_BOUND)
			all_bound = false;

		if (bo->gtt_offset != buffers[i].offset)
			all_matches = false;
	}

	if (!all_bound) {
		/* FIXME: Evict all and try again */
		ksim_assert(all_bound);
	}

	if (all_matches && (execbuffer2->flags & I915_EXEC_NO_RELOC))
		/* can skip relocs */;

	for (uint32_t i = 0; i < execbuffer2->buffer_count; i++) {
		struct drm_i915_gem_relocation_entry *relocs;

		bo = get_bo(buffers[i].handle);
		relocs = (void *) (uintptr_t) buffers[i].relocs_ptr;

		for (uint32_t j = 0; j < buffers[i].relocation_count; j++) {
			uint32_t handle;
			struct stub_bo *target;
			uint32_t *dst;

			if (execbuffer2->flags & I915_EXEC_HANDLE_LUT) {
				ksim_assert(relocs[j].target_handle <
					    execbuffer2->buffer_count);
				handle = buffers[relocs[j].target_handle].handle;
			} else {
				handle = relocs[j].target_handle;
			}

			target = get_bo(handle);
			ksim_assert(relocs[j].offset + sizeof(*dst) < bo->size);

			dst = bo->map + relocs[j].offset;
			if (relocs[j].presumed_offset != target->gtt_offset)
				*dst = target->gtt_offset + relocs[j].delta;
		}
	}

	uint32_t ring = execbuffer2->flags & I915_EXEC_RING_MASK;
	switch (ring) {
	case I915_EXEC_RENDER:
	case I915_EXEC_BLT:
		break;
	default:
		ksim_unreachable("unhandled ring");
	}

	uint64_t offset = bo->gtt_offset + execbuffer2->batch_start_offset;
	start_batch_buffer(offset, ring);

	return 0;
}

static int
dispatch_throttle(int fd, unsigned long request, void *p)
{
	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_THROTTLE\n");
	return 0;
}

static int
dispatch_create(int fd, unsigned long request,
		struct drm_i915_gem_create *create)
{
	struct stub_bo *bo = create_bo(create->size);

	bo->offset = alloc_range(create->size);
	create->handle = get_handle(bo);

	bo->map = mmap(NULL, bo->size, PROT_READ | PROT_WRITE,
		       MAP_SHARED, memfd, bo->offset);

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_CREATE: "
	      "new bo %d, size %ld\n", create->handle, bo->size);

	return 0;
}

static int
dispatch_pread(int fd, unsigned long request,
	       struct drm_i915_gem_pread *gem_pread)
{
	struct stub_bo *bo = get_bo(gem_pread->handle);

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_PREAD\n");

	/* Check for integer overflow */
	ksim_assert(gem_pread->offset + gem_pread->size > gem_pread->offset);
	ksim_assert(gem_pread->offset + gem_pread->size <= bo->size);

	return pread(memfd, (void *) (uintptr_t) gem_pread->data_ptr,
		     gem_pread->size, bo->offset + gem_pread->offset);
}

static int
dispatch_pwrite(int fd, unsigned long request,
		struct drm_i915_gem_pwrite *gem_pwrite)
{
	struct stub_bo *bo = get_bo(gem_pwrite->handle);

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_PWRITE: "
	      "bo %d, offset %d, size %d, bo size %ld\n",
	      gem_pwrite->handle,
	      gem_pwrite->offset, gem_pwrite->size, bo->size);

	ksim_assert(gem_pwrite->offset + gem_pwrite->size > gem_pwrite->offset);
	ksim_assert(gem_pwrite->offset + gem_pwrite->size <= bo->size);

	return pwrite(memfd, (void *) (uintptr_t) gem_pwrite->data_ptr,
		      gem_pwrite->size, bo->offset + gem_pwrite->offset);
}

static int
dispatch_mmap(int fd, unsigned long request,
	      struct drm_i915_gem_mmap *gem_mmap)
{
	struct stub_bo *bo = get_bo(gem_mmap->handle);
	void *p;

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_MMAP\n");

	ksim_assert(bo->offset != STUB_BO_USERPTR);

	if (bo->offset == STUB_BO_PRIME) {
		ksim_assert(bo->kernel_handle);
		return libc_ioctl(fd, request, gem_mmap);
	}

	ksim_assert(gem_mmap->flags == 0);
	ksim_assert(gem_mmap->offset + gem_mmap->size > gem_mmap->offset);
	ksim_assert(gem_mmap->offset + gem_mmap->size <= bo->size);

	p = mmap(NULL, gem_mmap->size, PROT_READ | PROT_WRITE,
		 MAP_SHARED, memfd, bo->offset + gem_mmap->offset);

	gem_mmap->addr_ptr = (uint64_t) p;

	return p != MAP_FAILED ? 0 : -1;
}

static int
dispatch_mmap_gtt(int fd, unsigned long request,
		  struct drm_i915_gem_mmap_gtt *map_gtt)
{
	struct stub_bo *bo = get_bo(map_gtt->handle);

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_MMAP_GTT\n");

#if 0
	if (bo->tiling_mode != I915_TILING_NONE)
		trace(TRACE_WARN, "gtt mapping tiled buffer\n");
#endif

	map_gtt->offset = bo->offset;

	return 0;
}

static int
dispatch_set_domain(int fd, unsigned long request,
		    struct drm_i915_gem_set_domain *set_domain)
{
	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_SET_DOMAIN\n");
	return 0;
}

static int
dispatch_set_tiling(int fd, unsigned long request,
		    struct drm_i915_gem_set_tiling *set_tiling)
{
	struct stub_bo *bo = get_bo(set_tiling->handle);

	ksim_assert((set_tiling->stride & 3u) == 0);
	ksim_assert((set_tiling->tiling_mode & ~3u) == 0);
	bo->stride = set_tiling->stride | set_tiling->tiling_mode;
	if (bo->kernel_handle)
		set_kernel_tiling(bo);

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_SET_TILING\n");
	return 0;
}


static int
dispatch_userptr(int fd, unsigned long request,
		 struct drm_i915_gem_userptr *userptr)
{
	struct stub_bo *bo = create_bo(userptr->user_size);
	int ret;

	ret = libc_ioctl(fd, request, userptr);
	if (ret == -1)
		return ret;

	bo->offset = STUB_BO_USERPTR;
	bo->map = (void *) (uintptr_t) userptr->user_ptr;
	bo->kernel_handle = userptr->handle;

	userptr->handle = get_handle(bo);

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_USERPTR size=%llu -> handle=%u\n",
	      userptr->user_size, userptr->handle);

	return 0;
}

static int
dispatch_close(int fd, unsigned long request,
	       struct drm_gem_close *gem_close)
{
	struct stub_bo *bo = get_bo(gem_close->handle);

	trace(TRACE_GEM, "DRM_IOCTL_GEM_CLOSE\n");
	close_bo(bo);

	return 0;
}

static int
dispatch_prime_fd_to_handle(int fd, unsigned long request,
			    struct drm_prime_handle *prime)
{
	struct stub_bo *bo;
	int ret, size;

	size = lseek(fd, 0, SEEK_END);
	bo = create_bo(size);

	ret = libc_ioctl(fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, prime);
	if (ret == -1)
		return -1;

	bo->offset = STUB_BO_PRIME;
	bo->kernel_handle = prime->handle;

	trace(TRACE_GEM, "DRM_IOCTL_PRIME_FD_TO_HANDLE size=%llu -> handle=%u\n",
	      bo->size, get_handle(bo));

	return 0;
}

static int
dispatch_prime_handle_to_fd(int fd, unsigned long request,
			    struct drm_prime_handle *prime)
{
	struct stub_bo *bo = get_bo(prime->handle);
	struct drm_prime_handle p;
	int ret;

	p.handle = get_kernel_handle(bo);
	p.flags = prime->flags;

	ret = libc_ioctl(fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &p);

	prime->fd = p.fd;

	trace(TRACE_GEM, "DRM_IOCTL_PRIME_HANDLE_TO_FD: "
		   "handle %d -> fd %d\n", prime->handle, p.fd);

	return ret;
}

__attribute__ ((visibility ("default"))) int
ioctl(int fd, unsigned long request, ...)
{
	va_list args;
	void *argp;
	struct stat buf;

	va_start(args, request);
	argp = va_arg(args, void *);
	va_end(args);

	if (_IOC_TYPE(request) == DRM_IOCTL_BASE &&
	    drm_fd != fd && fstat(fd, &buf) == 0 &&
	    (buf.st_mode & S_IFMT) == S_IFCHR && major(buf.st_rdev) == DRM_MAJOR) {
		drm_fd = fd;
		trace(TRACE_GEM, "intercept drm ioctl on fd %d\n", fd);
	}

	if (fd != drm_fd)
		return libc_ioctl(fd, request, argp);

	switch (request) {
	case DRM_IOCTL_I915_GETPARAM:
		return dispatch_getparam(fd, request, argp);

	case DRM_IOCTL_I915_SETPARAM: {
		struct drm_i915_setparam *setparam = argp;

		trace(TRACE_GEM, "DRM_IOCTL_I915_SETPARAM: param %d, value %d\n",
		      setparam->param, setparam->value);

		return 0;
	}

	case DRM_IOCTL_I915_GEM_EXECBUFFER:
		trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_EXECBUFFER: unhandled\n");
		return -1;

	case DRM_IOCTL_I915_GEM_EXECBUFFER2:
		return dispatch_execbuffer2(fd, request, argp);

	case DRM_IOCTL_I915_GEM_BUSY:
		trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_BUSY\n");
		return 0;

	case DRM_IOCTL_I915_GEM_SET_CACHING:
		trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_SET_CACHING\n");
		return 0;

	case DRM_IOCTL_I915_GEM_GET_CACHING:
		trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_GET_CACHING\n");
		return 0;

	case DRM_IOCTL_I915_GEM_THROTTLE:
		return dispatch_throttle(fd, request, argp);

	case DRM_IOCTL_I915_GEM_CREATE:
		return dispatch_create(fd, request, argp);

	case DRM_IOCTL_I915_GEM_PREAD:
		return dispatch_pread(fd, request, argp);

	case DRM_IOCTL_I915_GEM_PWRITE:
		return dispatch_pwrite(fd, request, argp);

	case DRM_IOCTL_I915_GEM_MMAP:
		return dispatch_mmap(fd, request, argp);

	case DRM_IOCTL_I915_GEM_MMAP_GTT:
		return dispatch_mmap_gtt(fd, request, argp);

	case DRM_IOCTL_I915_GEM_SET_DOMAIN:
		return dispatch_set_domain(fd, request, argp);

	case DRM_IOCTL_I915_GEM_SW_FINISH:
		trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_SW_FINISH\n");
		return 0;

	case DRM_IOCTL_I915_GEM_SET_TILING:
		return dispatch_set_tiling(fd, request, argp);

	case DRM_IOCTL_I915_GEM_GET_TILING:
		trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_GET_TILING\n");
		return 0;

	case DRM_IOCTL_I915_GEM_GET_APERTURE: {
		struct drm_i915_gem_get_aperture *get_aperture = argp;
		trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_GET_APERTURE\n");
		get_aperture->aper_available_size = 4245561344; /* bdw gt3 */
		return 0;
	}
	case DRM_IOCTL_I915_GEM_MADVISE:
		trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_MADVISE\n");
		return 0;

	case DRM_IOCTL_I915_GEM_WAIT:
		trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_WAIT\n");
		return 0;

	case DRM_IOCTL_I915_GEM_CONTEXT_CREATE: {
		struct drm_i915_gem_context_create *gem_context_create = argp;

		gem_context_create->ctx_id = 1;

		return 0;
	}

	case DRM_IOCTL_I915_GEM_CONTEXT_DESTROY:
		trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_CONTEXT_DESTROY\n");
		return 0;

	case DRM_IOCTL_I915_REG_READ:
		trace(TRACE_GEM, "DRM_IOCTL_I915_REG_READ\n");
		return 0;
	case DRM_IOCTL_I915_GET_RESET_STATS:
		trace(TRACE_GEM, "DRM_IOCTL_I915_GET_RESET_STATS\n");
		return 0;
	case DRM_IOCTL_I915_GEM_USERPTR:
		return dispatch_userptr(fd, request, argp);

	case DRM_IOCTL_GEM_CLOSE:
		return dispatch_close(fd, request, argp);

	case DRM_IOCTL_GEM_FLINK:
		errno = EINVAL;
		return -1;

	case DRM_IOCTL_GEM_OPEN:
		errno = EINVAL;
		return -1;

	case DRM_IOCTL_PRIME_FD_TO_HANDLE:
		return dispatch_prime_fd_to_handle(fd, request, argp);

	case DRM_IOCTL_PRIME_HANDLE_TO_FD:
		return dispatch_prime_handle_to_fd(fd, request, argp);

	case DRM_IOCTL_GET_MAGIC:
		return libc_ioctl(fd, request, argp);

	case DRM_IOCTL_VERSION:
		return libc_ioctl(fd, request, argp);

	default:
		trace(TRACE_WARN,
		      "gem: unhandled ioctl 0x%x\n", _IOC_NR(request));

		return 0;
	}


}

__attribute__ ((visibility ("default"))) void *
mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
	void *p;

	if (fd == -1 || fd != drm_fd)
		return libc_mmap(addr, length, prot, flags, fd, offset);

	p = libc_mmap(addr, length, prot, flags, memfd, offset);

	return p;
}

uint32_t trace_mask = ~0;
FILE *trace_file;
char *framebuffer_filename;
bool use_threads;

__attribute__ ((constructor)) static void
ksim_stub_init(void)
{
	const char *args;

	args = getenv("KSIM_ARGS");
	ksim_assert(args != NULL);
	printf("ksim args: %s\n", args);

	prctl(PR_SET_PDEATHSIG, SIGHUP);

	libc_close = dlsym(RTLD_NEXT, "close");
	libc_ioctl = dlsym(RTLD_NEXT, "ioctl");
	libc_mmap = dlsym(RTLD_NEXT, "mmap");
	if (libc_close == NULL || libc_ioctl == NULL || libc_mmap == NULL)
		error(-1, 0, "ksim: failed to get libc ioctl or close\n");

	trace_file = stdout;
}
