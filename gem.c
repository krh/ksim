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
#include <sys/sysmacros.h>
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
static int (*libc_munmap)(void *addr, size_t length);

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

struct gtt_map {
	struct stub_bo *bo;
	void *virtual;
	size_t length;
	off_t offset;
	int prot;
	struct list link;
};

static LIST_INITIALIZER(pending_map_list);
static LIST_INITIALIZER(dirty_map_list);
static LIST_INITIALIZER(clean_map_list);

static struct stub_bo bos[4096], *bo_free_list;
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
		ksim_assert(next_handle < ARRAY_LENGTH(bos));
		bo = &bos[next_handle++];
	}

	bo->gtt_offset = NOT_BOUND;
	bo->size = size;
	bo->stride = 0;

	return bo;
}

static struct stub_bo *
get_bo(int handle)
{
	if (0 < handle && handle < next_handle &&
            bos[handle].gtt_offset != FREED)
                return &bos[handle];

	return NULL;
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

	ksim_assert(bo != NULL);
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

	/* Anything on the dirty or clean list will be freed when at
	 * munmap time. */
	struct gtt_map *m;
	if (list_find(m, &pending_map_list, link, m->bo == bo)) {
		list_remove(&m->link);
		free(m);
	}

	if (bo->offset != STUB_BO_USERPTR)
		munmap(bo->map, bo->size);

	bo->gtt_offset = FREED;
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
	case I915_PARAM_HAS_EXEC_SOFTPIN:
		*getparam->value = 1;
		return 0;

	case I915_PARAM_HAS_BSD:
	case I915_PARAM_HAS_BLT:
	case I915_PARAM_HAS_VEBOX:
	case I915_PARAM_HAS_BSD2:
	case I915_PARAM_HAS_RESOURCE_STREAMER:
		*getparam->value = 0;
		return 0;

	case I915_PARAM_CMD_PARSER_VERSION:
		*getparam->value = 0;
		return 0;

	case I915_PARAM_MMAP_VERSION:
		*getparam->value = 0;
		return 0;

	case I915_PARAM_REVISION:
		*getparam->value = 0;
		return 0;
	case I915_PARAM_SUBSLICE_TOTAL:
		*getparam->value = 3;
		return 0;
	case I915_PARAM_EU_TOTAL:
		*getparam->value = 24;
		return 0;
	case I915_PARAM_HAS_EXEC_ASYNC:
		*getparam->value = 1;
		return 0;
	case I915_PARAM_MMAP_GTT_VERSION:
		*getparam->value = 1;
		return 0;
	default:
		trace(TRACE_WARN, "unhandled getparam %d\n",
		      getparam->param);
		errno = EINVAL;
		return -1;
	}
}

static void
tile_xmajor(struct stub_bo *bo, void *shadow)
{
	int stride = bo->stride & ~3u;
	int tile_stride = stride / 512;

	ksim_assert((stride & 511) == 0);

	/* We don't know the actual height of the buffer. Round down
	 * to a multiple of tile height so we get an integer number of
	 * tiles. The buffer size often gets rounded up to a power of
	 * two, but the buffer has to be a complete rectangulare grid
	 * of tiles.
	 */
	int height = bo->size / stride & ~7;

	for (int y = 0; y < height; y++) {
		int tile_y = y / 8;
		int iy = y & 7;
		void *src = shadow + y * stride;
		void *dst = bo->map + tile_y * tile_stride * 4096 + iy * 512;

		for (int x = 0; x < tile_stride; x++) {
			for (int c = 0; c < 512; c += 32) {
				__m256i m = _mm256_load_si256(src + x * 512 + c);
				_mm256_store_si256(dst + x * 4096 + c, m);
			}
		}
	}
}

static void
tile_ymajor(struct stub_bo *bo, void *shadow)
{
	int stride = bo->stride & ~3u;
	int tile_stride = stride / 128;
	const int column_stride = 32 * 16;
	int columns = stride / 16;

	/* Same comment as for height above in tile_xmajor(). */
	int height = (bo->size / stride) & ~31;

	ksim_assert((stride & 127) == 0);

	for (int y = 0; y < height; y += 2) {
		int tile_y = y / 32;
		int iy = y & 31;
		void *src = shadow + y * stride;
		void *dst = bo->map + tile_y * tile_stride * 4096 + iy * 16;

		for (int x = 0; x < columns; x++) {
			__m128i lo = _mm_load_si128((src + x * 16));
			__m128i hi = _mm_load_si128((src + x * 16 + stride));
			__m256i p = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
			_mm256_store_si256((dst + x * column_stride), p);
		}
	}
}

static void
flush_gtt_map(struct gtt_map *m)
{
	switch (m->bo->stride & 3) {
	case I915_TILING_X:
		tile_xmajor(m->bo, m->virtual);
		break;
	case I915_TILING_Y:
		tile_ymajor(m->bo, m->virtual);
		break;
	default:
		ksim_unreachable();
	}

	mprotect(m->virtual, m->length, m->prot & ~PROT_WRITE);
	trace(TRACE_GEM, "remapping bo %d as read-only\n", get_handle(m->bo));
}

static void
flush_dirty_maps(void)
{
	struct gtt_map *m;

	list_for_each_entry(m, &dirty_map_list, link)
		flush_gtt_map(m);

	list_insert_list(clean_map_list.prev, &dirty_map_list);
	list_init(&dirty_map_list);
}

static int
dispatch_execbuffer2(int fd, unsigned long request,
		     struct drm_i915_gem_execbuffer2 *execbuffer2)
{
	struct drm_i915_gem_exec_object2 *buffers =
		(void *) (uintptr_t) execbuffer2->buffers_ptr;
	const uint32_t count = execbuffer2->buffer_count;

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_EXECBUFFER2:\n");

	ksim_assert(count > 0);
	ksim_assert((execbuffer2->batch_len & 7) == 0);
	ksim_assert(execbuffer2->num_cliprects == 0);
	ksim_assert(execbuffer2->DR1 == 0);
	ksim_assert(execbuffer2->DR4 == 0);

	flush_dirty_maps();

	bool all_matches = true, all_bound = true;
	for (uint32_t i = 0; i < count; i++) {
		struct stub_bo *bo = get_bo(buffers[i].handle);

		/* Userspace can use an invalid BOs to check for
		 * supported features (it is assumed that the kernel
		 * will return an error if a flag is unsupported,
		 * meaning it will fail before checking that the BO
		 * used doesn't exist) :
		 *
		 * https://cgit.freedesktop.org/mesa/mesa/tree/src/intel/vulkan/anv_gem.c#n338
		 *
		 * The following implementation will report unknown
		 * BOs, meaning we make ksim support any feature.
		 */
		if (bo == NULL) {
			errno = ENOENT;
			return -1;
		}

		trace(TRACE_GEM, "    bo %d, size %ld, ",
		      buffers[i].handle, bo->size);

		if (buffers[i].flags & EXEC_OBJECT_PINNED) {
			bo->gtt_offset = buffers[i].offset;
			bind_bo(bo, bo->gtt_offset);
			trace(TRACE_GEM, "pinning to %08x\n", bo->gtt_offset);
		} else if (bo->gtt_offset == NOT_BOUND &&
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

	for (uint32_t i = 0; i < count; i++) {
		struct stub_bo *bo = get_bo(buffers[i].handle);
		struct drm_i915_gem_relocation_entry *relocs =
			(void *) (uintptr_t) buffers[i].relocs_ptr;

		for (uint32_t j = 0; j < buffers[i].relocation_count; j++) {
			uint32_t handle;
			struct stub_bo *target;
			uint64_t *dst;

			if (execbuffer2->flags & I915_EXEC_HANDLE_LUT) {
				ksim_assert(relocs[j].target_handle <
					    execbuffer2->buffer_count);
				handle = buffers[relocs[j].target_handle].handle;
			} else {
				handle = relocs[j].target_handle;
			}

			target = get_bo(handle);
			ksim_assert(target != NULL);
			ksim_assert(relocs[j].offset + sizeof(*dst) <= bo->size);

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

	struct stub_bo *bo = get_bo(buffers[count - 1].handle);
	ksim_assert(bo != NULL);
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

	if (bo == NULL) {
		errno = ENOENT;
		return -1;
	}

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

	if (bo == NULL) {
		errno = ENOENT;
		return -1;
	}

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

	if (bo == NULL) {
		errno = ENOENT;
		return -1;
	}

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

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_MMAP_GTT: handle %d\n",
	      map_gtt->handle);

	if (bo == NULL) {
		errno = ENOENT;
		return -1;
	}

	uint32_t tiling = bo->stride & 3;
	uint32_t stride = bo->stride & ~3u;
	if (tiling != I915_TILING_NONE) {
		ksim_assert((stride & 127) == 0);
		struct gtt_map *m = malloc(sizeof(*m));
		if (m == NULL)
			return -1;
		m->offset = alloc_range(bo->size);
		m->bo = bo;
		list_insert(pending_map_list.prev, &m->link);
		map_gtt->offset = m->offset;
	} else {
		map_gtt->offset = bo->offset;
	}

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

	if (bo == NULL) {
		errno = ENOENT;
		return -1;
	}

	ksim_assert((set_tiling->stride & 3u) == 0);
	ksim_assert((set_tiling->tiling_mode & ~3u) == 0);
	bo->stride = set_tiling->stride | set_tiling->tiling_mode;
	if (bo->kernel_handle)
		set_kernel_tiling(bo);

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_SET_TILING: bo %d, mode %d, stride %d\n",
	      set_tiling->handle, set_tiling->tiling_mode, set_tiling->stride);
	return 0;
}

static int
dispatch_get_tiling(int fd, unsigned long request,
		    struct drm_i915_gem_get_tiling *get_tiling)
{
	struct stub_bo *bo = get_bo(get_tiling->handle);

	if (bo == NULL) {
		errno = ENOENT;
		return -1;
	}

	get_tiling->tiling_mode = bo->stride & 3u;
	get_tiling->swizzle_mode = 0;
	get_tiling->phys_swizzle_mode = 0;

	trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_GET_TILING\n");

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

	if (bo == NULL) {
		errno = ENOENT;
		return -1;
	}

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
dispatch_version(int fd, unsigned long request,
		 struct drm_version *version)
{
	static const char name[] = "i915";
	static const char date[] = "20160919";
	static const char desc[] = "Intel Graphics";

	version->version_major = 1;
	version->version_minor = 6;
	version->version_patchlevel = 0;

	strncpy(version->name, name, version->name_len);
	version->name_len = strlen(name);
	strncpy(version->date, date, version->date_len);
	version->date_len = strlen(date);
	strncpy(version->desc, desc, version->desc_len);
	version->desc_len = strlen(desc);

	return 0;
}

static int
dispatch_prime_handle_to_fd(int fd, unsigned long request,
			    struct drm_prime_handle *prime)
{
	struct stub_bo *bo = get_bo(prime->handle);
	struct drm_prime_handle p;
	int ret;

	if (bo == NULL) {
		errno = ENOENT;
		return -1;
	}

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
	case DRM_IOCTL_I915_GEM_EXECBUFFER2_WR:
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
		return dispatch_get_tiling(fd, request, argp);

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

	case DRM_IOCTL_I915_GEM_CONTEXT_GETPARAM:
		stub("DRM_IOCTL_I915_GEM_CONTEXT_GETPARAM");
		return 0;
	case DRM_IOCTL_I915_GEM_CONTEXT_SETPARAM:
		stub("DRM_IOCTL_I915_GEM_CONTEXT_SETPARAM");
		return 0;

	case DRM_IOCTL_GET_CAP:
		stub("DRM_IOCTL_GET_CAP");
		return 0;
	case DRM_IOCTL_GEM_CLOSE:
		return dispatch_close(fd, request, argp);

	case DRM_IOCTL_PRIME_FD_TO_HANDLE:
		return dispatch_prime_fd_to_handle(fd, request, argp);

	case DRM_IOCTL_PRIME_HANDLE_TO_FD:
		return dispatch_prime_handle_to_fd(fd, request, argp);

	case DRM_IOCTL_VERSION:
		return dispatch_version(fd, request, argp);

	case DRM_IOCTL_GEM_FLINK:
	case DRM_IOCTL_GEM_OPEN:
	case DRM_IOCTL_GET_MAGIC:
		/* There are many more non-render ioctls, perhaps we
		 * should handle them all here. */
		trace(TRACE_WARN,
		      "gem: non-render ioctl 0x%x\n", _IOC_NR(request));

		errno = EACCES;
		return -1;

	default:
		trace(TRACE_WARN,
		      "gem: unhandled ioctl 0x%x\n", _IOC_NR(request));

		errno = EINVAL;
		return -1;
	}


}

__attribute__ ((visibility ("default"))) void *
mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
	void *p;

	if (fd == -1 || fd != drm_fd)
		return libc_mmap(addr, length, prot, flags, fd, offset);

	p = libc_mmap(addr, length, prot, flags, memfd, offset);

	struct gtt_map *m;
	if (list_find(m, &pending_map_list, link, m->offset == offset)) {
		m->virtual = p;
		m->length = length;
		m->prot = prot;
		list_remove(&m->link);
		list_insert(dirty_map_list.prev, &m->link);
		trace(TRACE_GEM, "mapping shadow buffer for tiled bo %d\n", get_handle(m->bo));
	}

	return p;
}

__attribute__ ((visibility ("default"))) int
munmap(void *addr, size_t length)
{
	struct gtt_map *m;

	if (list_find(m, &dirty_map_list, link, m->virtual == addr)) {
		flush_gtt_map(m);
		list_remove(&m->link);
		free(m);
	} else if (list_find(m, &clean_map_list, link, m->virtual == addr)) {
		list_remove(&m->link);
		free(m);
	}

	return libc_munmap(addr, length);
}

uint32_t trace_mask = TRACE_WARN;
uint32_t breakpoint_mask = 0;
FILE *trace_file;
char *framebuffer_filename;
bool use_threads;

static const struct { const char *name; uint32_t flag; } debug_tags[] = {
	{ "debug",	TRACE_DEBUG },
	{ "spam",	TRACE_SPAM },
	{ "warn",	TRACE_WARN },
	{ "gem",	TRACE_GEM },
	{ "cs",		TRACE_CS },
	{ "vf",		TRACE_VF },
	{ "vs",		TRACE_VS },
	{ "ps",		TRACE_PS },
	{ "eu",		TRACE_EU },
	{ "stub",	TRACE_STUB },
	{ "urb",	TRACE_URB },
	{ "queue",	TRACE_QUEUE },
	{ "avx",	TRACE_AVX },
	{ "ra",		TRACE_RA },
	{ "ts",		TRACE_TS },
	{ "gs",		TRACE_GS },
	{ "all",	~0 },
};

static uint32_t
parse_trace_flags(const char *value)
{
	uint32_t mask = 0;

	for (uint32_t i = 0, start = 0; ; i++) {
		if (value[i] != ',' && value[i] != ';')
			continue;
		for (uint32_t j = 0; j < ARRAY_LENGTH(debug_tags); j++) {
			if (strlen(debug_tags[j].name) == i - start &&
			    memcmp(debug_tags[j].name, &value[start], i - start) == 0) {
				mask |= debug_tags[j].flag;
			}
		}
		if (value[i] == ';')
			break;
		start = i + 1;
	}

	return mask;
}


static bool
is_prefix(const char *s, const char *prefix, const char **arg)
{
	const int len = strlen(prefix);

	if (strncmp(s, prefix, len) == 0) {
		if (s[len] == ';' || s[len] == '\0') {
			if (arg)
				*arg = NULL;
			return true;
		} else if (s[len] == '=') {
			*arg = &s[len + 1];
			return true;
		}
	}

	return false;
}

__attribute__ ((constructor)) static void
ksim_stub_init(void)
{
	const char *args, *s, *end, *value;
	char *filename;

	if (!__builtin_cpu_supports("avx2"))
		error(EXIT_FAILURE, 0, "AVX2 instructions not available");

	args = getenv("KSIM_ARGS");
	ksim_assert(args != NULL);

	for (s = args; end = strchr(s, ';'), end != NULL; s = end + 1) {
		if (is_prefix(s, "quiet", NULL))
			trace_mask = 0;
		else if (is_prefix(s, "file", &value)) {
			ksim_assert(trace_file == NULL);
			filename = strndup(value, end - value);
			trace_file = fopen(filename, "w");
			free(filename);
		} else if (is_prefix(s, "framebuffer", &value)) {
			framebuffer_filename = strndup(value, end - value);
		} else if (is_prefix(s, "trace", &value)) {
			trace_mask = parse_trace_flags(value);
		} else if (is_prefix(s, "breakpoint", &value)) {
			breakpoint_mask = parse_trace_flags(value);
			trace_mask |= breakpoint_mask;
		}
	}

	prctl(PR_SET_PDEATHSIG, SIGHUP);

	libc_close = dlsym(RTLD_NEXT, "close");
	libc_ioctl = dlsym(RTLD_NEXT, "ioctl");
	libc_mmap = dlsym(RTLD_NEXT, "mmap");
	libc_munmap = dlsym(RTLD_NEXT, "munmap");
	if (libc_close == NULL || libc_ioctl == NULL ||
	    libc_mmap == NULL || libc_munmap == NULL)
		error(-1, 0, "ksim: failed to get libc ioctl or close\n");

	if (trace_file == NULL)
		trace_file = stdout;
}
