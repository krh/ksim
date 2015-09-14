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

#include <i915_drm.h>

#include "ksim.h"

#define DRM_MAJOR 226

static int (*libc_close)(int fd);
static int (*libc_ioctl)(int fd, unsigned long request, void *argp);
static void *(*libc_mmap)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
static int (*libc_munmap)(void *addr, size_t length);

static int drm_fd = -1;

struct ugem_bo {
	uint64_t size;
	void *data;
	uint32_t tiling_mode;
	uint32_t stride;
	uint64_t offset;
	uint32_t read_domains;
	uint32_t write_domain;
	int kernel_handle;
};

struct gtt_entry {
	uint32_t handle;
};

#define gtt_order 20
static const uint64_t gtt_size = 4096ul << gtt_order;
static struct gtt_entry gtt[1 << gtt_order];
static uint64_t next_offset = 4096ul;

static struct ugem_bo bos[1024], *bo_free_list;
static int next_handle = 1;

static uint32_t
add_bo(uint64_t size)
{
	uint32_t handle;
	struct ugem_bo *bo;

	if (bo_free_list) {
		handle = bo_free_list - bos;
		bo = bo_free_list;
		bo_free_list = bo->data;
	} else {
		handle = next_handle++;
		bo = &bos[handle];
	}

	bo->write_domain = 0;
	bo->read_domains = 0;
	bo->size = size;
	bo->data = malloc(size);

	return handle;
}

static struct ugem_bo *
get_bo(uint32_t handle)
{
	struct ugem_bo *bo;

	ksim_assert(handle < next_handle);
	bo = &bos[handle];
	ksim_assert(bo->data != NULL);

	return bo;
}

static void
bind_bo(struct ugem_bo *bo, uint64_t offset)
{
	uint32_t num_pages = (bo->size + 4095) >> 12;
	uint32_t start_page = offset >> 12;

	ksim_assert(offset < gtt_size);
	ksim_assert(offset + bo->size < gtt_size);

	bo->offset = offset;
	for (uint32_t p = 0; p < num_pages; p++) {
		gtt[start_page + p] = (struct gtt_entry) {
			.handle = bo - bos,
		};
	}
}

static void
create_kernel_bo(int fd, struct ugem_bo *bo)
{
	struct drm_i915_gem_create create = { .size = bo->size };
	libc_ioctl(fd, DRM_IOCTL_I915_GEM_CREATE, &create);
	bo->kernel_handle = create.handle;

#ifdef REAL_DEAL
	struct drm_i915_gem_set_tiling set_tiling = {
		.handle = bo->kernel_handle,
		.tiling_mode = bo->tiling_mode,
		.stride = bo->stride,
	};
	libc_ioctl(fd, DRM_IOCTL_I915_GEM_SET_TILING, &set_tiling);
#else
	struct drm_i915_gem_set_tiling set_tiling = {
		.handle = bo->kernel_handle,
		.tiling_mode = I915_TILING_NONE,
		.stride = bo->stride,
	};
	libc_ioctl(fd, DRM_IOCTL_I915_GEM_SET_TILING, &set_tiling);
#endif

	struct drm_i915_gem_mmap mmap = {
		.handle = bo->kernel_handle,
		.offset = 0,
		.size = bo->size,
	};
	libc_ioctl(fd, DRM_IOCTL_I915_GEM_MMAP, &mmap);

	free(bo->data);
	bo->data = (void *) mmap.addr_ptr;
}

void *
map_gtt_offset(uint64_t offset, uint64_t *range)
{
	struct gtt_entry entry;
	struct ugem_bo *bo;

	ksim_assert(offset < gtt_size);
	entry = gtt[offset >> 12];
	bo = get_bo(entry.handle);

	ksim_assert(bo->offset <= offset);
	ksim_assert(offset < bo->offset + bo->size);

	*range = bo->offset + bo->size - offset;

	return bo->data + (offset - bo->offset);
}

static void
dispatch_execbuffer2(struct drm_i915_gem_execbuffer2 *execbuffer2)
{
	struct drm_i915_gem_exec_object2 *buffers = (void *) execbuffer2->buffers_ptr;
	uint32_t bound_count = 0;
	struct ugem_bo *bo;

	ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_EXECBUFFER2:\n");
	
	ksim_assert((execbuffer2->batch_len & 7) == 0);
	ksim_assert(execbuffer2->num_cliprects == 0);
	ksim_assert(execbuffer2->DR1 == 0);
	ksim_assert(execbuffer2->DR4 == 0);
	ksim_assert((execbuffer2->flags & I915_EXEC_RING_MASK) == I915_EXEC_RENDER);

	for (uint32_t i = 0; i < execbuffer2->buffer_count; i++) {
		bo = get_bo(buffers[i].handle);
		ksim_trace(TRACE_GEM, "    bo %d, size %ld, ",
			   buffers[i].handle, bo->size, bo->offset);
		if (bo->offset == 0 && next_offset + bo->size <= gtt_size) {
			uint64_t alignment = max_u64(buffers[i].alignment, 4096);

			bind_bo(bo, align_u64(next_offset, alignment));
			next_offset = bo->offset + bo->size;
			ksim_trace(TRACE_GEM, "binding to %08x\n", bo->offset);
		} else {
			ksim_trace(TRACE_GEM, "keeping at %08x\n", bo->offset);
		}

		if (bo->offset != 0)
			bound_count++;

		buffers->offset = bo->offset;
	}

	if (bound_count != execbuffer2->buffer_count)
		ksim_assert(!"could not bind all bos\n");

	bool all_matches = true;
	for (uint32_t i = 0; i < execbuffer2->buffer_count; i++) {
		bo = get_bo(buffers[i].handle);
		if (bo->offset != buffers[i].offset)
			all_matches = false;
	}

	if (all_matches && (execbuffer2->flags & I915_EXEC_NO_RELOC))
		/* can skip relocs */;

	for (uint32_t i = 0; i < execbuffer2->buffer_count; i++) {
		struct drm_i915_gem_relocation_entry *relocs;

		bo = get_bo(buffers[i].handle);
		relocs = (void *) buffers[i].relocs_ptr;

		for (uint32_t j = 0; j < buffers[i].relocation_count; j++) {
			uint32_t handle;
			struct ugem_bo *target;
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
			dst = bo->data + relocs[j].offset;
			if (relocs[j].presumed_offset != target->offset)
				*dst = target->offset + relocs[j].delta;
		}
	}	

	start_batch_buffer(bo->offset + execbuffer2->batch_start_offset);
}

__attribute__ ((visibility ("default"))) int
close(int fd)
{
	if (fd == drm_fd)
		drm_fd = -1;

	return libc_close(fd);
}

__attribute__ ((visibility ("default"))) int
ioctl(int fd, unsigned long request, ...)
{
	va_list args;
	void *argp;
	int ret;
	struct stat buf;

	va_start(args, request);
	argp = va_arg(args, void *);
	va_end(args);

	if (_IOC_TYPE(request) == DRM_IOCTL_BASE &&
	    drm_fd != fd && fstat(fd, &buf) == 0 &&
	    (buf.st_mode & S_IFMT) == S_IFCHR && major(buf.st_rdev) == DRM_MAJOR) {
		drm_fd = fd;
		ksim_trace(TRACE_DEBUG, "intercept drm ioctl on fd %d\n", fd);
	}

	if (fd != drm_fd)
		return libc_ioctl(fd, request, argp);

	switch (request) {
	case DRM_IOCTL_I915_GETPARAM: {
		struct drm_i915_getparam *getparam = argp;

		ret = libc_ioctl(fd, request, argp);

		return ret;
	}

	case DRM_IOCTL_I915_SETPARAM: {
		struct drm_i915_setparam *setparam = argp;

		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_SETPARAM: param %d, value %d\n",
			   setparam->param, setparam->value);

		return 0;
	}

	case DRM_IOCTL_I915_GEM_EXECBUFFER: {
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_EXECBUFFER: unhandled\n");
		return -1;
	}

	case DRM_IOCTL_I915_GEM_EXECBUFFER2:
		dispatch_execbuffer2(argp);
		return 0;

	case DRM_IOCTL_I915_GEM_BUSY:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_BUSY\n");
		return 0;

	case DRM_IOCTL_I915_GEM_SET_CACHING:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_SET_CACHING\n");
		return 0;

	case DRM_IOCTL_I915_GEM_GET_CACHING:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_GET_CACHING\n");
		return 0;

	case DRM_IOCTL_I915_GEM_THROTTLE:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_THROTTLE\n");
		return 0;

	case DRM_IOCTL_I915_GEM_CREATE: {
		struct drm_i915_gem_create *create = argp;

		create->handle = add_bo(create->size);
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_CREATE: handle %d, size %ld\n",
			   create->handle, create->size);

		return 0;
	}

	case DRM_IOCTL_I915_GEM_PREAD:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_PREAD\n");
		return 0;

	case DRM_IOCTL_I915_GEM_PWRITE: {
		struct drm_i915_gem_pwrite *pwrite = argp;
		struct ugem_bo *bo = get_bo(pwrite->handle);
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_PWRITE: "
			   "bo %d, offset %d, size %d, bo size %ld\n",
			   pwrite->handle, pwrite->offset, pwrite->size, bo->size);

		ksim_assert(pwrite->offset + pwrite->size > pwrite->offset &&
			    pwrite->offset + pwrite->size <= bo->size);
		memcpy(bo->data + pwrite->offset,
		       (void *) pwrite->data_ptr, pwrite->size);

		return 0;
	}

	case DRM_IOCTL_I915_GEM_MMAP: {
		struct drm_i915_gem_mmap *mmap = argp;
		struct ugem_bo *bo = get_bo(mmap->handle);

		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_MMAP\n");
			
		ksim_assert(mmap->flags == 0);
		ksim_assert(mmap->offset + mmap->size > mmap->offset &&
			    mmap->offset + mmap->size <= bo->size);

		mmap->addr_ptr = (uint64_t) (bo->data + mmap->offset);

		return 0;
	}


	case DRM_IOCTL_I915_GEM_MMAP_GTT: {
		struct drm_i915_gem_mmap_gtt *map_gtt = argp;
		struct ugem_bo *bo = get_bo(map_gtt->handle);

		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_MMAP_GTT\n");

		if (bo->tiling_mode != I915_TILING_NONE)
			ksim_trace(TRACE_WARN, "gtt mapping tiled buffer\n");

		map_gtt->offset = (uint64_t) bo;

		return 0;
	}

	case DRM_IOCTL_I915_GEM_SET_DOMAIN: {
		struct drm_i915_gem_set_domain *set_domain = argp;
		struct ugem_bo *bo = get_bo(set_domain->handle);

		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_SET_DOMAIN\n");

		bo->read_domains |= set_domain->read_domains;
		bo->write_domain |= set_domain->write_domain;
			
		return 0;
	}

	case DRM_IOCTL_I915_GEM_SW_FINISH:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_SW_FINISH\n");
		return 0;

	case DRM_IOCTL_I915_GEM_SET_TILING: {
		struct drm_i915_gem_set_tiling *set_tiling = argp;
		struct ugem_bo *bo = get_bo(set_tiling->handle);

		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_SET_TILING\n");

		bo->tiling_mode = set_tiling->tiling_mode;
		bo->stride = set_tiling->stride;

		return 0;
	}
			
	case DRM_IOCTL_I915_GEM_GET_TILING: {
		struct drm_i915_gem_get_tiling *get_tiling = argp;
		struct ugem_bo *bo = get_bo(get_tiling->handle);

		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_GET_TILING\n");

		get_tiling->tiling_mode = bo->tiling_mode;

		return 0;
	}

	case DRM_IOCTL_I915_GEM_GET_APERTURE: {
		struct drm_i915_gem_get_aperture *get_aperture = argp;
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_GET_APERTURE\n");
		get_aperture->aper_available_size = 4245561344; /* bdw gt3 */
		return 0;
	}
	case DRM_IOCTL_I915_GEM_MADVISE:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_MADVISE\n");
		return 0;

	case DRM_IOCTL_I915_GEM_WAIT:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_WAIT\n");
		return 0;
		
	case DRM_IOCTL_I915_GEM_CONTEXT_CREATE:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_CONTEXT_CREATE\n");
		return libc_ioctl(fd, request, argp);

	case DRM_IOCTL_I915_GEM_CONTEXT_DESTROY:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_CONTEXT_DESTROY\n");
		return libc_ioctl(fd, request, argp);


	case DRM_IOCTL_I915_REG_READ:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_REG_READ\n");
		return 0;
	case DRM_IOCTL_I915_GET_RESET_STATS:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GET_RESET_STATS\n");
		return 0;
	case DRM_IOCTL_I915_GEM_USERPTR:
		ksim_trace(TRACE_GEM, "DRM_IOCTL_I915_GEM_USERPTR\n");
		return 0;

	case DRM_IOCTL_GEM_CLOSE: {
		struct drm_gem_close *close = argp;
		struct ugem_bo *bo = &bos[close->handle];

		ksim_trace(TRACE_GEM, "DRM_IOCTL_GEM_CLOSE\n");
		if (!bo->kernel_handle)
			free(bo->data);
		bo->data = bo_free_list;
		bo_free_list = bo;

		return 0;
	}

	case DRM_IOCTL_GEM_OPEN: {
		struct drm_gem_open *open = argp;

		ksim_trace(TRACE_GEM, "DRM_IOCTL_GEM_OPEN\n");

		return -1;
	}

	case DRM_IOCTL_PRIME_FD_TO_HANDLE: {
		struct drm_prime_handle *prime = argp;

		ksim_trace(TRACE_GEM, "DRM_IOCTL_PRIME_FD_TO_HANDLE\n");

		ret = libc_ioctl(fd, request, argp);
		if (ret == 0) {
			off_t size;

			size = lseek(prime->fd, 0, SEEK_END);
			if (size == -1)
				error(-1, errno, "failed to get prime bo size\n");
		}

		return ret;
	}

	case DRM_IOCTL_PRIME_HANDLE_TO_FD: {
		struct drm_prime_handle *prime_handle = argp;
		struct ugem_bo *bo = get_bo(prime_handle->handle);
		int ret;

		ksim_trace(TRACE_GEM, "DRM_IOCTL_PRIME_HANDLE_TO_FD\n");

		if (!bo->kernel_handle)
			create_kernel_bo(fd, bo);
		
		struct drm_prime_handle r = {
			.handle = bo->kernel_handle,
			.flags = prime_handle->flags
		};

		ret = libc_ioctl(fd, request, &r);

		prime_handle->fd = r.fd;

		return ret;
	}

	default:
		ksim_trace(TRACE_GEM, "unhandled ioctl 0x%x\n", _IOC_NR(request));

		return 0;
	}
}

__attribute__ ((visibility ("default"))) void *
mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
	if (fd == -1 || fd != drm_fd)
		return libc_mmap(addr, length, prot, flags, fd, offset);

	struct ugem_bo *bo = (void *) offset;

	ksim_assert(length <= bo->size);
	ksim_trace(TRACE_GEM, "mmap on drm fd, bo %d\n", bo - bos);

	return bo->data;
}

__attribute__ ((visibility ("default"))) int
munmap(void *addr, size_t length)
{
	/* Argh, no good way to know if we're unmapping a bo. */

	return libc_munmap(addr, length);
}

static bool
is_prefix(const char *arg, const char *prefix, const char **value)
{
	int l = strlen(prefix);

	if (strncmp(arg, prefix, l) == 0 && (arg[l] == '\0' || arg[l] == '=')) {
		if (arg[l] == '=')
			*value = arg + l + 1;
		else
			*value = NULL;

		return true;
	}

	return false;
}

static const struct { const char *name; uint32_t flag; } debug_tags[] = {
	{ "debug",	TRACE_DEBUG },
	{ "spam",	TRACE_SPAM },
	{ "warn",	TRACE_WARN },
	{ "gem",	TRACE_GEM },
	{ "cs",		TRACE_CS },
	{ "vf",		TRACE_VF },
	{ "vs",		TRACE_VS },
	{ "ps",		TRACE_VS },
	{ "eu",		TRACE_EU },
	{ "stub",	TRACE_STUB },
	{ "all",	~0 },
};

uint32_t trace_mask = TRACE_WARN | TRACE_STUB;
FILE *trace_file;
char *framebuffer_filename;

static void
parse_trace_flags(const char *value)
{
	for (uint32_t i = 0, start = 0; ; i++) {
		if (value[i] != ',' && value[i] != '\0')
			continue;
		for (uint32_t j = 0; j < ARRAY_LENGTH(debug_tags); j++) {
			if (strlen(debug_tags[j].name) == i - start &&
			    memcmp(debug_tags[j].name, &value[start], i - start) == 0) {
				trace_mask |= debug_tags[j].flag;
			}
		}
		if (value[i] == '\0')
			break;
		start = i + 1;
	}
}

static void __attribute__ ((constructor))
init(void)
{
	const char *args = getenv("KSIM_ARGS");
	const char *value;
	char buffer[256];

	trace_file = stdout;
	for (uint32_t i = 0, start = 0; args[i]; i++) {
		if (args[i] == ';') {

			if (i - start + 1 > sizeof(buffer)) {
				ksim_trace(TRACE_WARN, "arg too long: %.*s\n",
					   i - start, &args[start]);
				continue;
			}
			memcpy(buffer, &args[start], i - start);
			buffer[i - start] = '\0';

			if (is_prefix(buffer, "quiet", &value)) {
				trace_mask = 0;
			} else if (is_prefix(buffer, "framebuffer", &value))  {
				if (value)
					framebuffer_filename = strdup(value);
				else
					framebuffer_filename = strdup("fb.png");
			} else if (is_prefix(buffer, "file", &value))  {
				trace_file = fopen(value, "w");
				if (trace_file == NULL)
					error(-1, errno,
					      "ksim: failed to open output file %s\n", value);
			} else if (is_prefix(buffer, "trace", &value)) {
				if  (value == NULL)
					trace_mask |= ~0;
				else
					parse_trace_flags(value);
			}
			start = i + 1;
		}
	}

	libc_close = dlsym(RTLD_NEXT, "close");
	libc_ioctl = dlsym(RTLD_NEXT, "ioctl");
	libc_mmap = dlsym(RTLD_NEXT, "mmap");
	libc_munmap = dlsym(RTLD_NEXT, "munmap");
	if (libc_close == NULL || libc_ioctl == NULL ||
	    libc_mmap == NULL || libc_munmap == NULL)
		error(-1, 0, "ksim: failed to get libc ioctl or close\n");
}
