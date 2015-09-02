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
static bool verbose = false;

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
validate_execbuffer2(struct drm_i915_gem_execbuffer2 *execbuffer2)
{
	struct drm_i915_gem_exec_object2 *buffers = (void *) execbuffer2->buffers_ptr;
	uint32_t bound_count = 0;
	struct ugem_bo *bo;

	printf("validate execbuf2, next_offset %ld, gtt size %ld\n",
	       next_offset, gtt_size);
	
	ksim_assert((execbuffer2->batch_len & 7) == 0);
	ksim_assert(execbuffer2->num_cliprects == 0);
	ksim_assert(execbuffer2->DR1 == 0);
	ksim_assert(execbuffer2->DR4 == 0);
	ksim_assert((execbuffer2->flags & I915_EXEC_RING_MASK) == I915_EXEC_RENDER);

	for (uint32_t i = 0; i < execbuffer2->buffer_count; i++) {
		bo = get_bo(buffers[i].handle);
		printf("bo %d, size %ld at offset %08x\n",
		       buffers[i].handle, bo->size, bo->offset);
		if (bo->offset == 0 && next_offset + bo->size <= gtt_size) {
			uint64_t alignment = max_u64(buffers[i].alignment, 4096);

			bind_bo(bo, align_u64(next_offset, alignment));
			next_offset = bo->offset + bo->size;
			printf("  bound to %08x\n", bo->offset);
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

	start_batch_buffer(bo->data + execbuffer2->batch_start_offset);
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
		if (verbose)
			printf("[ksim: intercept drm ioctl on fd %d]\n", fd);
	}

	if (fd != drm_fd)
		return libc_ioctl(fd, request, argp);

	switch (request) {
	case DRM_IOCTL_I915_GETPARAM: {
		struct drm_i915_getparam *getparam = argp;

		ret = libc_ioctl(fd, request, argp);

		return ret;
	}

	case DRM_IOCTL_I915_SETPARAM:
		printf("set param, handle %d\n");
		return libc_ioctl(fd, request, argp);

	case DRM_IOCTL_I915_GEM_EXECBUFFER: {
		static bool once;
		if (!once) {
			fprintf(stderr, "ksim: "
				"application uses DRM_IOCTL_I915_GEM_EXECBUFFER, not handled\n");
			once = true;
		}
		return libc_ioctl(fd, request, argp);
	}

	case DRM_IOCTL_I915_GEM_EXECBUFFER2:
		validate_execbuffer2(argp);

		return 0;

	case DRM_IOCTL_I915_GEM_BUSY:
		printf("busy\n");
		return 0;
			
	case DRM_IOCTL_I915_GEM_SET_CACHING:
		printf("set caching\n");
		return 0;

	case DRM_IOCTL_I915_GEM_GET_CACHING:
		printf("get caching\n");
		return 0;

	case DRM_IOCTL_I915_GEM_THROTTLE:
		printf("throttle\n");
		return 0;

	case DRM_IOCTL_I915_GEM_CREATE: {
		struct drm_i915_gem_create *create = argp;

		create->handle = add_bo(create->size);
		printf("new bo, handle %d, size %ld\n",
		       create->handle, create->size);

		return 0;
	}

	case DRM_IOCTL_I915_GEM_PREAD:
		printf("pread %d\n");
		return 0;

	case DRM_IOCTL_I915_GEM_PWRITE: {
		struct drm_i915_gem_pwrite *pwrite = argp;
		struct ugem_bo *bo = get_bo(pwrite->handle);
		printf("write: bo %d, offset %d, size %d, bo size %ld\n",
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
			
		ksim_assert(mmap->flags == 0);
		ksim_assert(mmap->offset + mmap->size > mmap->offset &&
			    mmap->offset + mmap->size <= bo->size);

		mmap->addr_ptr = (uint64_t) (bo->data + mmap->offset);

		return 0;
	}


	case DRM_IOCTL_I915_GEM_MMAP_GTT: {
		struct drm_i915_gem_mmap_gtt *map_gtt = argp;
		struct ugem_bo *bo = get_bo(map_gtt->handle);

		if (bo->tiling_mode != I915_TILING_NONE)
			ksim_warn("gtt mapping tiled buffer");

		map_gtt->offset = (uint64_t) bo;

		return 0;
	}

	case DRM_IOCTL_I915_GEM_SET_DOMAIN: {
		struct drm_i915_gem_set_domain *set_domain = argp;
		struct ugem_bo *bo = get_bo(set_domain->handle);

		printf("set_domain %d\n");

		bo->read_domains |= set_domain->read_domains;
		bo->write_domain |= set_domain->write_domain;
			
		return 0;
	}

	case DRM_IOCTL_I915_GEM_SW_FINISH:
		printf("sw_finish %d\n");
		return 0;

	case DRM_IOCTL_I915_GEM_SET_TILING: {
		struct drm_i915_gem_set_tiling *set_tiling = argp;
		struct ugem_bo *bo = get_bo(set_tiling->handle);

		bo->tiling_mode = set_tiling->tiling_mode;
		bo->stride = set_tiling->stride;

		return 0;
	}
			
	case DRM_IOCTL_I915_GEM_GET_TILING: {
		struct drm_i915_gem_get_tiling *get_tiling = argp;
		struct ugem_bo *bo = get_bo(get_tiling->handle);

		get_tiling->tiling_mode = bo->tiling_mode;

		return 0;
	}

	case DRM_IOCTL_I915_GEM_GET_APERTURE:
	case DRM_IOCTL_I915_GEM_MADVISE:
	case DRM_IOCTL_I915_GEM_WAIT:
		return 0;
		
	case DRM_IOCTL_I915_GEM_CONTEXT_CREATE:
	case DRM_IOCTL_I915_GEM_CONTEXT_DESTROY:
		return libc_ioctl(fd, request, argp);

	case DRM_IOCTL_I915_REG_READ:
	case DRM_IOCTL_I915_GET_RESET_STATS:
	case DRM_IOCTL_I915_GEM_USERPTR:
		return 0;

	case DRM_IOCTL_GEM_CLOSE: {
		struct drm_gem_close *close = argp;
		struct ugem_bo *bo = &bos[close->handle];
		free(bo->data);
		bo->data = bo_free_list;
		bo_free_list = bo;

		return 0;
	}

	case DRM_IOCTL_GEM_OPEN: {
		struct drm_gem_open *open = argp;

		ret = libc_ioctl(fd, request, argp);

		return ret;
	}

	case DRM_IOCTL_PRIME_FD_TO_HANDLE: {
		struct drm_prime_handle *prime = argp;

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

		if (!bo->kernel_handle) {
			struct drm_i915_gem_create create = { .size = bo->size };
			libc_ioctl(fd, DRM_IOCTL_I915_GEM_CREATE, &create);
			bo->kernel_handle = create.handle;

			struct drm_i915_gem_set_tiling set_tiling = {
				.handle = bo->kernel_handle,
				.tiling_mode = bo->tiling_mode,
				.stride = bo->stride,
			};

			libc_ioctl(fd, DRM_IOCTL_I915_GEM_SET_TILING, &set_tiling);

			struct drm_i915_gem_mmap mmap = {
				.handle = bo->kernel_handle,
				.offset = 0,
				.size = bo->size,
			};

			libc_ioctl(fd, DRM_IOCTL_I915_GEM_MMAP, &mmap);

			free(bo->data);
			bo->data = (void *) mmap.addr_ptr;
		}
		
		struct drm_prime_handle r = {
			.handle = bo->kernel_handle,
			.flags = prime_handle->flags
		};

		ret = libc_ioctl(fd, request, &r);

		prime_handle->fd = r.fd;

		return ret;
	}

	default:
		printf("unhandled ioctl 0x%x\n", _IOC_NR(request));
		return libc_ioctl(fd, request, argp);
	}
}

__attribute__ ((visibility ("default"))) void *
mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
	if (fd == -1 || fd != drm_fd)
		return libc_mmap(addr, length, prot, flags, fd, offset);

	printf("mmap on drm fd\n");

	struct ugem_bo *bo = (void *) offset;

	return bo->data;
}

__attribute__ ((visibility ("default"))) int
munmap(void *addr, size_t length)
{
	/* Argh, no good way to know if we're unmapping a bo. */

	return libc_munmap(addr, length);
}

static void __attribute__ ((constructor))
init(void)
{
	const char *args = getenv("KSIM_ARGS");

	for (uint32_t i = 0, start = 0; args[i]; i++) {
		if (args[i] == ';') {
			if (i - start == 7 &&
			    memcmp(&args[start], "verbose", 7) == 0)
				verbose = true;
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
