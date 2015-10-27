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

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <error.h>
#include <errno.h>
#include <poll.h>
#include <wait.h>
#include <sys/socket.h>

#include "ksim.h"

#define KSIM_STUB_PATH ".libs/ksim-stub.so"

static int socket_fd;

static void
load_client(int argc, char *argv[], int mfd, int s)
{
	const char *current = getenv("LD_PRELOAD");
	char preload[1024], string[16];
	int child_mfd, child_s;

	if (current) {
		snprintf(preload, sizeof(preload),
			 KSIM_STUB_PATH ":%s", current);
		setenv("LD_PRELOAD", preload, 1);
	} else {
		setenv("LD_PRELOAD", KSIM_STUB_PATH, 1);
	}

	/* dup the to get non-CLOEXEC ones we can pass to the client. */
	child_mfd = dup(mfd);
	if (child_mfd == -1)
		error(EXIT_FAILURE, errno, "failed to dup mfd");
	child_s = dup(s);
	if (child_s == -1)
		error(EXIT_FAILURE, errno, "failed to dup socket fd");

	snprintf(string, sizeof(string), "%d,%d,%d",
		 child_mfd, child_s, trace_mask);
	setenv("KSIM_ARGS", string, 1);

	if (execvp(argv[0], argv) == - 1)
		error(EXIT_FAILURE, errno, "failed to exec %s", argv[0]);
}

struct gem_bo {
	uint64_t offset;
	uint64_t size;
	void *map;
};

struct gtt_entry {
	uint32_t handle;
};

static struct gem_bo bos[1024];
static int memfd;

#define gtt_order 20
static const uint64_t gtt_size = 4096ul << gtt_order;
static struct gtt_entry gtt[1 << gtt_order];

static struct gem_bo *
get_bo(int handle)
{
	struct gem_bo *bo = &bos[handle];

	ksim_assert(bo->size > 0);
	ksim_assert(bo->map != NULL);

	return bo;
}

static void
bind_bo(struct gem_bo *bo, uint64_t offset)
{
	uint32_t num_pages = (bo->size + 4095) >> 12;
	uint32_t start_page = offset >> 12;

	ksim_assert(bo->size > 0);
	ksim_assert(offset < gtt_size);
	ksim_assert(offset + bo->size < gtt_size);

	bo->offset = offset;
	for (uint32_t p = 0; p < num_pages; p++)
		gtt[start_page + p].handle = bo - bos;
}

void *
map_gtt_offset(uint64_t offset, uint64_t *range)
{
	struct gtt_entry entry;
	struct gem_bo *bo;

	ksim_assert(offset < gtt_size);
	entry = gtt[offset >> 12];
	bo = get_bo(entry.handle);

	ksim_assert(bo->offset != NOT_BOUND && bo->size > 0);
	ksim_assert(bo->offset <= offset);
	ksim_assert(offset < bo->offset + bo->size);

	*range = bo->offset + bo->size - offset;

	return bo->map + (offset - bo->offset);
}

static void
send_message(struct message *m)
{
	write(socket_fd, m, sizeof *m);
}

static void
handle_gem_create(struct message *m)
{
	struct gem_bo *bo = &bos[m->handle];

	bo->offset = NOT_BOUND;
	bo->size = m->size;
	bo->map = mmap(NULL, bo->size, PROT_READ | PROT_WRITE,
		       MAP_SHARED, memfd, m->offset);
	ksim_assert(bo->map != MAP_FAILED);
}

static void
handle_gem_close(struct message *m)
{
	struct gem_bo *bo = get_bo(m->handle);

	munmap(bo->map, bo->size);
	bo->offset = NOT_BOUND;
	bo->size = 0;
}

static void
handle_gem_bind(struct message *m)
{
	struct gem_bo *bo = get_bo(m->handle);

	bind_bo(bo, m->offset);
}

static void
handle_gem_exec(struct message *m)
{
	start_batch_buffer(m->offset);
}

static void
handle_gem_set_domain(struct message *m)
{
	struct message r = {
		.type = MSG_GEM_REPLY
	};

	send_message(&r);
}

static int
handle_requests(void)
{
	struct iovec iov[1];
	int len, fd;
	struct msghdr msg;
	struct cmsghdr *cmsg;
	char cmsg_buffer[CMSG_LEN(sizeof(fd))];

	union {
		struct message m;
		char buffer[1024];
	} u;

	iov[0].iov_base = &u.buffer;
	iov[0].iov_len = sizeof(u.buffer);

	msg.msg_name = NULL;
	msg.msg_namelen = 0;
	msg.msg_iov = iov;
	msg.msg_iovlen = 1;
	msg.msg_control = cmsg_buffer;
	msg.msg_controllen = sizeof(cmsg_buffer);
	msg.msg_flags = 0;

	do {
		len = recvmsg(socket_fd, &msg, MSG_CMSG_CLOEXEC);
	} while (len < 0 && errno == EINTR);
	if (len == 0)
		return 0;
	if (len == -1)
		error(EXIT_FAILURE, errno, "read error from client");

	cmsg = CMSG_FIRSTHDR(&msg);
	if (cmsg &&
	    cmsg->cmsg_level == SOL_SOCKET &&
	    cmsg->cmsg_type == SCM_RIGHTS) {
		ksim_assert(cmsg->cmsg_len - CMSG_LEN(0) == sizeof(fd));
		fd = *(int *) CMSG_DATA(cmsg);
	} else {
		fd = -1;
	}

	ksim_assert(len >= sizeof(u.m));
	switch (u.m.type) {
	case MSG_GEM_CREATE:
		handle_gem_create(&u.m);
		break;
	case MSG_GEM_CLOSE:
		handle_gem_close(&u.m);
		break;
	case MSG_GEM_BIND:
		handle_gem_bind(&u.m);
		break;
	case MSG_GEM_EXEC:
		handle_gem_exec(&u.m);
		break;
	case MSG_GEM_SET_DOMAIN:
		handle_gem_set_domain(&u.m);
		break;
	}

	return len;
}

static void
print_help(FILE *file)
{
	fprintf(file,
		"Usage: ksim [OPTION]... [--] COMMAND ARGUMENTS\n"
		"\n"
		"Run COMMAND with ARGUMENTS and under the ksim simulator.\n"
		"\n"
		"  -o, --output=FILE           Output ksim messages to FILE.\n"
		"  -f, --framebuffer[=FILE]    Output render target 0 to FILE as png.\n"
		"      --trace[=TAGS]          Enable tracing for the given message tags.\n"
		"                                Valid tags are 'debug', 'spam', 'warn', 'gem',\n"
		"                                'cs', 'vf', 'vs', 'ps', 'eu', 'stub', 'all'.\n"
		"                                Default value is 'stub,warn'.  With no argument,\n"
		"                                turn on all tags.\n"
		"  -q, --quiet                 Disable all trace messages.\n"
		"  -t                          Use threads.\n"
		"      --help                  Display this help message and exit.\n"
		"\n");
}

uint32_t trace_mask = TRACE_WARN | TRACE_STUB;
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
	{ "all",	~0 },
};

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

int
main(int argc, char *argv[])
{
	pid_t child;
	int i, sv[2], status;
	const char *value;

	trace_file = stdout;

	if (!__builtin_cpu_supports("avx2"))
		error(EXIT_FAILURE, 0, "AVX2 instructions not available");

	for (i = 1; i < argc; i++) {
		if (is_prefix(argv[i], "--trace", &value)) {
			if (value == NULL)
				trace_mask = ~0;
			else
				parse_trace_flags(value);
		} else if (is_prefix(argv[i], "--framebuffer", &value)) {
			if (value)
				framebuffer_filename = strdup(value);
			else
				framebuffer_filename = strdup("fb.png");
		} else if (strcmp(argv[i], "--quiet") == 0 ||
			   strcmp(argv[i], "-q") == 0) {
			trace_mask = 0;
		} else if (strcmp(argv[i], "-t") == 0) {
			use_threads = true;
		} else if (strcmp(argv[i], "--help") == 0) {
			print_help(stdout);
			exit(EXIT_SUCCESS);
		} else if (strcmp(argv[i], "--") == 0) {
			i++;
			break;
		} else if (argv[i][0] == '-') {
			printf("ksim: Unknown option: %s\n\n", argv[i]);
			print_help(stdout);
			exit(EXIT_SUCCESS);
		} else {
			break;
		}
	}

	if (i == argc) {
		print_help(stdout);
		exit(EXIT_FAILURE);
	}

	memfd = memfd_create("ksim bo", MFD_CLOEXEC);
	ftruncate(memfd, MEMFD_INITIAL_SIZE);
	socketpair(AF_LOCAL, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sv);

	child = fork();
	if (child == -1)
		error(EXIT_FAILURE, errno, "fork failed");
	if (child == 0)
		load_client(argc - i, argv + i, memfd, sv[1]);
	close(sv[1]);

	socket_fd = sv[0];
	while (handle_requests() > 0)
		;

	wait(&status);

	return EXIT_SUCCESS;
}
