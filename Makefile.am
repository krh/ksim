SUBDIRS = libdisasm . test

bin_SCRIPTS = ksim

module_LTLIBRARIES = ksim-stub.la
moduledir = $(libdir)

ksim_stub_la_CFLAGS = -fvisibility=hidden -Wall -march=native -mrtm
ksim_stub_la_CPPFLAGS = $(KSIM_CFLAGS)
ksim_stub_la_LDFLAGS = -module -avoid-version -no-undefined
ksim_stub_la_LIBADD = libdisasm/libdisasm.la -ldl -lpng -lm -lpthread $(KSIM_LIBS)

ksim_stub_la_SOURCES = \
	stub.c \
	command-streamer.c \
	pipe.c \
	formats.c \
	thread.c \
	wm.c \
	compute.c \
	eu.c \
	avx-disasm.c \
	write-png.c \
	write-png.h
