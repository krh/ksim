SUBDIRS = . test

bin_SCRIPTS = ksim

module_LTLIBRARIES = ksim-stub.la
moduledir = $(libdir)

# These should be set from configure
mesadir = ../mesa
mesabuilddir = ../mesa/build

# We have to link to the actual .a file, since using the .la file will
# use --whole-archive and end up pulling in more dependencies (nir etc).
libi965_compiler = $(mesabuilddir)/src/mesa/drivers/dri/i965/.libs/libi965_compiler.a

ksim_stub_la_CFLAGS = -fvisibility=hidden -Wall -march=native -mrtm
ksim_stub_la_CPPFLAGS = $(KSIM_CFLAGS) -I$(mesadir)/src/intel
ksim_stub_la_LDFLAGS = -module -avoid-version -no-undefined
ksim_stub_la_LIBADD = $(libi965_compiler) -ldl -lpng -lm -lpthread $(KSIM_LIBS)

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
