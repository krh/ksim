SUBDIRS = libdisasm .

bin_PROGRAMS = ksim

ksim_SOURCES = loader.c command-streamer.c \
	pipe.c formats.c thread.c wm.c write-png.c write-png.h
ksim_CFLAGS = -fvisibility=hidden -Wall
ksim_LDADD = libdisasm/libdisasm.la -lpng

module_LTLIBRARIES = ksim-stub.la
moduledir = $(libdir)

ksim_stub_la_LDFLAGS = -module -avoid-version -no-undefined
ksim_stub_la_LIBADD = -ldl
ksim_stub_la_SOURCES = stub.c
ksim_stub_la_CPPFLAGS = $(DRM_CFLAGS)
ksim_stub_la_CFLAGS = -fvisibility=hidden -Wall
