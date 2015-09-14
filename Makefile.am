SUBDIRS = libdisasm .

module_LTLIBRARIES = ksim.la
moduledir = $(libdir)

ksim_la_LDFLAGS = -module -avoid-version -no-undefined
ksim_la_LIBADD = libdisasm/libdisasm.la -lpng
ksim_la_SOURCES = ksim.h gem.c command-streamer.c \
	pipe.c formats.c thread.c wm.c write-png.c write-png.h
ksim_la_CPPFLAGS = $(DRM_CFLAGS)
ksim_la_CFLAGS = -fvisibility=hidden -Wall

bin_SCRIPTS = ksim
CLEANFILES = $(bin_SCRIPTS)
