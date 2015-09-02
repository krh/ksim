module_LTLIBRARIES = ksim.la
moduledir = $(libdir)

ksim_la_LDFLAGS = -module -avoid-version -no-undefined
ksim_la_SOURCES = ksim.h gem.c command-streamer.c pipe.c eu.c formats.c
ksim_la_CPPFLAGS = $(DRM_CFLAGS)
ksim_la_CFLAGS = -fvisibility=hidden

bin_SCRIPTS = ksim
CLEANFILES = $(bin_SCRIPTS)
