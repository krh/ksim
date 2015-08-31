module_LTLIBRARIES = ksim.la
moduledir = $(libdir)

ksim_la_LDFLAGS = -module -avoid-version -no-undefined
ksim_la_SOURCES = ksim.h gem.c command-streamer.c
ksim_la_CPPFLAGS = $(DRM_CFLAGS)

bin_SCRIPTS = ksim
CLEANFILES = $(bin_SCRIPTS)
