AM_CFLAGS = \
	-Wall -std=c99 -march=native -mrtm \
	-fvisibility=hidden

AM_CPPFLAGS = -I$(top_srcdir)/intel

noinst_LTLIBRARIES = libdisasm.la

mesa_sources = \
	brw_device_info.h \
	brw_defines.h \
	brw_disasm.c \
	brw_eu.h \
	brw_eu_compact.c \
	brw_inst.h \
	brw_structs.h \
	brw_reg.h \
	brw_packed_float.c

libdisasm_la_SOURCES = \
	$(mesa_sources) \
	brw_context.h \
	gen_disasm.h \
	disasm.c

update :
	cp $(addprefix $(MESA)/src/mesa/drivers/dri/i965/,$(mesa_sources)) .
