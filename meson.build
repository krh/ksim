project('ksim', 'c')

c = meson.get_compiler('c')
opcodes = c.find_library('opcodes')

libdrm = dependency('libdrm')
libdrm_intel = dependency('libdrm_intel')
libpng16 = dependency('libpng16')
threads = dependency('threads')
# This is the portable way of using mathlib, it does
# not exist on all platforms.
mathlib = c.find_library('m', required : false)

conf = configuration_data()
conf.set('prefix', get_option('prefix'))
conf.set('exec_prefix', '${prefix}')
conf.set('libdir', '${exec_prefix}/' + get_option('libdir'))

configure_file(input : 'ksim.in',
               output : 'ksim',
               install_dir : get_option('bindir'), 
               configuration : conf)

disasm_files = files(
	'external/brw_context.h',
	'external/brw_defines.h',
	'external/brw_disasm.c',
	'external/brw_eu_compact.c',
	'external/brw_eu.h',
	'external/brw_eu.c',
	'external/brw_inst.h',
	'external/brw_packed_float.c',
	'external/brw_reg.h',
	'external/gen_device_info.h')

ksim_files = files(
	'avx-builder.c',
	'avx-builder.h',
	'command-streamer.c',
	'compute.c',
	'eu.h',
	'eu.c',
	'formats.c',
	'gen9_pack.h',
	'ksim.h',
	'pipe.c',
	'render-cache.c',
	'sampler.c',
	'stub.c',
	'thread.c',
	'wm.c',
	'write-png.c',
	'write-png.h')

shared_library('ksim-stub',
	ksim_files, disasm_files,
	c_args : [ '-fvisibility=hidden', '-march=core-avx2', '-D_GNU_SOURCE',
		   '-DKSIM_BUILD_' + get_option('buildtype').to_upper() ],
	dependencies : [ libdrm, libdrm_intel, libpng16, opcodes, threads, mathlib ],
	link_args : [ '-ldl' ],
	name_prefix : '',
	install : true)

executable('cs-runner',
	files('test/cs-runner.c'),
	c_args : [ '-fvisibility=hidden', '-march=native', '-mrtm' ],
	dependencies : [ libdrm_intel, mathlib ],
	install : false)

avxbuilder_test = executable('test-avx-builder',
	files('avx-builder.c'),
	c_args : [ '-march=native', '-mrtm',  '-DTEST_AVX_BUILDER' ],
	dependencies : [ opcodes ],
	install : false)

test('avx-builder', avxbuilder_test)
