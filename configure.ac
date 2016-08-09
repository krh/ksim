AC_PREREQ([2.60])
AC_INIT([ksim],
	[1.11],
	[https://what-bugs.com],
	[ksim])

# Check for libopcodes from binutils
AC_SEARCH_LIBS(init_disassemble_info, opcodes)

AC_CONFIG_SRCDIR([Makefile.am])
AC_USE_SYSTEM_EXTENSIONS
AC_GNU_SOURCE
AM_INIT_AUTOMAKE([1.12 foreign subdir-objects dist-xz])
AM_SILENT_RULES([yes])
AC_DISABLE_STATIC
AC_PROG_LIBTOOL

PKG_CHECK_MODULES(KSIM, [libdrm_intel >= 2.4.55 libdrm libpng16])

AC_CONFIG_FILES([Makefile libdisasm/Makefile test/Makefile])
AC_CONFIG_FILES([ksim], [chmod +x ksim])

AC_OUTPUT
