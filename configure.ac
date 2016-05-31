AC_PREREQ([2.60])
AC_INIT([ksim],
	[1.11],
	[https://what-bugs.com],
	[ksim])

AC_CONFIG_SRCDIR([Makefile.am])
AC_USE_SYSTEM_EXTENSIONS
AC_GNU_SOURCE
AM_INIT_AUTOMAKE([1.12 foreign subdir-objects dist-xz])
AM_SILENT_RULES([yes])
AC_DISABLE_STATIC
AC_PROG_LIBTOOL

PKG_CHECK_MODULES(DRM, [libdrm_intel >= 2.4.55 libdrm])

AC_CONFIG_FILES([Makefile libdisasm/Makefile])
AC_CONFIG_FILES([ksim], [chmod +x ksim])

AC_OUTPUT
