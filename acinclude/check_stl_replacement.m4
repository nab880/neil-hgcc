
AC_DEFUN([CHECK_STL_REPLACEMENT_HEADERS], [
	use_replacements=$1	
#no matter what, we need the sys type header
AC_SUBST([SYS_TYPES_FULL_PATH], "`$pyexe $srcdir/config_tools/get_include_path $CXX sys/types.h $CPPFLAGS $CXXFLAGS`")
AC_CONFIG_FILES([
 replacements/libraries/pthread/sstmac_sys_types.h
])
enable_repl_headers=yes
if test "x$use_replacements" = "xyes"; then
  enable_repl_headers=yes
  AM_CONDITIONAL([USE_REPLACEMENT_HEADERS], [true])
else
  enable_repl_headers=no
  AM_CONDITIONAL([USE_REPLACEMENT_HEADERS], [false])
fi
AC_CONFIG_FILES([
  replacements/ssthg_pthread_clear.h
])

])
