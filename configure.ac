# -*- Autoconf -*-
#
#

# Version info
m4_define([HGCC_VERSION_TAG], 1)
m4_define([HGCC_SUBVERSION_TAG], 0)
m4_define([HGCC_SUBSUBVERSION_TAG], 0)

dnl Enable this for releases
m4_define([HGCC_SNAPSHOT_TAG])
dnl Enable this for beta releases
dnl m4_define([HGCC_SNAPSHOT_TAG],-beta1)
# Enable this for development snapshots (should generally be enabled)
#m4_define([HGCC_SNAPSHOT_TAG],-snapshot)

# Construct the version
m4_define([HGCC_ACVERSION_TAG],
          [HGCC_VERSION_TAG.HGCC_SUBVERSION_TAG.HGCC_SUBSUBVERSION_TAG])

dnl AC_PREREQ([2.71]) avoid this if possible
AC_INIT([hgcc],[HGCC_ACVERSION_TAG],[sst-help@sandia.gov])
AC_CONFIG_MACRO_DIR([acinclude])
AC_CONFIG_HEADERS(config.h)
AX_PREFIX_CONFIG_H(hgcc_config.h, HGCC)
AC_CANONICAL_TARGET

#if test "X$prefix" == "XNONE"; then
#  MY_PREFIX=$ac_default_prefix
#else
#  MY_PREFIX=`cd $prefix ; pwd`
#fi
#MY_ABS_SRCPATH=`cd $srcdir ; pwd`

# More version info.
AH_TEMPLATE([VERSION], [Major version number])
AH_TEMPLATE([SUBVERSION], [Minor version number])
AH_TEMPLATE([SUBSUBVERSION], [Point version number])
AC_DEFINE_UNQUOTED(VERSION, [HGCC_VERSION_TAG])
AC_DEFINE_UNQUOTED(SUBVERSION, [HGCC_SUBVERSION_TAG])
AC_DEFINE_UNQUOTED(SUBSUBVERSION, [HGCC_SUBSUBVERSION_TAG])
AC_SUBST(HGCC_LIBVERSION, [HGCC_LIBVERSION_TAG])

# Init automake
AM_INIT_AUTOMAKE([tar-pax -Wall -Werror foreign -Wno-portability subdir-objects])
AM_PROG_AS
m4_ifdef([AM_PROG_AR], [AM_PROG_AR])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

#-------------------------------------------------
# Environment 
#-------------------------------------------------

# Find out if this is running Mac OS X.
darwin=false
case $target_os in
  darwin*)  darwin=true ;;
esac
AM_CONDITIONAL([DARWIN],[test "$darwin" = true])

dnl Apple ld rejects -Wl,--start-group/--end-group (GNU ld); omit on Darwin.
if test "$darwin" = true; then
  LINK_GROUP_START=
  LINK_GROUP_END=
else
  LINK_GROUP_START="-Wl,--start-group"
  LINK_GROUP_END="-Wl,--end-group"
fi
AC_SUBST([LINK_GROUP_START])
AC_SUBST([LINK_GROUP_END])

# Check compilers and environment
dnl AC_PROG_RANLIB
AC_PROG_CC
AC_PROG_CXX
AC_LANG([C++])
AC_PROG_LN_S

LT_INIT

LT_INIT([shared disable-static])

SST_CPPFLAGS=
CHECK_MPI_PARALLEL()
CHECK_SST_CORE()
CHECK_SST_ELEMENTS()

AC_ARG_ENABLE([use-replacements],
  [AS_HELP_STRING([--enable-use-replacements], [Enable replacement STL headers])],
  [use_replacements=yes],
  [use_replacements=no]
)

CHECK_STL_REPLACEMENT_HEADERS($[use-replacements])
os=`$srcdir/config_tools/get_os`

if test "X$os" = "Xubuntu"; then
LDFLAGS="$LDFLAGS -Wl,--no-as-needed"
AM_CONDITIONAL([HAVE_UBUNTU], [true])
else
AM_CONDITIONAL([HAVE_UBUNTU], [false])
fi

CHECK_PYTHON()

CHECK_SDK()

CHECK_CLANG()

CHECK_GCC()

CHECK_DEFAULT_INCLUDES()

CHECK_CXX_STD()

#AM_CONDITIONAL(HAVE_REGEXP, false)

CHECK_CLANG_LLVM()

CHECK_SPACK()

#-------------------------------------------------
# Finalize 
#-------------------------------------------------

dnl Don't reorder options.
dnl AC_PRESERVE_HELP_ORDER

AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([hg++], [chmod +x hg++])
AC_CONFIG_FILES([hgcc], [chmod +x hgcc])
AC_CONFIG_FILES([hgccvars.py])

AC_OUTPUT

# --------- Print out a configuration summary. ---------
echo
echo " Configuration Summary:"
echo "---------------------------------------------------------------"
echo "$srcdir"
echo "Install prefix     $prefix"
echo "OS                 $os"
echo "C compiler         $CC"
echo "C++ compiler       $CXX"
echo "CFLAGS             $CFLAGS"
echo "CXXFLAGS           $CXXFLAGS $SST_CXXFLAGS $STD_CXXFLAGS"
echo "CPPFLAGS           $CPPFLAGS"
echo "LDFLAGS            $LDFLAGS"
echo "C++ Standard       $cxxstd"
echo "Replacements	 $USE_REPLACEMENT_HEADERS"
echo 

echo "Python             $pyexe"
echo "Clang Toolchain    $found_clang"
echo "---------------------------------------------------------------"

