

AC_DEFUN([CHECK_CLANG], [

  AC_MSG_CHECKING([Clang flags])
  have_clang=`$pyexe $srcdir/config_tools/get_clang $CXX`
  AC_MSG_RESULT([$have_clang])

  if test "X$have_clang" = "Xyes"; then
    if test "X$darwin" = "Xtrue"; then
      AC_SUBST([LD_SO_FLAGS], ["-bundle -undefined dynamic_lookup"])
    else
      AC_SUBST([LD_SO_FLAGS], ["-shared -undefined dynamic_lookup"])
    fi
    AM_CONDITIONAL([HAVE_CLANG], true)
  else
    if test "X$darwin" = "Xtrue"; then
      AC_SUBST([LD_SO_FLAGS], ["-bundle"])
    else
      AC_SUBST([LD_SO_FLAGS], ["-shared"])
    fi
    AM_CONDITIONAL([HAVE_CLANG], false)
  fi
])

AC_DEFUN([CHECK_CLANG_LLVM], [
  AC_ARG_WITH(clang,
    [AS_HELP_STRING(
        [--with-clang],
        [Whether Clang libTooling is available for static analysis]
      )
    ],
    [
      clang=$withval
    ], 
    [
      clang=no
    ]
  )

  CLANG_INSTALL_DIR=$clang

  if test "$clang" != "no"; then
    if test "$clang" = "yes"; then
      AC_MSG_ERROR([--with-clang option requires an explicit path to Clang installation root: --with-clang=<CLANG_ROOT>])
    fi
    SAVE_LDFLAGS=$LDFLAGS
    SAVE_CPPFLAGS=$CPPFLAGS
    SAVE_CXXFLAGS=$CXXFLAGS
    CLANG_LDFLAGS=
    CLANG_CPPFLAGS=

    if test "$clang" != "yes"; then
      CLANG_LDFLAGS=-L$clang/lib 
      CLANG_CPPFLAGS=-I$clang/include
      LDFLAGS="$LDFLAGS $CLANG_LDFLAGS"
      CPPFLAGS="$CPPFLAGS $CLANG_CPPFLAGS"
    fi

    dnl SST_CXXFLAGS last so -std=c++17 wins over STD_CXXFLAGS (libTooling needs C++17).
    CXXFLAGS="$CXXFLAGS $STD_CXXFLAGS $SST_CXXFLAGS"

    AC_CHECK_HEADER([clang/AST/AST.h],
      found_clang=yes
      AC_SUBST(CLANG_LDFLAGS)
      AC_SUBST(CLANG_CPPFLAGS)
      AC_SUBST(CLANG_INSTALL_DIR)
      ,
      found_clang=no
      AC_MSG_ERROR([Unable to find valid Clang libTooling at specified location])
    )

    CPPFLAGS="$SAVE_CPPFLAGS"
    CXXFLAGS="$SAVE_CXXFLAGS"
    LDFLAGS="$SAVE_LDFLAGS"
  else
    found_clang=no
  fi

  if test "X$found_clang" = "Xno"; then
    AM_CONDITIONAL(HAVE_CLANG, false)
    AM_CONDITIONAL(CLANG_NEED_LIBCPP,false)
    AC_SUBST([CLANG_LIBTOOLING_CXX_RESOURCE_FLAGS], [""])
    AC_SUBST([USE_LIBCXX_FOR_AST], [False])
    AC_SUBST([AST_GNUXX_REMAP], [False])
    AC_SUBST([CLANG_MAJOR_VERSION], [0])
  else
    AM_CONDITIONAL(HAVE_CLANG, true)
    offset=`$pyexe $srcdir/config_tools/get_offsetof_macro $CXX`
    AC_MSG_CHECKING([offsetof macro definition])
    AC_MSG_RESULT([$offset])
    AC_DEFINE_UNQUOTED([OFFSET_OF_MACRO], [$offset], "the definition of the offsetof macro")

    # need to figure out clang absolute include paths
    # because clang libtooling is an abominiation hard-wired to relative paths
    
    CLANG_LIBTOOLING_SYSTEM_LIBS=`$clang/bin/llvm-config --system-libs`
    CLANG_LIBTOOLING_LIBS=`$clang/bin/llvm-config --libs`

    dnl LLVM's CMake sometimes bakes absolute paths to static archives
    dnl (e.g. /usr/lib/x86_64-linux-gnu/libzstd.a) into --system-libs.
    dnl Rewrite any /abs/path/libFOO.a -> -lFOO so the linker uses whatever
    dnl libFOO is reachable via LIBRARY_PATH / LD_LIBRARY_PATH / -L flags.
    CLANG_LIBTOOLING_SYSTEM_LIBS=`echo "$CLANG_LIBTOOLING_SYSTEM_LIBS" \
      | sed -E 's|/[^[:space:]]*/lib([A-Za-z0-9_+.-]+)\.a|-l\1|g'`

    LLVM_LIBS="$CLANG_LIBTOOLING_LIBS"
    LLVM_SYSTEM_LIBS="$CLANG_LIBTOOLING_SYSTEM_LIBS"
    LLVM_CPPFLAGS="$CLANG_CPPFLAGS"
    LLVM_LDFLAGS="$CLANG_LDFLAGS"

    dnl Use llvm-config rather than parsing 'clang --version', whose first line
    dnl varies by vendor (e.g. Homebrew prefixes 'Homebrew clang version ...').
    clang_major_version=`$clang/bin/llvm-config --version 2>/dev/null | cut -d '.' -f 1`
    case "$clang_major_version" in
      ''|*[[!0-9]]*) clang_major_version=0 ;;
    esac
    AC_SUBST([CLANG_MAJOR_VERSION], [$clang_major_version])
    if test "$clang_major_version" = "9"; then
      AM_CONDITIONAL(CLANG_NEED_LIBCPP,true)
    else
      AM_CONDITIONAL(CLANG_NEED_LIBCPP,false)
    fi
    dnl LLVM 16+ commonly ships libclang-cpp as the supported C++ API link
    dnl target. Linking individual libclang*.a archives can leave unresolved
    dnl symbols on Homebrew LLVM 22 and similar installs.
    if test "$clang_major_version" -ge 16; then
      CLANG_LINK_MONOLITHIC=yes
    else
      CLANG_LINK_MONOLITHIC=no
    fi
    AM_CONDITIONAL([CLANG_LINK_MONOLITHIC], [test "X$CLANG_LINK_MONOLITHIC" = "Xyes"])

    clang_compatibility=`$pyexe $srcdir/config_tools/check_clang_compatibility $CXX $clang $srcdir/config_tools/clang_version_test.cc $CXXFLAGS $STD_CXXFLAGS $SST_CXXFLAGS`

    if test "X$clang_compatibility" != "X"; then
      AC_MSG_ERROR([$clang_compatibility])
    fi

    AC_SUBST([CLANG_LIBTOOLING_LIBS])
    AC_SUBST([CLANG_LIBTOOLING_SYSTEM_LIBS])
    AC_SUBST([CLANG_LIBTOOLING_CXX_FLAGS], "`$pyexe $srcdir/config_tools/get_clang_includes $clang -E -v -std=c++1y -stdlib=libc++ -x c++`")
    AC_SUBST([CLANG_LIBTOOLING_C_FLAGS], "`$pyexe $srcdir/config_tools/get_clang_includes $clang -E -v`")
    dnl Resource-dir include only: avoids libc++ -I mixing with libstdc++ on Linux AST builds.
    resource_dir=`$clang/bin/clang -print-resource-dir`
    AC_SUBST([CLANG_LIBTOOLING_CXX_RESOURCE_FLAGS], ["-I${resource_dir}/include"])
    if test "X$darwin" = "Xtrue"; then
      AC_SUBST([USE_LIBCXX_FOR_AST], [True])
      AC_SUBST([AST_GNUXX_REMAP], [False])
    else
      AC_SUBST([USE_LIBCXX_FOR_AST], [False])
      AC_MSG_CHECKING([whether Clang needs gnu++ dialect for libstdc++ AST parse])
      astprobe=conftest_ast$$.cxx
      trap 'rm -f "$astprobe"' 0 1 2 13 15
      echo '#include <map>' > "$astprobe"
      if $clang/bin/clang++ -fsyntax-only -stdlib=libstdc++ -std=c++11 "$astprobe" >/dev/null 2>&1; then
        AC_SUBST([AST_GNUXX_REMAP], [False])
        AC_MSG_RESULT([no])
      else
        AC_SUBST([AST_GNUXX_REMAP], [True])
        AC_MSG_RESULT([yes])
      fi
      rm -f "$astprobe"
    fi
    clang_has_float128=`$pyexe $srcdir/config_tools/get_float_128 $clang/bin/clang++`
  fi

  if test "X$clang_has_float128" = "Xyes"; then
    AC_SUBST([have_float_128], [True])
  else
    AC_SUBST([have_float_128], [False])
  fi

  AC_SUBST([LLVM_LIBS])
  AC_SUBST([LLVM_SYSTEM_LIBS])
  AC_SUBST([LLVM_CPPFLAGS])
  AC_SUBST([LLVM_LDFLAGS])
])

