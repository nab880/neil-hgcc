// Core: without --mpi, `--app-name` also names a C main. skeleton.h's
// USER_MAIN macro is C++ only, so ssthg_clang renames and registers a C main
// itself; it must register NAME exactly, not a name made from the source path,
// and derive a valid identifier for the renamed main from it, even when NAME
// starts with a digit or contains characters such as '+'.

// RUN: rm -rf %t.d && mkdir -p %t.d/flag %t.d/explicit %t.d/noflag
// RUN: %hgcc --app-name=2mm-lit+capp -c %s -o %t.d/flag/out.o
// RUN: %FileCheck %s --check-prefix=MAIN --input-file=%t.d/flag/sst.pp.app-name.c
// RUN: %FileCheck %s --check-prefix=REG --input-file=%t.d/flag/sstGlobals.pp.app-name.c.cpp

// As for C++, an explicit -Dssthg_app_name wins over --app-name.
// RUN: %hgcc -Dssthg_app_name=lit-explicit --app-name=2mm-lit+capp -c %s -o %t.d/explicit/out.o
// RUN: %FileCheck %s --check-prefix=EXPLICIT --input-file=%t.d/explicit/sstGlobals.pp.app-name.c.cpp

// Without the flag the name still comes from the source path, so it is not
// empty, and a SST_HG_APP_NAME set in the caller's environment is ignored.
// RUN: env SST_HG_APP_NAME=litleak %hgcc -c %s -o %t.d/noflag/out.o
// A name that cannot go into a string literal as it is, is refused.
// RUN: not %hgcc '--app-name=a"b' -c %s -o %t.d/bad.o 2>&1 | %FileCheck %s --check-prefix=BADNAME
// RUN: not %hgcc '-Dssthg_app_name=a\b' --app-name=ok -c %s -o %t.d/bad2.o 2>&1 | %FileCheck %s --check-prefix=BADDEF
// RUN: %FileCheck %s --check-prefix=NOFLAG --implicit-check-not=litleak --input-file=%t.d/noflag/sstGlobals.pp.app-name.c.cpp

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  return 0;
}

// MAIN: int sst_hg_user_main_2mm_lit_capp(
// REG: userSkeletonMainInitFxn("2mm-lit+capp",{{ *}}sst_hg_user_main_2mm_lit_capp)
// EXPLICIT: userSkeletonMainInitFxn("lit-explicit",{{ *}}sst_hg_user_main_lit_explicit)
// clang-format wraps the long path-derived name, so check only that the
// registration and exe_main_name are not empty strings.
// NOFLAG: sst_hg_initer_
// NOFLAG-NOT: ""
// NOFLAG: exe_main_name[] =
// NOFLAG-NOT: ""
// BADNAME: hgcc: the app name from --app-name, 'a"b', may not contain a double quote
// BADDEF: hgcc: the app name from -Dssthg_app_name, 'a\\b', may not contain
