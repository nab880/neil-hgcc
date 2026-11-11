// Core: without --mpi, `--app-name` registers the app under that name by
// defining ssthg_app_name, so an unmodified main needs no edit.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc --app-name=sst_hg_lit_app -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.app-name.cc

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  return 0;
}

// CHECK: userSkeletonMainInitFxn({{ *}}"sst_hg_lit_app"
