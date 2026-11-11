// Core src2src: user `main` is renamed to `sst_hg_user_main_*`.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: %hgcc -c %s -o %t.d/out.o
// RUN: %FileCheck %s --input-file=%t.d/sst.pp.main-refactor.cc

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  return 0;
}

// CHECK: userSkeletonMain
// CHECK: userSkeletonMainInitFxn
