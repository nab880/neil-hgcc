// Core: hg++ selects skeletonize mode unless the user chose a mode. With no
// mode variable set, the compute pragma rewrites the loop, and so it does
// when a mode variable reads as 0 (as C's atoi reads it). SST_HG_SKELETONIZE=0
// keeps the loop, and so does another mode variable that reads as non-zero;
// ssthg_clang refuses two modes, so hg++ must not add its own then.

// RUN: rm -rf %t.d && mkdir -p %t.d/default %t.d/off %t.d/encap %t.d/encap0 %t.d/encap00 %t.d/puppet %t.d/shadow
// RUN: %hgcc -c %s -o %t.d/default/out.o
// RUN: %FileCheck --check-prefix=SKEL %s --input-file=%t.d/default/sst.pp.rewrite-mode.cc
// RUN: env SST_HG_SKELETONIZE=0 %hgcc -c %s -o %t.d/off/out.o
// RUN: %FileCheck --check-prefix=KEEP %s --input-file=%t.d/off/sst.pp.rewrite-mode.cc
// RUN: env SST_HG_ENCAPSULATE=1 %hgcc -c %s -o %t.d/encap/out.o 2>&1 | %FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: %FileCheck --check-prefix=KEEP %s --input-file=%t.d/encap/sst.pp.rewrite-mode.cc
// RUN: env SST_HG_ENCAPSULATE=0 %hgcc -c %s -o %t.d/encap0/out.o
// RUN: %FileCheck --check-prefix=SKEL %s --input-file=%t.d/encap0/sst.pp.rewrite-mode.cc
// RUN: env SST_HG_ENCAPSULATE=00 %hgcc -c %s -o %t.d/encap00/out.o
// RUN: %FileCheck --check-prefix=SKEL %s --input-file=%t.d/encap00/sst.pp.rewrite-mode.cc
// RUN: env SST_HG_PUPPETIZE=1 %hgcc -c %s -o %t.d/puppet/out.o 2>&1 | %FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: env SST_HG_SHADOWIZE=1 %hgcc -c %s -o %t.d/shadow/out.o 2>&1 | %FileCheck --allow-empty --check-prefix=QUIET %s

void sst_hg_lit_rewrite_mode() {
  int acc = 0;
#pragma sst compute
  for (int i = 0; i < 1000; ++i) {
    acc += i;
  }
  (void)acc;
}

// SKEL-LABEL: sst_hg_lit_rewrite_mode
// SKEL-NOT: acc += i
// SKEL: sst_hg_compute_detailed(
// KEEP-LABEL: sst_hg_lit_rewrite_mode
// KEEP: acc += i
// KEEP-NOT: sst_hg_compute_detailed(
// QUIET-NOT: input error
