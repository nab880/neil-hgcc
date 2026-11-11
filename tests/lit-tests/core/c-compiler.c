// Core: hgcc runs the C compiler configure chose even when CC carries an
// option, as autoconf 2.73 makes it ("clang -std=gnu23"); the whole string
// used to be run as the program name. lit sets SST_HG_CC, which replaces CC,
// so this test unsets it. It can only fail where configure put an option
// into CC.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: env -u SST_HG_CC %hgcc -c %s -o %t.d/out.o
// RUN: test -f %t.d/out.o

int sst_hg_lit_c_compiler(int x) { return x + 1; }
