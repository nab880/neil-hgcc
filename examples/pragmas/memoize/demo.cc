// Compile with: hg++ --memoize -c demo.cc
// Captures variable types used in the next statement for memoization analysis.

int sst_hg_demo_x = 0;
int sst_hg_demo_y = 0;

void sst_hg_demo_memoize() {
#pragma sst memoize variables(sst_hg_demo_x,sst_hg_demo_y)
  sst_hg_demo_x += sst_hg_demo_y;
}
