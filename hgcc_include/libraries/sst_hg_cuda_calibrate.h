/* sst_hg_cuda calibration helper (CUDA_PLAN.md §4.1, CUDA-IMPL-PLAN.md P4).
 *
 * The "calibrate-once production" path: compile your real .cu with your own
 * nvcc on a real GPU and -include this header. Wrap each launch in an
 * SST_HG_CALIBRATE(name, threads) scope; the RAII timer records the kernel's
 * wall-clock time with cudaEvents, and an atexit hook writes a
 * gpu_kernel_times JSON table (docs/gpu_calibration.schema.json). Hand that
 * file to the simulator via the platform parameter gpu_kernel_times and the
 * Mercury GPU library replays the measured time instead of the roofline
 * estimate -- no annotation, no re-derivation.
 *
 * This header is dependency-free and host-portable: the table/dump core needs
 * only the C++ standard library, so it can be unit-tested off-GPU. The GPU
 * timer (Scope, the macro) is compiled only under nvcc (__CUDACC__).
 *
 * Validation scope: the JSON accumulator/dump is unit-tested off-GPU and the
 * header is compile-verified under real nvcc, but the cudaEvent measurement
 * itself runs only on a real device -- it is exercised when you calibrate on
 * your GPU, not in this repo's host-only tests. Sanity-check the emitted times
 * on your first calibration run.
 *
 * Workflow:
 *   nvcc -include sst_hg_cuda_calibrate.h kernels.cu -o calib && ./calib
 *   # writes gpu_kernel_times.json (or $SST_HG_CALIBRATE_OUT)
 *
 * Contract: 'name' is the kernel's Itanium-mangled symbol -- the exact string
 * the rewriter passes as the first argument of sst_hg_cuda_launch (it is the
 * table key the loader matches on). Find it in the generated sst.pp.<tu>.cu,
 * or with `nm`/`cuobjdump -symbols`. 'threads' is the total launched thread
 * count, gridDim.{x,y,z} * blockDim.{x,y,z}.
 */
#ifndef SST_HG_CUDA_CALIBRATE_H
#define SST_HG_CUDA_CALIBRATE_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace sst_hg_calibrate {

/* kernel mangled name -> (total threads -> measured seconds samples). Samples
 * at the same thread count are averaged on dump, so repeated launches at one
 * size average out timing jitter. */
inline std::map<std::string, std::map<uint64_t, std::vector<double>>>& table() {
  static std::map<std::string, std::map<uint64_t, std::vector<double>>> t;
  return t;
}

inline void install_atexit();  // defined below; registers the dump hook

/* Record one measurement. Non-positive threads/seconds are dropped: the schema
 * requires both > 0, and a zero time usually means the launch was never timed
 * (e.g. an error) -- recording it would poison the loader's interpolation.
 * Recording anything arms the atexit dump, so the table is always written. */
inline void add(const char* kernel, uint64_t threads, double seconds) {
  install_atexit();
  if (kernel == nullptr || threads == 0 || seconds <= 0.0) return;
  table()[kernel][threads].push_back(seconds);
}

/* Write the accumulated table as gpu_kernel_times JSON. Destination is
 * $SST_HG_CALIBRATE_OUT, else ./gpu_kernel_times.json. One sample per distinct
 * thread count (the arithmetic mean of that count's measurements). Existing
 * contents are overwritten -- one run of one executable emits one table. */
inline void dump() {
  const char* path = std::getenv("SST_HG_CALIBRATE_OUT");
  if (path == nullptr || path[0] == '\0') path = "gpu_kernel_times.json";
  std::FILE* f = std::fopen(path, "w");
  if (f == nullptr) {
    std::fprintf(stderr, "sst_hg_calibrate: cannot open %s for writing\n", path);
    return;
  }
  std::fprintf(f, "{\n  \"version\": 1,\n  \"kernels\": {");
  bool firstKernel = true;
  for (const auto& kv : table()) {
    // Skip a kernel with no usable samples (schema requires minItems: 1).
    bool any = false;
    for (const auto& ts : kv.second) {
      if (!ts.second.empty()) { any = true; break; }
    }
    if (!any) continue;
    std::fprintf(f, "%s\n    \"%s\": [", firstKernel ? "" : ",",
                 kv.first.c_str());
    firstKernel = false;
    bool firstSample = true;
    for (const auto& ts : kv.second) {
      if (ts.second.empty()) continue;
      double sum = 0.0;
      for (double s : ts.second) sum += s;
      double mean = sum / static_cast<double>(ts.second.size());
      std::fprintf(f, "%s\n      { \"threads\": %llu, \"seconds\": %.12g }",
                   firstSample ? "" : ",",
                   static_cast<unsigned long long>(ts.first), mean);
      firstSample = false;
    }
    std::fprintf(f, "\n    ]");
  }
  std::fprintf(f, "\n  }\n}\n");
  std::fclose(f);
  if (firstKernel) {
    std::fprintf(stderr, "sst_hg_calibrate: no kernel samples recorded; wrote an "
                 "empty table to %s (did you wrap launches in SST_HG_CALIBRATE?)\n",
                 path);
  }
}

/* Register dump() to run at process exit. Idempotent. table() is forced into
 * existence first so its static is constructed before this atexit registration
 * -- the standard then guarantees the dump handler runs before the table's
 * destructor, not after (which would be a use-after-destruction in dump). */
inline void install_atexit() {
  static bool installed = false;
  if (!installed) {
    installed = true;
    (void)table();
    std::atexit(dump);
  }
}

}  // namespace sst_hg_calibrate

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace sst_hg_calibrate {

/* RAII kernel timer: records a cudaEvent at construction, and on destruction
 * records a second event, synchronizes, and files the elapsed time under
 * 'name'/'threads'. Scope it so the destructor fires right after the launch:
 *
 *   { SST_HG_CALIBRATE("_Z6vecAddPKfS0_Pfi", (uint64_t)grid * block);
 *     vecAdd<<<grid, block>>>(da, db, dc, n); }
 *
 * The destructor's cudaEventSynchronize is the only barrier added, so the
 * measured time is just this launch (plus its queue wait). Calibrate on a
 * single process; for multiple ranks set $SST_HG_CALIBRATE_OUT per rank. */
struct Scope {
  cudaEvent_t start_;
  cudaEvent_t stop_;
  const char* name_;
  uint64_t threads_;

  Scope(const char* name, uint64_t threads) : name_(name), threads_(threads) {
    install_atexit();
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
  }

  ~Scope() {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    add(name_, threads_, static_cast<double>(ms) * 1e-3);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
};

}  // namespace sst_hg_calibrate

#define SST_HG_CALIBRATE_CAT2(a, b) a##b
#define SST_HG_CALIBRATE_CAT(a, b) SST_HG_CALIBRATE_CAT2(a, b)
/* Time the launch that follows, in the current scope. */
#define SST_HG_CALIBRATE(name, threads)                       \
  ::sst_hg_calibrate::Scope SST_HG_CALIBRATE_CAT(__sst_hg_cal_, __LINE__)( \
      (name), (threads))

#endif  // __CUDACC__

#endif  // SST_HG_CUDA_CALIBRATE_H
