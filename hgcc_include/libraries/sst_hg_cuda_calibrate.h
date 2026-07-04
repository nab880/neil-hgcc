/* sst_hg_cuda calibration helper: record kernel times and dump gpu_kernel_times JSON. */
#ifndef SST_HG_CUDA_CALIBRATE_H
#define SST_HG_CUDA_CALIBRATE_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace sst_hg_calibrate {

/* kernel mangled name -> (total threads -> measured seconds samples). */
inline std::map<std::string, std::map<uint64_t, std::vector<double>>>& table() {
  static std::map<std::string, std::map<uint64_t, std::vector<double>>> t;
  return t;
}

/* Guards table() so concurrent host threads can calibrate without corrupting
 * the map. Cheap at this scale and only taken on add/dump. */
inline std::mutex& table_mutex() {
  static std::mutex m;
  return m;
}

/* Emit `s` as a JSON string body (no surrounding quotes), escaping the
 * characters JSON forbids so an odd kernel name can't produce an invalid table
 * that the loader rejects far from the cause. */
inline std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (unsigned char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (c < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

inline void install_atexit();  // defined below; registers the dump hook

/* Drop non-positive samples; recording arms the atexit dump. */
inline void add(const char* kernel, uint64_t threads, double seconds) {
  install_atexit();
  if (kernel == nullptr || threads == 0 || seconds <= 0.0) return;
  std::lock_guard<std::mutex> lock(table_mutex());
  table()[kernel][threads].push_back(seconds);
}

/* Write gpu_kernel_times JSON to $SST_HG_CALIBRATE_OUT or ./gpu_kernel_times.json. */
inline void dump() {
  const char* path = std::getenv("SST_HG_CALIBRATE_OUT");
  if (path == nullptr || path[0] == '\0') path = "gpu_kernel_times.json";
  std::FILE* f = std::fopen(path, "w");
  if (f == nullptr) {
    std::fprintf(stderr, "sst_hg_calibrate: cannot open %s for writing\n", path);
    return;
  }
  // Hold the lock for the whole write so a concurrent add() can't reallocate a
  // vector mid-iteration.
  std::lock_guard<std::mutex> lock(table_mutex());
  std::fprintf(f, "{\n  \"version\": 1,\n  \"kernels\": {");
  bool firstKernel = true;
  for (const auto& kv : table()) {
    // Skip kernels with no usable samples.
    bool any = false;
    for (const auto& ts : kv.second) {
      if (!ts.second.empty()) { any = true; break; }
    }
    if (!any) continue;
    std::fprintf(f, "%s\n    \"%s\": [", firstKernel ? "" : ",",
                 json_escape(kv.first).c_str());
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

/* Idempotent atexit registration. Construct table() AND table_mutex() before
 * registering dump so both outlive it: atexit handlers and function-local static
 * dtors share one LIFO, so anything dump() touches must be constructed first or
 * it is destroyed before the exit-time dump runs (mutex lock on a dead mutex). */
inline void install_atexit() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    (void)table();
    (void)table_mutex();
    std::atexit(dump);
  });
}

}  // namespace sst_hg_calibrate

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace sst_hg_calibrate {

/* RAII cudaEvent timer; scope around the launch to measure. */
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
