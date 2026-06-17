/* Fixed-behavior sst_hg_cuda stub for spike probe 5. */
#include "hg_cuda.h"

#include <cstdio>

namespace {
/* Bump-allocated fake device cookies in a high invalid-ish range. */
const uint64_t kCookieBase = 0x4000000000000000ull;
uint64_t g_next_cookie = kCookieBase;
uint64_t g_cookie_end = kCookieBase;
uint64_t g_counter = 1; /* streams/events share a trivial id space */
}

extern "C" {

void* sst_hg_cuda_malloc(uint64_t bytes)
{
  void* p = (void*)g_next_cookie;
  uint64_t sz = (bytes + 255u) & ~255ull;
  g_next_cookie += sz ? sz : 256u;
  g_cookie_end = g_next_cookie;
  printf("sst_hg_cuda_malloc(%llu) -> %p\n", (unsigned long long)bytes, p);
  return p;
}

void sst_hg_cuda_free(void* dptr)
{
  printf("sst_hg_cuda_free(%p)\n", dptr);
}

int sst_hg_cuda_is_device_ptr(const void* p)
{
  uint64_t v = (uint64_t)p;
  return v >= kCookieBase && v < g_cookie_end;
}

void sst_hg_cuda_memcpy(void* dst, const void* src, uint64_t bytes, int kind,
                        void* stream)
{
  printf("sst_hg_cuda_memcpy(dst=%p src=%p bytes=%llu kind=%d stream=%p)\n",
         dst, src, (unsigned long long)bytes, kind, stream);
}

void sst_hg_cuda_launch(const char* kernelName,
                        uint32_t gx, uint32_t gy, uint32_t gz,
                        uint32_t bx, uint32_t by, uint32_t bz,
                        uint64_t shmemBytes, void* stream,
                        uint64_t flops, uint64_t intops,
                        uint64_t bytesRead, uint64_t bytesWritten)
{
  printf("sst_hg_cuda_launch(%s grid=%u,%u,%u block=%u,%u,%u shmem=%llu "
         "stream=%p perthread{f=%llu i=%llu r=%llu w=%llu})\n",
         kernelName, gx, gy, gz, bx, by, bz,
         (unsigned long long)shmemBytes, stream,
         (unsigned long long)flops, (unsigned long long)intops,
         (unsigned long long)bytesRead, (unsigned long long)bytesWritten);
}

void* sst_hg_cuda_stream_create(void)
{
  void* s = (void*)++g_counter;
  printf("sst_hg_cuda_stream_create() -> %p\n", s);
  return s;
}

void sst_hg_cuda_stream_destroy(void* s) { printf("sst_hg_cuda_stream_destroy(%p)\n", s); }
void sst_hg_cuda_stream_sync(void* s) { printf("sst_hg_cuda_stream_sync(%p)\n", s); }

void* sst_hg_cuda_event_create(void)
{
  void* e = (void*)++g_counter;
  printf("sst_hg_cuda_event_create() -> %p\n", e);
  return e;
}

void sst_hg_cuda_event_record(void* evt, void* stream)
{
  printf("sst_hg_cuda_event_record(%p, %p)\n", evt, stream);
}

void sst_hg_cuda_event_sync(void* evt) { printf("sst_hg_cuda_event_sync(%p)\n", evt); }

float sst_hg_cuda_event_elapsed_ms(void* start, void* stop)
{
  printf("sst_hg_cuda_event_elapsed_ms(%p, %p)\n", start, stop);
  return 0.0f;
}

void sst_hg_cuda_device_sync(void) { printf("sst_hg_cuda_device_sync()\n"); }
int sst_hg_cuda_get_device_count(void) { return 1; }
void sst_hg_cuda_set_device(int dev) { printf("sst_hg_cuda_set_device(%d)\n", dev); }

} /* extern "C" */
