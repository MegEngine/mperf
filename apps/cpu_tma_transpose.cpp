
#include <stdio.h>
#if defined __ANDROID__ || defined __linux__
#include <sched.h>
#if defined __ANDROID__
#include <dlfcn.h>
#endif
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>

#include "mperf/cpu_affinity.h"
#include "mperf/timer.h"
#include "mperf/tma/tma.h"

namespace {

template <typename T>
void v_trans_base(const void* src, int src_step, int height, int width,
                  int channel, void* dst, int dst_step) {
    const T* src_ptr = (const T*)src;
    T* dst_ptr = (T*)dst;
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            for (int k = 0; k < channel; k++) {
                dst_ptr[i * dst_step + j * channel + k] =
                        src_ptr[j * src_step + i * channel + k];
            }
        }
    }
}

/* ======================== Prefetch ======================== */
#if defined(__aarch64__) || defined(__arm__)
#ifdef __aarch64__
static inline void prefetch(const void* ptr) {
    asm volatile("PRFM PLDL1KEEP, [%0]" : : "r"(ptr));
}
static inline void prefetch_l2(const void* ptr) {
    asm volatile("PRFM PLDL2KEEP, [%0]" : : "r"(ptr));
}
static inline void prefetch_l3(const void* ptr) {
    asm volatile("PRFM PLDL3KEEP, [%0]" : : "r"(ptr));
}
static inline void prefetch_non_temporal(const void* ptr) {
    asm volatile("PRFM PLDL1STRM, [%0]" : : "r"(ptr));
}
static inline void prefetchw(const void* ptr) {
    asm volatile("PRFM PSTL1KEEP, [%0]" : : "r"(ptr));
}

#define ASM_PREFETCH(address) "PRFM PLDL1KEEP, " address "\n"
#else
static inline void prefetch(const void* ptr) {
    asm volatile("pld [%0]" ::"r"(ptr));
}
static inline void prefetch_l2(const void* ptr) {
    asm volatile("pld [%0]" ::"r"(ptr));
}
static inline void prefetch_l3(const void* ptr) {
    asm volatile("pld [%0]" ::"r"(ptr));
}
static inline void prefetch_non_temporal(const void* ptr) {
    asm volatile("pld [%0]" ::"r"(ptr));
}
static inline void prefetchw(const void* ptr) {
    (void)ptr;
}
#define ASM_PREFETCH(address) "PLD " address "\n"
#endif
static inline void prefetch_6x(const void* ptr) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[ptr]]")
                 ASM_PREFETCH("[%[ptr], #64]")
                 ASM_PREFETCH("[%[ptr], #128]")
                 ASM_PREFETCH("[%[ptr], #192]")
                 ASM_PREFETCH("[%[ptr], #256]")
                 ASM_PREFETCH("[%[ptr], #320]")
                 :
                 : [ptr] "r"(ptr)
                 : "memory");
    // clang-format on
}
static inline void prefetch_5x(const void* ptr) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[ptr]]")
                 ASM_PREFETCH("[%[ptr], #64]")
                 ASM_PREFETCH("[%[ptr], #128]")
                 ASM_PREFETCH("[%[ptr], #192]")
                 ASM_PREFETCH("[%[ptr], #256]")
                 :
                 : [ptr] "r"(ptr)
                 : "memory");
    // clang-format on
}
static inline void prefetch_4x(const void* ptr) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[ptr]]")
                 ASM_PREFETCH("[%[ptr], #64]")
                 ASM_PREFETCH("[%[ptr], #128]")
                 ASM_PREFETCH("[%[ptr], #192]")
                 :
                 : [ptr] "r"(ptr)
                 : "memory");
    // clang-format on
}
static inline void prefetch_3x(const void* ptr) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[ptr]]")
                 ASM_PREFETCH("[%[ptr], #64]")
                 ASM_PREFETCH("[%[ptr], #128]")
                 :
                 : [ptr] "r"(ptr)
                 : "memory");
    // clang-format on
}
static inline void prefetch_2x(const void* ptr) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[ptr]]")
                 ASM_PREFETCH("[%[ptr], #64]")
                 :
                 : [ptr] "r"(ptr)
                 : "memory");
    // clang-format on
}
static inline void prefetch_1x(const void* ptr) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[ptr]]") : : [ptr] "r"(ptr) : "memory");
    // clang-format on
}
#endif

#if defined(__SSE__)
static inline void prefetch(const void* ptr) {
    asm volatile("prefetcht0 %[p]" : : [p] "m"(*(const char*)ptr));
}
static inline void prefetch_l2(const void* ptr) {
    asm volatile("prefetcht1 %[p]" : : [p] "m"(*(const volatile char*)ptr));
}
static inline void prefetch_l3(const void* ptr) {
    asm volatile("prefetcht2 %[p]" : : [p] "m"(*(const volatile char*)ptr));
}
static inline void prefetch_non_temporal(const void* ptr) {
    asm volatile("prefetchnta %[p]" : : [p] "m"(*(const volatile char*)ptr));
}
static inline void prefetchw(const void* ptr) {
    asm volatile("PREFETCHW %[p]" : : [p] "m"(*(const char*)ptr));
}
static inline void prefetch_6x(const void* ptr) {
    prefetch(ptr);
    prefetch((const char*)ptr + 64);
    prefetch((const char*)ptr + 64 * 2);
    prefetch((const char*)ptr + 64 * 3);
    prefetch((const char*)ptr + 64 * 4);
    prefetch((const char*)ptr + 64 * 5);
}
static inline void prefetch_5x(const void* ptr) {
    prefetch(ptr);
    prefetch((const char*)ptr + 64);
    prefetch((const char*)ptr + 64 * 2);
    prefetch((const char*)ptr + 64 * 3);
    prefetch((const char*)ptr + 64 * 4);
}
static inline void prefetch_4x(const void* ptr) {
    prefetch(ptr);
    prefetch((const char*)ptr + 64);
    prefetch((const char*)ptr + 64 * 2);
    prefetch((const char*)ptr + 64 * 3);
}
static inline void prefetch_3x(const void* ptr) {
    prefetch(ptr);
    prefetch((const char*)ptr + 64);
    prefetch((const char*)ptr + 64 * 2);
}
static inline void prefetch_2x(const void* ptr) {
    prefetch(ptr);
    prefetch((const char*)ptr + 64);
}
static inline void prefetch_1x(const void* ptr) {
    prefetch(ptr);
}
#endif

#define PRE_L2_STEP 2

// Load & Store 32B
// INT
#if defined __x86_64__ || defined _M_X64
#ifdef __AVX2__
#include <immintrin.h>
template <bool align = false>
inline __m256i v256_load_32b(const uint32_t* ptr);
template <>
inline __m256i v256_load_32b<false>(const uint32_t* ptr) {
    return _mm256_loadu_si256((const __m256i*)ptr);
}
template <>
inline __m256i v256_load_32b<true>(const uint32_t* ptr) {
    return _mm256_load_si256((const __m256i*)ptr);
}

template <bool align = false>
inline void v256_store_32b(uint32_t* ptr, __m256i val);
template <>
inline void v256_store_32b<false>(uint32_t* ptr, __m256i val) {
    _mm256_storeu_si256((__m256i*)ptr, val);
}
template <>
inline void v256_store_32b<true>(uint32_t* ptr, __m256i val) {
    _mm256_store_si256((__m256i*)ptr, val);
}
inline void v256_store2_32b(uint32_t* p0, uint32_t* p1, __m256i val) {
    _mm_store_si128((__m128i*)p0, _mm256_extractf128_si256(val, 0));
    _mm_store_si128((__m128i*)p1, _mm256_extractf128_si256(val, 1));
}

// float
template <bool align = false>
inline __m256 v256_load_32b(const float* ptr);
template <>
inline __m256 v256_load_32b<false>(const float* ptr) {
    return _mm256_loadu_ps(ptr);
}
template <>
inline __m256 v256_load_32b<true>(const float* ptr) {
    return _mm256_load_ps(ptr);
}

template <bool align = false>
inline void v256_store_32b(float* ptr, __m256 val);
template <>
inline void v256_store_32b<false>(float* ptr, __m256 val) {
    _mm256_storeu_ps(ptr, val);
}
template <>
inline void v256_store_32b<true>(float* ptr, __m256 val) {
    _mm256_store_ps(ptr, val);
}
template <bool align = false>
inline void v256_store2_32b(float* p0, float* p1, __m256 val);
template <>
inline void v256_store2_32b<false>(float* p0, float* p1, __m256 val) {
    _mm_storeu_ps(p0, _mm256_extractf128_ps(val, 0));
    _mm_storeu_ps(p1, _mm256_extractf128_ps(val, 1));
}
template <>
inline void v256_store2_32b<true>(float* p0, float* p1, __m256 val) {
    _mm_store_ps(p0, _mm256_extractf128_ps(val, 0));
    _mm_store_ps(p1, _mm256_extractf128_ps(val, 1));
}
template <bool align>
inline __m256 v256_load3_32b(const float* p0, const float* p1);
template <>
inline __m256 v256_load3_32b<false>(const float* p0, const float* p1) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(p0)),
                                _mm_loadu_ps(p1), 1);
}
template <>
inline __m256 v256_load3_32b<true>(const float* p0, const float* p1) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(p0)),
                                _mm_load_ps(p1), 1);
}
#endif

#include <nmmintrin.h>
template <bool align>
inline __m128i v_load_32b(const uint32_t* ptr);
template <>
inline __m128i v_load_32b<false>(const uint32_t* ptr) {
    return _mm_loadu_si128((const __m128i*)ptr);
}
template <>
inline __m128i v_load_32b<true>(const uint32_t* ptr) {
    return _mm_load_si128((const __m128i*)ptr);
}

template <bool align>
inline void v_store_32b(uint32_t* ptr, __m128i val);
template <>
inline void v_store_32b<false>(uint32_t* ptr, __m128i val) {
#if 1
    _mm_storeu_si128((__m128i*)ptr, val);
#else
    _mm_stream_si128((__m128i*)ptr, val);
#endif
}
template <>
inline void v_store_32b<true>(uint32_t* ptr, __m128i val) {
#if 1
    _mm_store_si128((__m128i*)ptr, val);
#else
    _mm_stream_si128((__m128i*)ptr, val);
#endif
}

template <bool align = false>
static inline void v_trans_c1_8x8_32b(const void* src, void* dst, int src_step,
                                      int dst_step) {
// v_trans_base<uint32_t>(src, src_step, 8, 8, 1, dst, dst_step);
// return;
#ifndef __AVX2__  // else branch is faster
    const float* src_ptr = (const float*)src;
    float* dst_ptr = (float*)dst;
    __m256 a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;
#if 0  // avx2-algo0
    a0 = v256_load_32b<align>(src_ptr + 0 * src_step);
    a1 = v256_load_32b<align>(src_ptr + 1 * src_step);
    a2 = v256_load_32b<align>(src_ptr + 2 * src_step);
    a3 = v256_load_32b<align>(src_ptr + 3 * src_step);
    a4 = v256_load_32b<align>(src_ptr + 4 * src_step);
    a5 = v256_load_32b<align>(src_ptr + 5 * src_step);
    a6 = v256_load_32b<align>(src_ptr + 6 * src_step);
    a7 = v256_load_32b<align>(src_ptr + 7 * src_step);

#if 0  // avx2-algo0-0
    b0 = _mm256_unpacklo_ps(a0, a2);
    b1 = _mm256_unpacklo_ps(a1, a3);
    b2 = _mm256_unpackhi_ps(a0, a2);
    b3 = _mm256_unpackhi_ps(a1, a3);
    b4 = _mm256_unpacklo_ps(a4, a6);
    b5 = _mm256_unpacklo_ps(a5, a7);
    b6 = _mm256_unpackhi_ps(a4, a6);
    b7 = _mm256_unpackhi_ps(a5, a7);

    a0 = _mm256_unpacklo_ps(b0, b1);
    a1 = _mm256_unpackhi_ps(b0, b1);
    a2 = _mm256_unpacklo_ps(b2, b3);
    a3 = _mm256_unpackhi_ps(b2, b3);
    a4 = _mm256_unpacklo_ps(b4, b5);
    a5 = _mm256_unpackhi_ps(b4, b5);
    a6 = _mm256_unpacklo_ps(b6, b7);
    a7 = _mm256_unpackhi_ps(b6, b7);
#else  // avx2-algo0-1
    b0 = _mm256_unpacklo_ps(a0, a1);
    b1 = _mm256_unpackhi_ps(a0, a1);
    b2 = _mm256_unpacklo_ps(a2, a3);
    b3 = _mm256_unpackhi_ps(a2, a3);
    b4 = _mm256_unpacklo_ps(a4, a5);
    b5 = _mm256_unpackhi_ps(a4, a5);
    b6 = _mm256_unpacklo_ps(a6, a7);
    b7 = _mm256_unpackhi_ps(a6, a7);

    a0 = _mm256_shuffle_ps(b0, b2, _MM_SHUFFLE(1, 0, 1, 0));
    a1 = _mm256_shuffle_ps(b0, b2, _MM_SHUFFLE(3, 2, 3, 2));
    a2 = _mm256_shuffle_ps(b1, b3, _MM_SHUFFLE(1, 0, 1, 0));
    a3 = _mm256_shuffle_ps(b1, b3, _MM_SHUFFLE(3, 2, 3, 2));
    a4 = _mm256_shuffle_ps(b4, b6, _MM_SHUFFLE(1, 0, 1, 0));
    a5 = _mm256_shuffle_ps(b4, b6, _MM_SHUFFLE(3, 2, 3, 2));
    a6 = _mm256_shuffle_ps(b5, b7, _MM_SHUFFLE(1, 0, 1, 0));
    a7 = _mm256_shuffle_ps(b5, b7, _MM_SHUFFLE(3, 2, 3, 2));
#endif
    b0 = _mm256_permute2f128_ps(a0, a4, 0x20);
    b1 = _mm256_permute2f128_ps(a1, a5, 0x20);
    b2 = _mm256_permute2f128_ps(a2, a6, 0x20);
    b3 = _mm256_permute2f128_ps(a3, a7, 0x20);
    b4 = _mm256_permute2f128_ps(a0, a4, 0x31);
    b5 = _mm256_permute2f128_ps(a1, a5, 0x31);
    b6 = _mm256_permute2f128_ps(a2, a6, 0x31);
    b7 = _mm256_permute2f128_ps(a3, a7, 0x31);

    v256_store_32b<align>(dst_ptr + 0 * dst_step, b0);
    v256_store_32b<align>(dst_ptr + 1 * dst_step, b1);
    v256_store_32b<align>(dst_ptr + 2 * dst_step, b2);
    v256_store_32b<align>(dst_ptr + 3 * dst_step, b3);
    v256_store_32b<align>(dst_ptr + 4 * dst_step, b4);
    v256_store_32b<align>(dst_ptr + 5 * dst_step, b5);
    v256_store_32b<align>(dst_ptr + 6 * dst_step, b6);
    v256_store_32b<align>(dst_ptr + 7 * dst_step, b7);
#else  // avx2-algo-1 replace shuffles with blends
    a0 = v256_load3_32b<align>(src_ptr + 0 * src_step + 0,
                               src_ptr + 4 * src_step + 0);
    prefetch(src_ptr + 0 * src_step + 64);
    prefetch_l2(src_ptr + 0 * src_step + 64 * PRE_L2_STEP);
    prefetch(src_ptr + 4 * src_step + 64);
    prefetch_l2(src_ptr + 4 * src_step + 64 * PRE_L2_STEP);
    a1 = v256_load3_32b<align>(src_ptr + 1 * src_step + 0,
                               src_ptr + 5 * src_step + 0);
    prefetch(src_ptr + 1 * src_step + 64);
    prefetch_l2(src_ptr + 1 * src_step + 64 * PRE_L2_STEP);
    prefetch(src_ptr + 5 * src_step + 64);
    prefetch_l2(src_ptr + 5 * src_step + 64 * PRE_L2_STEP);
    a2 = v256_load3_32b<align>(src_ptr + 2 * src_step + 0,
                               src_ptr + 6 * src_step + 0);
    prefetch(src_ptr + 2 * src_step + 64);
    prefetch_l2(src_ptr + 2 * src_step + 64 * PRE_L2_STEP);
    prefetch(src_ptr + 6 * src_step + 64);
    prefetch_l2(src_ptr + 6 * src_step + 64 * PRE_L2_STEP);
    a3 = v256_load3_32b<align>(src_ptr + 3 * src_step + 0,
                               src_ptr + 7 * src_step + 0);
    prefetch(src_ptr + 3 * src_step + 64);
    prefetch_l2(src_ptr + 3 * src_step + 64 * PRE_L2_STEP);
    prefetch(src_ptr + 7 * src_step + 64);
    prefetch_l2(src_ptr + 7 * src_step + 64 * PRE_L2_STEP);
    a4 = v256_load3_32b<align>(src_ptr + 0 * src_step + 4,
                               src_ptr + 4 * src_step + 4);
    a5 = v256_load3_32b<align>(src_ptr + 1 * src_step + 4,
                               src_ptr + 5 * src_step + 4);
    a6 = v256_load3_32b<align>(src_ptr + 2 * src_step + 4,
                               src_ptr + 6 * src_step + 4);
    a7 = v256_load3_32b<align>(src_ptr + 3 * src_step + 4,
                               src_ptr + 7 * src_step + 4);

    b0 = _mm256_unpacklo_ps(a0, a1);
    b1 = _mm256_unpackhi_ps(a0, a1);
    b2 = _mm256_unpacklo_ps(a2, a3);
    b3 = _mm256_unpackhi_ps(a2, a3);
    b4 = _mm256_unpacklo_ps(a4, a5);
    b5 = _mm256_unpackhi_ps(a4, a5);
    b6 = _mm256_unpacklo_ps(a6, a7);
    b7 = _mm256_unpackhi_ps(a6, a7);

    __m256 v;
    // a0 = _mm256_shuffle_ps(b0,b2, 0x44);
    // a1 = _mm256_shuffle_ps(b0,b2, 0xEE);
    v = _mm256_shuffle_ps(b0, b2, 0x4E);
    a0 = _mm256_blend_ps(b0, v, 0xCC);
    v256_store_32b<align>(dst_ptr + 0 * dst_step, a0);
    a1 = _mm256_blend_ps(b2, v, 0x33);
    v256_store_32b<align>(dst_ptr + 1 * dst_step, a1);
    // a2 = _mm256_shuffle_ps(b1,b3, 0x44);
    // a3 = _mm256_shuffle_ps(b1,b3, 0xEE);
    v = _mm256_shuffle_ps(b1, b3, 0x4E);
    a2 = _mm256_blend_ps(b1, v, 0xCC);
    v256_store_32b<align>(dst_ptr + 2 * dst_step, a2);
    a3 = _mm256_blend_ps(b3, v, 0x33);
    v256_store_32b<align>(dst_ptr + 3 * dst_step, a3);
    // a4 = _mm256_shuffle_ps(b4,b6, 0x44);
    // a5 = _mm256_shuffle_ps(b4,b6, 0xEE);
    v = _mm256_shuffle_ps(b4, b6, 0x4E);
    a4 = _mm256_blend_ps(b4, v, 0xCC);
    v256_store_32b<align>(dst_ptr + 4 * dst_step, a4);
    a5 = _mm256_blend_ps(b6, v, 0x33);
    v256_store_32b<align>(dst_ptr + 5 * dst_step, a5);
    // a6 = _mm256_shuffle_ps(b5,b7, 0x44);
    // a7 = _mm256_shuffle_ps(b5,b7, 0xEE);
    v = _mm256_shuffle_ps(b5, b7, 0x4E);
    a6 = _mm256_blend_ps(b5, v, 0xCC);
    v256_store_32b<align>(dst_ptr + 6 * dst_step, a6);
    a7 = _mm256_blend_ps(b7, v, 0x33);
    v256_store_32b<align>(dst_ptr + 7 * dst_step, a7);
#endif
#else
    const uint32_t* src_ptr = (const uint32_t*)src;
    uint32_t* dst_ptr = (uint32_t*)dst;
    // prefetchw(dst_ptr + 0 * dst_step);
    // prefetchw(dst_ptr + 1 * dst_step);
    // prefetchw(dst_ptr + 2 * dst_step);
    // prefetchw(dst_ptr + 3 * dst_step);
    // prefetchw(dst_ptr + 4 * dst_step);
    // prefetchw(dst_ptr + 5 * dst_step);
    // prefetchw(dst_ptr + 6 * dst_step);
    // prefetchw(dst_ptr + 7 * dst_step);

    __m128i i00 = v_load_32b<align>(src_ptr + 0 * src_step + 0);
    __m128i i01 = v_load_32b<align>(src_ptr + 0 * src_step + 4);
    prefetch(src_ptr + 0 * src_step + 64);
    // prefetch_l2(src_ptr + 0 * src_step + 64 * PRE_L2_STEP);
    __m128i i10 = v_load_32b<align>(src_ptr + 1 * src_step + 0);
    __m128i i11 = v_load_32b<align>(src_ptr + 1 * src_step + 4);
    prefetch(src_ptr + 1 * src_step + 64);
    // prefetch_l2(src_ptr + 1 * src_step + 64 * PRE_L2_STEP);
    __m128i i20 = v_load_32b<align>(src_ptr + 2 * src_step + 0);
    __m128i i21 = v_load_32b<align>(src_ptr + 2 * src_step + 4);
    prefetch(src_ptr + 2 * src_step + 64);
    // prefetch_l2(src_ptr + 2 * src_step + 64 * PRE_L2_STEP);
    __m128i i30 = v_load_32b<align>(src_ptr + 3 * src_step + 0);
    __m128i i31 = v_load_32b<align>(src_ptr + 3 * src_step + 4);
    prefetch(src_ptr + 3 * src_step + 64);
    // prefetch_l2(src_ptr + 3 * src_step + 64 * PRE_L2_STEP);
    __m128i i40 = v_load_32b<align>(src_ptr + 4 * src_step + 0);
    __m128i i41 = v_load_32b<align>(src_ptr + 4 * src_step + 4);
    prefetch(src_ptr + 4 * src_step + 64);
    // prefetch_l2(src_ptr + 4 * src_step + 64 * PRE_L2_STEP);
    __m128i i50 = v_load_32b<align>(src_ptr + 5 * src_step + 0);
    __m128i i51 = v_load_32b<align>(src_ptr + 5 * src_step + 4);
    prefetch(src_ptr + 5 * src_step + 64);
    // prefetch_l2(src_ptr + 5 * src_step + 64 * PRE_L2_STEP);
    __m128i i60 = v_load_32b<align>(src_ptr + 6 * src_step + 0);
    __m128i i61 = v_load_32b<align>(src_ptr + 6 * src_step + 4);
    prefetch(src_ptr + 6 * src_step + 64);
    // prefetch_l2(src_ptr + 6 * src_step + 64 * PRE_L2_STEP);
    __m128i i70 = v_load_32b<align>(src_ptr + 7 * src_step + 0);
    __m128i i71 = v_load_32b<align>(src_ptr + 7 * src_step + 4);
    prefetch(src_ptr + 7 * src_step + 64);
    // prefetch_l2(src_ptr + 7 * src_step + 64 * PRE_L2_STEP);

#if 1
    // Unpack 32 bit elements from:
    const __m128i a00 = _mm_unpacklo_epi32(i00, i10);
    const __m128i a10 = _mm_unpacklo_epi32(i20, i30);
    v_store_32b<align>(dst_ptr + 0 * dst_step + 0,
                       _mm_unpacklo_epi64(a00, a10));
    prefetchw(dst_ptr + 0 * dst_step + 64);
    v_store_32b<align>(dst_ptr + 1 * dst_step + 0,
                       _mm_unpackhi_epi64(a00, a10));
    prefetchw(dst_ptr + 1 * dst_step + 64);

    const __m128i a01 = _mm_unpacklo_epi32(i01, i11);
    const __m128i a11 = _mm_unpacklo_epi32(i21, i31);
    v_store_32b<align>(dst_ptr + 4 * dst_step + 0,
                       _mm_unpacklo_epi64(a01, a11));
    prefetchw(dst_ptr + 4 * dst_step + 64);
    v_store_32b<align>(dst_ptr + 5 * dst_step + 0,
                       _mm_unpackhi_epi64(a01, a11));
    prefetchw(dst_ptr + 5 * dst_step + 64);

    const __m128i a20 = _mm_unpackhi_epi32(i00, i10);
    const __m128i a30 = _mm_unpackhi_epi32(i20, i30);
    v_store_32b<align>(dst_ptr + 2 * dst_step + 0,
                       _mm_unpacklo_epi64(a20, a30));
    prefetchw(dst_ptr + 2 * dst_step + 64);
    v_store_32b<align>(dst_ptr + 3 * dst_step + 0,
                       _mm_unpackhi_epi64(a20, a30));
    prefetchw(dst_ptr + 3 * dst_step + 64);

    const __m128i a21 = _mm_unpackhi_epi32(i01, i11);
    const __m128i a31 = _mm_unpackhi_epi32(i21, i31);
    v_store_32b<align>(dst_ptr + 6 * dst_step + 0,
                       _mm_unpacklo_epi64(a21, a31));
    prefetchw(dst_ptr + 6 * dst_step + 64);
    v_store_32b<align>(dst_ptr + 7 * dst_step + 0,
                       _mm_unpackhi_epi64(a21, a31));
    prefetchw(dst_ptr + 7 * dst_step + 64);

    const __m128i a40 = _mm_unpacklo_epi32(i40, i50);
    const __m128i a50 = _mm_unpacklo_epi32(i60, i70);
    v_store_32b<align>(dst_ptr + 0 * dst_step + 4,
                       _mm_unpacklo_epi64(a40, a50));
    v_store_32b<align>(dst_ptr + 1 * dst_step + 4,
                       _mm_unpackhi_epi64(a40, a50));

    const __m128i a41 = _mm_unpacklo_epi32(i41, i51);
    const __m128i a51 = _mm_unpacklo_epi32(i61, i71);
    v_store_32b<align>(dst_ptr + 4 * dst_step + 4,
                       _mm_unpacklo_epi64(a41, a51));
    v_store_32b<align>(dst_ptr + 5 * dst_step + 4,
                       _mm_unpackhi_epi64(a41, a51));
    const __m128i a60 = _mm_unpackhi_epi32(i40, i50);

    const __m128i a70 = _mm_unpackhi_epi32(i60, i70);

    v_store_32b<align>(dst_ptr + 2 * dst_step + 4,
                       _mm_unpacklo_epi64(a60, a70));

    v_store_32b<align>(dst_ptr + 3 * dst_step + 4,
                       _mm_unpackhi_epi64(a60, a70));

    const __m128i a61 = _mm_unpackhi_epi32(i41, i51);
    const __m128i a71 = _mm_unpackhi_epi32(i61, i71);
    v_store_32b<align>(dst_ptr + 6 * dst_step + 4,
                       _mm_unpacklo_epi64(a61, a71));
    v_store_32b<align>(dst_ptr + 7 * dst_step + 4,
                       _mm_unpackhi_epi64(a61, a71));
#else
    // Unpack 32 bit elements from:
    const __m128i a00 = _mm_unpacklo_epi32(i00, i10);
    const __m128i a01 = _mm_unpacklo_epi32(i01, i11);
    const __m128i a10 = _mm_unpacklo_epi32(i20, i30);
    const __m128i a11 = _mm_unpacklo_epi32(i21, i31);
    const __m128i a20 = _mm_unpackhi_epi32(i00, i10);
    const __m128i a21 = _mm_unpackhi_epi32(i01, i11);
    const __m128i a30 = _mm_unpackhi_epi32(i20, i30);
    const __m128i a31 = _mm_unpackhi_epi32(i21, i31);
    const __m128i a40 = _mm_unpacklo_epi32(i40, i50);
    const __m128i a41 = _mm_unpacklo_epi32(i41, i51);
    const __m128i a50 = _mm_unpacklo_epi32(i60, i70);
    const __m128i a51 = _mm_unpacklo_epi32(i61, i71);
    const __m128i a60 = _mm_unpackhi_epi32(i40, i50);
    const __m128i a61 = _mm_unpackhi_epi32(i41, i51);
    const __m128i a70 = _mm_unpackhi_epi32(i60, i70);
    const __m128i a71 = _mm_unpackhi_epi32(i61, i71);

    // Unpack 64 bit elements to:
    v_store_32b<align>(dst_ptr + 0 * dst_step + 0,
                       _mm_unpacklo_epi64(a00, a10));
    v_store_32b<align>(dst_ptr + 0 * dst_step + 4,
                       _mm_unpacklo_epi64(a40, a50));
    v_store_32b<align>(dst_ptr + 1 * dst_step + 0,
                       _mm_unpackhi_epi64(a00, a10));
    v_store_32b<align>(dst_ptr + 1 * dst_step + 4,
                       _mm_unpackhi_epi64(a40, a50));
    v_store_32b<align>(dst_ptr + 2 * dst_step + 0,
                       _mm_unpacklo_epi64(a20, a30));
    v_store_32b<align>(dst_ptr + 2 * dst_step + 4,
                       _mm_unpacklo_epi64(a60, a70));
    v_store_32b<align>(dst_ptr + 3 * dst_step + 0,
                       _mm_unpackhi_epi64(a20, a30));
    v_store_32b<align>(dst_ptr + 3 * dst_step + 4,
                       _mm_unpackhi_epi64(a60, a70));
    v_store_32b<align>(dst_ptr + 4 * dst_step + 0,
                       _mm_unpacklo_epi64(a01, a11));
    v_store_32b<align>(dst_ptr + 4 * dst_step + 4,
                       _mm_unpacklo_epi64(a41, a51));
    v_store_32b<align>(dst_ptr + 5 * dst_step + 0,
                       _mm_unpackhi_epi64(a01, a11));
    v_store_32b<align>(dst_ptr + 5 * dst_step + 4,
                       _mm_unpackhi_epi64(a41, a51));
    v_store_32b<align>(dst_ptr + 6 * dst_step + 0,
                       _mm_unpacklo_epi64(a21, a31));
    v_store_32b<align>(dst_ptr + 6 * dst_step + 4,
                       _mm_unpacklo_epi64(a61, a71));
    v_store_32b<align>(dst_ptr + 7 * dst_step + 0,
                       _mm_unpackhi_epi64(a21, a31));
    v_store_32b<align>(dst_ptr + 7 * dst_step + 4,
                       _mm_unpackhi_epi64(a61, a71));
#endif
#endif
}

#else
#include <arm_neon.h>

static inline uint32x4x2_t vtrnq_u64_to_u32(const uint32x4_t& a0,
                                            const uint32x4_t& a1) {
    uint32x4x2_t b0;
    b0.val[0] = vcombine_u32(vget_low_u32(a0), vget_low_u32(a1));
    b0.val[1] = vcombine_u32(vget_high_u32(a0), vget_high_u32(a1));
    return b0;
}

template <bool align = false>
static inline void v_trans_c1_8x8_32b(const void* src, void* dst, int src_step,
                                      int dst_step) {
    const uint32_t* src_ptr = (const uint32_t*)src;
    uint32_t* dst_ptr = (uint32_t*)dst;
#ifdef __aarch64__
    uint32x4x2_t a0 = vld1q_u32_x2(src_ptr + 0 * src_step);
    prefetch(src_ptr + 0 * src_step + 64);
    prefetch_l2(src_ptr + 0 * src_step + 64 * PRE_L2_STEP);
    uint32x4x2_t a1 = vld1q_u32_x2(src_ptr + 1 * src_step);
    prefetch(src_ptr + 1 * src_step + 64);
    prefetch_l2(src_ptr + 1 * src_step + 64 * PRE_L2_STEP);
    uint32x4x2_t a2 = vld1q_u32_x2(src_ptr + 2 * src_step);
    prefetch(src_ptr + 2 * src_step + 64);
    prefetch_l2(src_ptr + 2 * src_step + 64 * PRE_L2_STEP);
    uint32x4x2_t a3 = vld1q_u32_x2(src_ptr + 3 * src_step);
    prefetch(src_ptr + 3 * src_step + 64);
    prefetch_l2(src_ptr + 3 * src_step + 64 * PRE_L2_STEP);
    uint32x4x2_t a4 = vld1q_u32_x2(src_ptr + 4 * src_step);
    prefetch(src_ptr + 4 * src_step + 64);
    prefetch_l2(src_ptr + 4 * src_step + 64 * PRE_L2_STEP);
    uint32x4x2_t a5 = vld1q_u32_x2(src_ptr + 5 * src_step);
    prefetch(src_ptr + 5 * src_step + 64);
    prefetch_l2(src_ptr + 5 * src_step + 64 * PRE_L2_STEP);
    uint32x4x2_t a6 = vld1q_u32_x2(src_ptr + 6 * src_step);
    prefetch(src_ptr + 6 * src_step + 64);
    prefetch_l2(src_ptr + 6 * src_step + 64 * PRE_L2_STEP);
    uint32x4x2_t a7 = vld1q_u32_x2(src_ptr + 7 * src_step);
    prefetch(src_ptr + 7 * src_step + 64);
    prefetch_l2(src_ptr + 7 * src_step + 64 * PRE_L2_STEP);

    // Swap 32 bit elements from:
    // a0: 00 01 02 03 04 05 06 07
    // a1: 10 11 12 13 14 15 16 17
    // a2: 20 21 22 23 24 25 26 27
    // a3: 30 31 32 33 34 35 36 37
    // a4: 40 41 42 43 44 45 46 47
    // a5: 50 51 52 53 54 55 56 57
    // a6: 60 61 62 63 64 65 66 67
    // a7: 70 71 72 73 74 75 76 77
    // to:
    // b0: 00 10 02 12 01 11 03 13
    // b1: 20 30 22 32 21 31 23 33
    // b2: 40 50 42 52 41 51 43 53
    // b3: 60 70 62 72 61 71 63 73
    // b4: 04 14 06 16 05 15 07 17
    // b5: 24 34 26 36 25 35 27 37
    // b6: 44 54 46 56 45 55 47 57
    // b7: 64 74 66 76 65 75 67 77
    const uint32x4x2_t b0 = vtrnq_u32(a0.val[0], a1.val[0]);
    const uint32x4x2_t b1 = vtrnq_u32(a2.val[0], a3.val[0]);
    const uint32x4x2_t b2 = vtrnq_u32(a4.val[0], a5.val[0]);
    const uint32x4x2_t b3 = vtrnq_u32(a6.val[0], a7.val[0]);
    const uint32x4x2_t b4 = vtrnq_u32(a0.val[1], a1.val[1]);
    const uint32x4x2_t b5 = vtrnq_u32(a2.val[1], a3.val[1]);
    const uint32x4x2_t b6 = vtrnq_u32(a4.val[1], a5.val[1]);
    const uint32x4x2_t b7 = vtrnq_u32(a6.val[1], a7.val[1]);
#else
    uint32x4_t a00 = vld1q_u32(src_ptr + 0 * src_step + 0);
    uint32x4_t a01 = vld1q_u32(src_ptr + 0 * src_step + 4);
    prefetch(src_ptr + 0 * src_step + 64);
    prefetch_l2(src_ptr + 0 * src_step + 64 * PRE_L2_STEP);
    uint32x4_t a10 = vld1q_u32(src_ptr + 1 * src_step + 0);
    uint32x4_t a11 = vld1q_u32(src_ptr + 1 * src_step + 4);
    prefetch(src_ptr + 1 * src_step + 64);
    prefetch_l2(src_ptr + 1 * src_step + 64 * PRE_L2_STEP);
    uint32x4_t a20 = vld1q_u32(src_ptr + 2 * src_step + 0);
    uint32x4_t a21 = vld1q_u32(src_ptr + 2 * src_step + 4);
    prefetch(src_ptr + 2 * src_step + 64);
    prefetch_l2(src_ptr + 2 * src_step + 64 * PRE_L2_STEP);
    uint32x4_t a30 = vld1q_u32(src_ptr + 3 * src_step + 0);
    uint32x4_t a31 = vld1q_u32(src_ptr + 3 * src_step + 4);
    prefetch(src_ptr + 3 * src_step + 64);
    prefetch_l2(src_ptr + 3 * src_step + 64 * PRE_L2_STEP);
    uint32x4_t a40 = vld1q_u32(src_ptr + 4 * src_step + 0);
    uint32x4_t a41 = vld1q_u32(src_ptr + 4 * src_step + 4);
    prefetch(src_ptr + 4 * src_step + 64);
    prefetch_l2(src_ptr + 4 * src_step + 64 * PRE_L2_STEP);
    uint32x4_t a50 = vld1q_u32(src_ptr + 5 * src_step + 0);
    uint32x4_t a51 = vld1q_u32(src_ptr + 5 * src_step + 4);
    prefetch(src_ptr + 5 * src_step + 64);
    prefetch_l2(src_ptr + 5 * src_step + 64 * PRE_L2_STEP);
    uint32x4_t a60 = vld1q_u32(src_ptr + 6 * src_step + 0);
    uint32x4_t a61 = vld1q_u32(src_ptr + 6 * src_step + 4);
    prefetch(src_ptr + 6 * src_step + 64);
    prefetch_l2(src_ptr + 6 * src_step + 64 * PRE_L2_STEP);
    uint32x4_t a70 = vld1q_u32(src_ptr + 7 * src_step + 0);
    uint32x4_t a71 = vld1q_u32(src_ptr + 7 * src_step + 4);
    prefetch(src_ptr + 7 * src_step + 64);
    prefetch_l2(src_ptr + 7 * src_step + 64 * PRE_L2_STEP);

    // Swap 32 bit elements from:
    const uint32x4x2_t b0 = vtrnq_u32(a00, a10);
    const uint32x4x2_t b1 = vtrnq_u32(a20, a30);
    const uint32x4x2_t b2 = vtrnq_u32(a40, a50);
    const uint32x4x2_t b3 = vtrnq_u32(a60, a70);
    const uint32x4x2_t b4 = vtrnq_u32(a01, a11);
    const uint32x4x2_t b5 = vtrnq_u32(a21, a31);
    const uint32x4x2_t b6 = vtrnq_u32(a41, a51);
    const uint32x4x2_t b7 = vtrnq_u32(a61, a71);
#endif
    // Swap 64 bit elements to:
    // c0: 00 10 20 30 02 12 22 32
    // c1: 01 11 21 31 03 13 23 33
    // c2: 40 50 60 70 42 52 62 72
    // c3: 41 51 61 71 43 53 63 73
    // c4: 04 14 24 34 06 16 26 36
    // c5: 05 15 25 35 07 17 27 37
    // c6: 44 54 64 74 46 56 66 76
    // c7: 45 55 65 75 47 57 67 77
    const uint32x4x2_t c0 = vtrnq_u64_to_u32(b0.val[0], b1.val[0]);
    const uint32x4x2_t c1 = vtrnq_u64_to_u32(b0.val[1], b1.val[1]);
    const uint32x4x2_t c2 = vtrnq_u64_to_u32(b2.val[0], b3.val[0]);
    const uint32x4x2_t c3 = vtrnq_u64_to_u32(b2.val[1], b3.val[1]);
    const uint32x4x2_t c4 = vtrnq_u64_to_u32(b4.val[0], b5.val[0]);
    const uint32x4x2_t c5 = vtrnq_u64_to_u32(b4.val[1], b5.val[1]);
    const uint32x4x2_t c6 = vtrnq_u64_to_u32(b6.val[0], b7.val[0]);
    const uint32x4x2_t c7 = vtrnq_u64_to_u32(b6.val[1], b7.val[1]);

    // Swap 128 bit elements to:
    // a0: 00 10 20 30 40 50 60 70
    // a1: 01 11 21 31 41 51 61 71
    // a2: 02 12 22 32 42 52 62 72
    // a3: 03 13 23 33 43 53 63 73
    // a4: 04 14 24 34 44 54 64 74
    // a5: 05 15 25 35 45 55 65 75
    // a6: 06 16 26 36 46 56 66 76
    // a7: 07 17 27 37 47 57 67 77
    vst1q_u32(dst_ptr + 0 * dst_step + 0, c0.val[0]);
    vst1q_u32(dst_ptr + 0 * dst_step + 4, c2.val[0]);
    vst1q_u32(dst_ptr + 1 * dst_step + 0, c1.val[0]);
    vst1q_u32(dst_ptr + 1 * dst_step + 4, c3.val[0]);
    vst1q_u32(dst_ptr + 2 * dst_step + 0, c0.val[1]);
    vst1q_u32(dst_ptr + 2 * dst_step + 4, c2.val[1]);
    vst1q_u32(dst_ptr + 3 * dst_step + 0, c1.val[1]);
    vst1q_u32(dst_ptr + 3 * dst_step + 4, c3.val[1]);
    vst1q_u32(dst_ptr + 4 * dst_step + 0, c4.val[0]);
    vst1q_u32(dst_ptr + 4 * dst_step + 4, c6.val[0]);
    vst1q_u32(dst_ptr + 5 * dst_step + 0, c5.val[0]);
    vst1q_u32(dst_ptr + 5 * dst_step + 4, c7.val[0]);
    vst1q_u32(dst_ptr + 6 * dst_step + 0, c4.val[1]);
    vst1q_u32(dst_ptr + 6 * dst_step + 4, c6.val[1]);
    vst1q_u32(dst_ptr + 7 * dst_step + 0, c5.val[1]);
    vst1q_u32(dst_ptr + 7 * dst_step + 4, c7.val[1]);
}
#endif
}  // namespace

//! This func used to compare with func generated by TVM.
template <int B3H, int B3W, int B2H, int B2W, int B1H, int B1W, int KH = 8,
          int KW = 8>
void transpose_32b_c1_base(const float* src, float* dst, int height, int width,
                           int channel, int sstride1, int dstride1,
                           int sstride2, int dstride2) {
    int h = 0;
    int w = 0;
    int BHS = B3H * B2H * B1H * KH;
    int BWS = B3W * B2W * B1W * KW;
    if (BHS <= height && BWS <= width) {
        int tile_total_hnum = height / BHS;
        int tile_total_wnum = width / BWS;
        // printf("out(%d,%d),L3(%d,%d),L2(%d,%d),L1(%d,%d)\n", tile_total_hnum,
        //        tile_total_wnum, B3H, B3W, B2H, B2W, B1H, B1W);

        for (int rh = 0; rh < tile_total_hnum; rh++) {
            for (int rw = 0; rw < tile_total_wnum; rw++) {
                for (int b3h = 0; b3h < B3H; b3h++) {
                    for (int b3w = 0; b3w < B3W; b3w++) {
                        for (int b2h = 0; b2h < B2H; b2h++) {
                            for (int b2w = 0; b2w < B2W; b2w++) {
                                for (int b1h = 0; b1h < B1H; b1h++) {
                                    for (int b1w = 0; b1w < B1W; b1w++) {
                                        // clang-format off
                                        v_trans_c1_8x8_32b<false>(
                                                src + (rh*B3H*B2H*B1H*KH + b3h*B2H*B1H*KH + b2h*B1H*KH + b1h*KH) * sstride1 + (rw*B3W*B2W*B1W*KW + b3w*B2W*B1W*KW + b2w*B1W*KW + b1w*KW) * sstride2,
                                                dst + (rw*B3W*B2W*B1W*KW + b3w*B2W*B1W*KW + b2w*B1W*KW + b1w*KW) * dstride1 + (rh*B3H*B2H*B1H*KH + b3h*B2H*B1H*KH + b2h*B1H*KH + b1h*KH) * dstride2,
                                                sstride1, dstride1);
                                        // clang-format on
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        h += tile_total_hnum;
        w += tile_total_wnum;
    }
    // process bottom tail region
    for (int h2 = h * BHS; h2 < height; h2++) {
        for (int w2 = 0; w2 < width; w2++) {
            for (int c = 0; c < channel; c++) {
                dst[w2 * dstride1 + h2 * dstride2 + c] =
                        src[h2 * sstride1 + w2 * sstride2 + c];
            }
        }
    }

    // process right tail region
    for (int h2 = 0; h2 < h * BHS; h2++) {
        for (int w2 = w * BWS; w2 < width; w2++) {
            for (int c = 0; c < channel; c++) {
                dst[w2 * dstride1 + h2 * dstride2 + c] =
                        src[h2 * sstride1 + w2 * sstride2 + c];
            }
        }
    }
}

#define ENABLE_TMA 1
#define MEMCPY 0

int main() {
    int dev_id = 0;
    if (set_cpu_thread_affinity_spec_core(dev_id)) {
        printf("faild set thread affinity(core %d)\n", dev_id);
    }

    // int height = 2048;
    // int width = 4096;

    int height = 1 << 13;
    int width = 1 << 13;

    size_t img_size = height * width * sizeof(float);

    size_t iter_num = 10;
    printf("the iter num is %zu\n", iter_num);

    float* src = new float[height * width];
    float* dst = new float[height * width];

    for (int idx = 0; idx < height * width; ++idx) {
        src[idx] = rand() % 255;
    }

    volatile uint8_t res = 0;

    // warmup
#if MEMCPY
    memcpy(dst, src, img_size);
#else
    transpose_32b_c1_base<2, 2, 1, 1, 2, 2, 8, 8>(src, dst, height, width, 1,
                                                  width, height, 1, 1);
#endif

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dst[x * height + y] = src[y * width + x];
        }
    }

#if ENABLE_TMA
#if defined(__aarch64__)
    mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::A55);
    // mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::A510);

    // clang-format off
    mpf_tma.init({"Frontend_Bound",
                      "Fetch_Latency", 
                          "ICache_Misses",
                          "ITLB_Misses",
                          "Predecode_Error",
                      "Fetch_Bandwidth",
                  "Bad_Speculation",
                      "Branch_Mispredicts",
                  "Backend_Bound",
                      "Memory_Bound",
                          "Load_Bound",
                              "Load_DTLB",
                              "Load_Cache",
                          "Store_Bound",
                              "Store_TLB",
                              "Store_Buffer",
                      "Core_Bound",
                          "Interlock_Bound",
                              "Interlock_AGU",
                              "Interlock_FPU",
                          "Core_Bound_Others",
                  "Retiring",
                      "LD_Retiring",
                      "ST_Retiring",
                      "DP_Retiring",
                      "ASE_Retiring",
                      "VFP_Retiring",
                      "PC_Write_Retiring",
                          "BR_IMMED_Retiring",
                          "BR_RETURN_Retiring",
                          "BR_INDIRECT_Retiring"});
    // clang-format on
#else
    mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::HSX_SERVER);
    mpf_tma.init(
            {"Frontend_Bound", "Bad_Speculation", "Backend_Bound", "Retiring"});
#endif

    size_t gn = mpf_tma.group_num();
    size_t uncore_evt_num = mpf_tma.uncore_events_num();
    printf("the gn and uncore_evt_nums %zu, %zu\n", gn, uncore_evt_num);
    for (size_t i = 0; i < gn; ++i) {
        mpf_tma.start(i);
#endif
        for (size_t j = 0; j < iter_num; ++j) {
#if MEMCPY
            memcpy(dst, src, img_size);
#else
        // 1, 4, 1, 1, 4, 2, 8, 8
        transpose_32b_c1_base<2, 2, 1, 1, 2, 2, 8, 8>(src, dst, height, width,
                                                      1, width, height, 1, 1);
#endif
            res += dst[0];
        }
#if ENABLE_TMA
        mpf_tma.sample_and_stop(iter_num);
    }

    // FIXME(hc): when the iter_numm greater than 5, we have ran into that
    // uncore event value became too big, so we call `sample(1)` very iter in
    // uncore event sample to walk around it
    for (size_t i = 0; i < uncore_evt_num; ++i) {
        mpf_tma.start_uncore(i);
#endif
        for (size_t j = 0; j < iter_num; ++j) {
#if MEMCPY
            memcpy(dst, src, img_size);
#else
        // 1, 4, 1, 1, 4, 2, 8, 8
        transpose_32b_c1_base<2, 2, 1, 1, 2, 2, 8, 8>(src, dst, height, width,
                                                      1, width, height, 1, 1);
#endif
            res += dst[0];
#if ENABLE_TMA
            mpf_tma.sample(1);
#endif
        }
#if ENABLE_TMA
        // mpf_tma.sample_and_stop(iter_num);
    }
    mpf_tma.deinit();
#endif
    printf("the res is %u\n", res);

    delete[] src;
    delete[] dst;

    return 0;
}
