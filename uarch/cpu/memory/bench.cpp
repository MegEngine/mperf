#include "bench.h"

#if defined(__ANDROID__) || defined(ANDROID)
#include "malloc.h"
#define HAS_MEMALIGN
#elif !defined(_MSC_VER)
#define HAS_POSIX_MEMALIGN
#endif

namespace mperf {
double benchmp_simple(benchmp_f initialize, benchmp_f benchmark,
                      benchmp_f cleanup, int enough, int parallel, int warmup,
                      int repetitions, void* cookie) {
    double cost = 0.0f;
    if (initialize)
        (*initialize)(0, cookie);

    if (benchmark) {
        // warmup
        (*benchmark)(warmup, cookie);
        // execute
        WallTimer t;
        (*benchmark)(repetitions, cookie);
        cost = t.get_msecs() / 1000 / repetitions;
    }

    if (cleanup)
        (*cleanup)(0, cookie);

    return cost;
}

static volatile uint64_t use_result_dummy;

void keep_int(int result) {
    use_result_dummy += result;
}

void keep_pointer(void* result) {
    use_result_dummy += (long)result;
}

void* valloc_internal(size_t size) {
#ifdef HAS_POSIX_MEMALIGN
    void* ptr = NULL;
    if (posix_memalign(&ptr, getpagesize(), size)) {
        return NULL;
    }
    return ptr;
#elif defined(HAS_MEMALIGN)
    return memalign(getpagesize(), size);
#endif
    return nullptr;
}
}  // namespace mperf
