#include "./common.h"

using namespace mperf;

namespace mperf {
void cpu_insts_gflops_latency() {
    aarch64();
    armv7();
    x86_avx();
    x86_sse();
}
}  // namespace mperf