#ifndef FEATHER_UTIL_CPU_FEATURES_H
#define FEATHER_UTIL_CPU_FEATURES_H
#include <cstdlib>

namespace feather {
// All native x86 kernels currently share this ISA baseline. This function and
// the registration unit must themselves be compiled without those ISA flags.
inline bool HostSupportsX86Kernels() {
    const char* disabled = std::getenv("FEATHER_DISABLE_X86");
    if (disabled != nullptr && disabled[0] == '1') return false;
#if defined(FEATHER_WITH_X86) && (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#else
    return false;
#endif
}
}  // namespace feather
#endif
