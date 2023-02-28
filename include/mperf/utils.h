/**
 * \file include/mperf/utils.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#pragma once
#include <stdint.h>
#include <cstdarg>
#include <memory>
#include <string>
#include <vector>
#include "mperf_build_config.h"

#ifndef mperf_trap
#define mperf_trap() __builtin_trap()
#endif

#define mperf_likely(v) __builtin_expect(bool(v), 1)
#define mperf_unlikely(v) __builtin_expect(bool(v), 0)

#define MPERF_MARK_USED_VAR(v) static_cast<void>(v)
#define MPERF_ATTRIBUTE_TARGET(simd) __attribute__((target(simd)))
#define MPERF_ALWAYS_INLINE __attribute__((always_inline))

//! mperf_assert
#define mperf_assert(expr, ...)                                               \
    do {                                                                      \
        if (mperf_unlikely(!(expr))) {                                        \
            ::mperf::__assert_fail__(__FILE__, __LINE__, __PRETTY_FUNCTION__, \
                                     #expr, ##__VA_ARGS__);                   \
        }                                                                     \
    } while (0)

#define mperf_assert_eq_dtype(lhs, rhs)                                       \
    do {                                                                      \
        mperf_assert(lhs.data_type == rhs.data_type, "%s is %s, %s is %s.",   \
                     #lhs, lhs.data_type.name(), #rhs, rhs.data_type.name()); \
    } while (0)

#if MPERF_WITH_MANGLING
#define mperf_mangle(x) ("")
#else
#define mperf_mangle(x) (x)
#endif

namespace mperf {
enum class LogLevel { DEBUG, INFO, WARN, ERROR };

typedef void (*LogHandler)(LogLevel level, const char* file, const char* func,
                           int line, const char* fmt, va_list ap);

/*!
 * \brief set the callback to receive all log messages
 *
 * Note: the log handler can be NULL (which is also the default value). In this
 * case, no log message would be recorded.
 *
 * \return original log handler
 */
LogHandler set_log_handler(LogHandler handler);

void __assert_fail__(const char* file, int line, const char* func,
                     const char* expr, const char* msg_fmt = nullptr, ...)
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((format(printf, 5, 6), noreturn))
#endif
        ;

std::string ssprintf(const char* fmt, ...)
        __attribute__((format(printf, 1, 2)));

/* ================ logging ================  */
#define LOG_TAG "MPERF"
#define mperf_log_debug(fmt...) \
    _mperf_do_log(::mperf::LogLevel::DEBUG, __FILE__, __func__, __LINE__, fmt)
#define mperf_log(fmt...) \
    _mperf_do_log(::mperf::LogLevel::INFO, __FILE__, __func__, __LINE__, fmt)
#define mperf_log_warn(fmt...) \
    _mperf_do_log(::mperf::LogLevel::WARN, __FILE__, __func__, __LINE__, fmt)
#define mperf_log_error(fmt...) \
    _mperf_do_log(::mperf::LogLevel::ERROR, __FILE__, __func__, __LINE__, fmt)

#if MPERF_WITH_LOGGING > 0
void __log__(LogLevel level, const char* file, const char* func, int line,
             const char* fmt, ...) __attribute__((format(printf, 5, 6)));

#define _mperf_do_log ::mperf::__log__
#else
#define _mperf_do_log(...) \
    do {                   \
    } while (0)
#endif  // MPERF_WITH_LOGGING

void stdout_log_handler(mperf::LogLevel level, const char* file,
                        const char* func, int line, const char* fmt,
                        va_list ap);

void info_log_handler(mperf::LogLevel level, const char* file, const char* func,
                      int line, const char* fmt, va_list ap);

std::vector<std::string> StrSplit(const std::string& str, char delim);
}  // namespace mperf