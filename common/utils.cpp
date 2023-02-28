/**
 * \file common/utils.cpp
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */

#include "mperf/utils.h"
#include <memory.h>
#include <unistd.h>
#include <cstdlib>
#include "mperf/exception.h"
#include "mperf_build_config.h"

using namespace mperf;

/* ===================== logging =====================  */
namespace {
LogHandler g_log_handler = stdout_log_handler;

std::string svsprintf(const char* fmt, va_list ap_orig) {
    int size = 100; /* Guess we need no more than 100 bytes */
    char* p;

    if ((p = (char*)malloc(size)) == nullptr)
        return "svsprintf: malloc failed";

    for (;;) {
        va_list ap;
        va_copy(ap, ap_orig);
        int n = vsnprintf(p, size, fmt, ap);
        va_end(ap);

        if (n < 0)
            return "svsprintf: vsnprintf failed";

        if (n < size) {
            std::string rst(p);
            free(p);
            return rst;
        }

        size = n + 1;

        char* np = (char*)realloc(p, size);
        if (!np) {
            free(p);
            return "svsprintf: realloc failed";
        } else
            p = np;
    }
}

LogLevel config_default_log_level() {
    auto default_level = LogLevel::ERROR;
    //! env to config LogLevel
    //!  DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3
    //! for example , export MPERF_LOG_LEVEL=0, means set LogLevel to
    //! DEBUG
    if (auto env = ::std::getenv("MPERF_LOG_LEVEL"))
        default_level = static_cast<LogLevel>(std::stoi(env));

    return default_level;
}

LogLevel min_log_level = config_default_log_level();

}  // anonymous namespace

#if MPERF_WITH_LOGGING > 0
void mperf::__log__(LogLevel level, const char* file, const char* func,
                    int line, const char* fmt, ...) {
    if (!g_log_handler)
        return;
    va_list ap;
    va_start(ap, fmt);
    g_log_handler(level, file, func, line, fmt, ap);
    va_end(ap);
}
#endif  // MPERF_WITH_LOGGING

LogHandler mperf::set_log_handler(LogHandler handler) {
    LogHandler ret = g_log_handler;
    g_log_handler = handler;
    return ret;
}

std::string mperf::ssprintf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::string rst = svsprintf(fmt, ap);
    va_end(ap);
    return rst;
}

void mperf::__assert_fail__(const char* file, int line, const char* func,
                            const char* expr, const char* msg_fmt, ...) {
    MPERF_MARK_USED_VAR(file);
    MPERF_MARK_USED_VAR(line);
    MPERF_MARK_USED_VAR(func);
    MPERF_MARK_USED_VAR(expr);
    std::string msg;
    if (msg_fmt) {
        va_list ap;
        va_start(ap, msg_fmt);
        msg = "\nextra message: ";
        msg.append(svsprintf(msg_fmt, ap));
        va_end(ap);
    }
    mperf_throw(MperfError, "assertion `%s' failed at %s:%d: %s%s", expr, file,
                line, func, msg.c_str());
}

namespace mperf {

static const char* Level2Str[]{"DEBUG", "INFO", "WARNING", "ERROR"};

void stdout_log_handler(mperf::LogLevel level, const char* file,
                        const char* func, int line, const char* fmt,
                        va_list ap) {
    constexpr int MAX_MSG_LEN = 1024;
    if (static_cast<int>(level) < static_cast<int>(min_log_level)) {
        return;
    }

    char msg[MAX_MSG_LEN];
    vsnprintf(msg, sizeof(msg), fmt, ap);
    if (file == nullptr && func == nullptr) {
        fprintf(stdout, "%s [%s] %s\n", LOG_TAG,
                mperf::Level2Str[static_cast<int>(level)], msg);
    } else
        fprintf(stdout, "%s [%s] %s @%s:%d %s\n", LOG_TAG,
                mperf::Level2Str[static_cast<int>(level)], msg, file, line,
                func);
}

void info_log_handler(mperf::LogLevel level, const char* file, const char* func,
                      int line, const char* fmt, va_list ap) {
    constexpr int MAX_MSG_LEN = 1024;
    if (static_cast<int>(level) != 1) {
        return;
    }

    char msg[MAX_MSG_LEN];
    vsnprintf(msg, sizeof(msg), fmt, ap);
    if (file == nullptr && func == nullptr) {
        fprintf(stdout, "%s [%s] %s\n", LOG_TAG,
                mperf::Level2Str[static_cast<int>(level)], msg);
    } else
        fprintf(stdout, "%s [%s] %s @%s:%d %s\n", LOG_TAG,
                mperf::Level2Str[static_cast<int>(level)], msg, file, line,
                func);
}

std::vector<std::string> StrSplit(const std::string& str, char delim) {
    if (str.empty())
        return {};
    std::vector<std::string> ret;
    size_t first = 0;
    size_t next = str.find(delim);
    for (; next != std::string::npos;
         first = next + 1, next = str.find(delim, first)) {
        ret.push_back(str.substr(first, next - first));
    }
    ret.push_back(str.substr(first));
    return ret;
}
}  // namespace mperf
