/**
 * \file include/mperf/exception.h
 *
 * This file is part of mperf.
 *
 * \copyright Copyright (c) 2022-2023 Megvii Inc. All rights reserved.
 */
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include "mperf_build_config.h"

namespace mperf {

class MperfError : public std::exception {
protected:
    std::string m_msg;

public:
    MperfError(const std::string& msg) : m_msg(msg) { m_msg.append("\n"); }

    const char* what() const noexcept override { return m_msg.c_str(); }

    ~MperfError() noexcept = default;
};

bool has_uncaught_exception();

void __on_exception_throw__(const std::exception& exc)
        __attribute__((noreturn));

}  // namespace mperf

//! throw raw exception object
#define mperf_throw_raw(_exc...) throw _exc
//! try block
#define MPERF_TRY try
//! catch block
#define MPERF_CATCH(_decl, _stmt) catch (_decl) _stmt

//! used after try-catch block, like try-finally construct in python
#define MPERF_FINALLY(_stmt) \
    MPERF_CATCH(..., {       \
        _stmt;               \
        throw;               \
    })                       \
    _stmt

//! throw exception with given message
#define mperf_throw(_exc, _msg...) \
    mperf_throw_raw(_exc(mperf_mangle(::mperf::ssprintf(_msg))))

//! throw exception with given message if condition is true
#define mperf_throw_if(_cond, _exc, _msg...) \
    do {                                     \
        if (mperf_unlikely((_cond)))         \
            mperf_throw(_exc, _msg);         \
    } while (0)

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
