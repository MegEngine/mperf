// ref. HWCPipe project
#pragma once

namespace mperf {
class Value {
public:
    Value() : is_int_(true), int_(0), double_(0.0f) {}
    Value(long long value) : is_int_(true), int_(value) {}
    Value(double value) : is_int_(false), double_(value) {}

    template <typename T>
    T get() const {
        return is_int_ ? static_cast<T>(int_) : static_cast<T>(double_);
    }

    void set(long long value) {
        int_ = value;
        is_int_ = true;
    }

    void set(double value) {
        double_ = value;
        is_int_ = false;
    }

private:
    bool is_int_;
    long long int_{0};
    double double_{0.0};
};
}  // namespace mperf
