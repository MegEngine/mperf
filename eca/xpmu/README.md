# Part of the code in this module comes from [HWCPipe](https://github.com/ARM-software/HWCPipe), and mperf has made some modifications.

## Introduction

XPMU is a simple and extensible interface for reading CPU and GPU PMU (Performance Monitor Unit) counters.

## Tips
* when you run into the error `Failed to get a file descriptor for CYCLES` on android platform, please try change to root user or 
    ``` bash
    echo -1 > /proc/sys/kernel/perf_event_paranoid
    ```
* when you want to get `Metric_DRAM_BW_Use` value on arm cpu, you need check there are `dsu` PMU in `/sys/bus/event_source/devices` directory, and you need get the type of `dsu` pmu by
    ``` bash
    cat > /sys/bus/event_source/devices/dsu/type
    ```
    and then, you use the type of `dsu` pmu as the `type` field of struct `perf_event_attr`.
