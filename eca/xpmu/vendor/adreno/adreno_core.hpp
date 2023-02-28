#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <mutex>

#include "adreno.h"

namespace adreno_userspace {

inline int adreno_activate_event(int device, unsigned int event_group,
                                      unsigned int event_selector) {
    struct event_activate activate;
    memset(&activate, 0, sizeof(struct event_activate));
    activate.event_group = event_group;
    activate.event_selector = event_selector;

    int ret = ioctl(device, IOCTL_EVENTS_ACTIVATE, &activate);
    if (ret == -1) {
        perror("ioctl: get an available event failed");
    }

    return ret;
}

inline int adreno_deactivate_event(int device, unsigned int event_group,
                                        unsigned int event_selector) {
    struct event_deactivate deactivate;
    memset(&deactivate, 0, sizeof(struct event_deactivate));
    deactivate.event_group = event_group;
    deactivate.event_selector = event_selector;

    int ret = ioctl(device, IOCTL_EVENT_DEACTIVATE, &deactivate);
    if (ret == -1) {
        perror("ioctl: put an allocated event failed");
    }

    return ret;
}

inline int adreno_read_events(int device, unsigned int num_entries,
                                    struct adreno_event_read_entry* entries,
                                    uint64_t* values) {
    struct event_read read;
    memset(&read, 0, sizeof(struct event_read));
    read.num_entries = num_entries;
    read.entries = entries;
    int ret = ioctl(device, IOCTL_EVENTS_READ, &read);
    if (ret == -1) {
        perror("ioctl: read in the current value of an event set "
               "failed");
        return ret;
    }

    for (size_t i = 0; i < num_entries; ++i) {
        values[i] = entries[i].event_value;
    }

    return 0;
}

}  // namespace adreno_userspace
