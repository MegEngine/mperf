#pragma once

#include <stdint.h>
#include <unordered_map>

/* event groups */
#define EVENT_GROUP_CP 0x0
#define EVENT_GROUP_RBBM 0x1
#define EVENT_GROUP_PC 0x2
#define EVENT_GROUP_VFD 0x3
#define EVENT_GROUP_HLSQ 0x4
#define EVENT_GROUP_VPC 0x5
#define EVENT_GROUP_TSE 0x6
#define EVENT_GROUP_RAS 0x7
#define EVENT_GROUP_UCHE 0x8
#define EVENT_GROUP_TP 0x9
#define EVENT_GROUP_SP 0xA
#define EVENT_GROUP_RB 0xB
#define EVENT_GROUP_PWR 0xC
#define EVENT_GROUP_VBIF 0xD
#define EVENT_GROUP_VBIF_PWR 0xE
#define EVENT_GROUP_MH 0xF
#define EVENT_GROUP_PA_SU 0x10
#define EVENT_GROUP_SQ 0x11
#define EVENT_GROUP_SX 0x12
#define EVENT_GROUP_TCF 0x13
#define EVENT_GROUP_TCM 0x14
#define EVENT_GROUP_TCR 0x15
#define EVENT_GROUP_L2 0x16
#define EVENT_GROUP_VSC 0x17
#define EVENT_GROUP_CCU 0x18
#define EVENT_GROUP_LRZ 0x19
#define EVENT_GROUP_CMP 0x1A
#define EVENT_GROUP_ALWAYSON 0x1B
#define EVENT_GROUP_SP_PWR 0x1C
#define EVENT_GROUP_TP_PWR 0x1D
#define EVENT_GROUP_RB_PWR 0x1E
#define EVENT_GROUP_CCU_PWR 0x1F
#define EVENT_GROUP_UCHE_PWR 0x20
#define EVENT_GROUP_CP_PWR 0x21
#define EVENT_GROUP_GPMU_PWR 0x22
#define EVENT_GROUP_ALWAYSON_PWR 0x23
#define EVENT_GROUP_GLC 0x24
#define EVENT_GROUP_FCHE 0x25
#define EVENT_GROUP_MHUB 0x26
#define EVENT_GROUP_CUSTOM 0x27
#define EVENT_GROUP_MAX 0x28
#define EVENTS_NOT_USED 0xFFFFFFFF
#define EVENTS_BROKEN 0xFFFFFFFE

const std::unordered_map<std::string, int> group_name_to_id{
        {"EVENT_GROUP_CP", EVENT_GROUP_CP},
        {"EVENT_GROUP_RBBM", EVENT_GROUP_RBBM},
        {"EVENT_GROUP_PC", EVENT_GROUP_PC},
        {"EVENT_GROUP_VFD", EVENT_GROUP_VFD},
        {"EVENT_GROUP_HLSQ", EVENT_GROUP_HLSQ},
        {"EVENT_GROUP_VPC", EVENT_GROUP_VPC},
        {"EVENT_GROUP_TSE", EVENT_GROUP_TSE},
        {"EVENT_GROUP_RAS", EVENT_GROUP_RAS},
        {"EVENT_GROUP_UCHE", EVENT_GROUP_UCHE},
        {"EVENT_GROUP_TP", EVENT_GROUP_TP},
        {"EVENT_GROUP_SP", EVENT_GROUP_SP},
        {"EVENT_GROUP_RB", EVENT_GROUP_RB},
        {"EVENT_GROUP_PWR", EVENT_GROUP_PWR},
        {"EVENT_GROUP_VBIF", EVENT_GROUP_VBIF},
        {"EVENT_GROUP_VBIF_PWR", EVENT_GROUP_VBIF_PWR},
        {"EVENT_GROUP_MH", EVENT_GROUP_MH},
        {"EVENT_GROUP_PA_SU", EVENT_GROUP_PA_SU},
        {"EVENT_GROUP_SQ", EVENT_GROUP_SQ},
        {"EVENT_GROUP_SX", EVENT_GROUP_SX},
        {"EVENT_GROUP_TCF", EVENT_GROUP_TCF},
        {"EVENT_GROUP_TCM", EVENT_GROUP_TCM},
        {"EVENT_GROUP_TCR", EVENT_GROUP_TCR},
        {"EVENT_GROUP_L2", EVENT_GROUP_L2},
        {"EVENT_GROUP_VSC", EVENT_GROUP_VSC},
        {"EVENT_GROUP_CCU", EVENT_GROUP_CCU},
        {"EVENT_GROUP_LRZ", EVENT_GROUP_LRZ},
        {"EVENT_GROUP_CMP", EVENT_GROUP_CMP},
        {"EVENT_GROUP_ALWAYSON", EVENT_GROUP_ALWAYSON},
        {"EVENT_GROUP_SP_PWR", EVENT_GROUP_SP_PWR},
        {"EVENT_GROUP_TP_PWR", EVENT_GROUP_TP_PWR},
        {"EVENT_GROUP_RB_PWR", EVENT_GROUP_RB_PWR},
        {"EVENT_GROUP_CCU_PWR", EVENT_GROUP_CCU_PWR},
        {"EVENT_GROUP_UCHE_PWR", EVENT_GROUP_UCHE_PWR},
        {"EVENT_GROUP_CP_PWR", EVENT_GROUP_CP_PWR},
        {"EVENT_GROUP_GPMU_PWR", EVENT_GROUP_GPMU_PWR},
        {"EVENT_GROUP_ALWAYSON_PWR", EVENT_GROUP_ALWAYSON_PWR},
        {"EVENT_GROUP_GLC", EVENT_GROUP_GLC},
        {"EVENT_GROUP_FCHE", EVENT_GROUP_FCHE},
        {"EVENT_GROUP_MHUB", EVENT_GROUP_MHUB},
        {"EVENT_GROUP_CUSTOM", EVENT_GROUP_CUSTOM},
        {"EVENT_GROUP_MAX", EVENT_GROUP_MAX},
        {"EVENTS_NOT_USED", EVENTS_NOT_USED},
        {"EVENTS_BROKEN", EVENTS_BROKEN}};

/**
 * the arg to IOCTL_EVENTS_ACTIVATE
 */
struct event_activate {
    unsigned int event_group;
    unsigned int event_selector;
    unsigned int regster_offset_low;
    unsigned int regster_offset_high;
    unsigned int __pad;
};

/**
 * the arg to IOCTL_EVENTS_DEACTIVATE
 */
struct event_deactivate {
    unsigned int event_group;
    unsigned int event_selector;
    unsigned int __pad[2];
};

/**
 * the arg to IOCTL_EVENTS_READ
 */
struct adreno_event_read_entry {
    unsigned int event_group;
    unsigned int event_selector;
    unsigned long long event_value;
};

struct event_read {
    struct adreno_event_read_entry* entries;
    unsigned int num_entries;
    unsigned int __pad[2];
};

#define IOC_TYPE 0x09
#define IOCTL_EVENTS_ACTIVATE _IOWR(IOC_TYPE, 0x38, struct event_activate)
#define IOCTL_EVENT_DEACTIVATE _IOW(IOC_TYPE, 0x39, struct event_deactivate)
#define IOCTL_EVENTS_READ _IOWR(IOC_TYPE, 0x3B, struct event_read)
