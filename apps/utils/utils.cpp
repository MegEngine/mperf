#include "utils.h"

void mperf_usage(int argc, char* argv[], std::string& usage) {
    fprintf(stderr, "Usage: %s %s", argv[0], usage.c_str());
    exit(-1);
}

char last(char* s) {
    while (*s++)
        ;
    return (s[-2]);
}

uint64 bytes(char* s) {
    uint64 n;

    if (sscanf(s, "%llu", &n) < 1)
        return 0;

    if ((last(s) == 'k') || (last(s) == 'K'))
        n *= 1024;
    if ((last(s) == 'm') || (last(s) == 'M'))
        n *= (1024 * 1024);
    return (n);
}
