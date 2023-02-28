#pragma once
#include <assert.h>
#include <getopt.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

typedef unsigned long long uint64;

void mperf_usage(int argc, char* argv[], std::string& usage);

uint64 bytes(char* s);
