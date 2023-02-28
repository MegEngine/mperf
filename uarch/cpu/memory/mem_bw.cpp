#include <string>
#include "bench.h"

#define TYPE int
#define USE_SIMPLE_BENCH 0
#define streq !strcmp

using namespace mperf;

void srd(int iterations, void* cookie);
void swr(int iterations, void* cookie);
void scp(int iterations, void* cookie);
void fwr(int iterations, void* cookie);
void frd(int iterations, void* cookie);
void frdwr(int iterations, void* cookie);
void fcp(int iterations, void* cookie);
void loop_bzero(int iterations, void* cookie);
void loop_bcopy(int iterations, void* cookie);
void f_triad(int iterations, void* cookie);
void f_rnd_rd(int iterations, void* cookie);
void f_rnd_wr(int iterations, void* cookie);
void f_add1(int iterations, void* cookie);
void f_add2(int iterations, void* cookie);
void f_mla(int iterations, void* cookie);
void f_sum(int iterations, void* cookie);
void f_dot(int iterations, void* cookie);
void init_loop(int iterations, void* cookie);
void cleanup(int iterations, void* cookie);

typedef struct _state {
    size_t nbytes;
    int need_buf2;
    int need_buf3;
    int aligned;
    TYPE* buf;
    TYPE* buf2;
    TYPE* buf2_orig;
    TYPE* buf3;
    TYPE* buf3_orig;
    TYPE* lastone;
} state_t;

#define BENCHMP cost = benchmp_simple

float adjusted_bandwidth_simple(double t, int b, double rw_count,
                                const char* fname, const char* prefix);

float mperf::cpu_mem_bw(BenchParam param, int aligned, int nbytes, char* mop,
                        char* core_list) {
    int parallel = param.parallel;
    int warmup = param.warmup;
    int repetitions = param.repetitions;
    state_t state;

    /* should have two, possibly three [indicates align] arguments left */
    state.aligned = state.need_buf2 = 0;
    state.aligned = state.need_buf3 = 0;
    state.aligned = aligned;
    state.nbytes = nbytes;

    if (streq(mop, "scp") || streq(mop, "fcp") || streq(mop, "bcopy") ||
        streq(mop, "triad") || streq(mop, "add2") || streq(mop, "mla") ||
        streq(mop, "sum") || streq(mop, "dot")) {
        state.need_buf2 = 1;
    }

    if (streq(mop, "triad") || streq(mop, "mla") || streq(mop, "sum") ||
        streq(mop, "dot")) {
        state.need_buf3 = 1;
    }

    double cost = 0.0f;
    double rw_count = 1.0;
    std::string prefix = "";
    if (streq(mop, "srd")) {
        BENCHMP(init_loop, srd, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 1 / 4.0f;
        prefix = "read-stride-4";
    } else if (streq(mop, "swr")) {
        BENCHMP(init_loop, swr, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 1 / 4.0f;
        prefix = "write-stride-4";
    } else if (streq(mop, "scp")) {
        BENCHMP(init_loop, scp, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 1 / 2.0f;
        prefix = "copy-stride-4";
    } else if (streq(mop, "frd")) {
        BENCHMP(init_loop, frd, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 1.0;
        prefix = "read";
    } else if (streq(mop, "fwr")) {
        BENCHMP(init_loop, fwr, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 1.0;
        prefix = "write";
    } else if (streq(mop, "frdwr")) {
        BENCHMP(init_loop, frdwr, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 2.0;
        prefix = "read-write-same-addr";
    } else if (streq(mop, "fcp")) {
        BENCHMP(init_loop, fcp, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 2.0;
        prefix = "copy";
    } else if (streq(mop, "bzero")) {
        BENCHMP(init_loop, loop_bzero, cleanup, 0, parallel, warmup,
                repetitions, &state);
        rw_count = 1.0;
        prefix = "libc-bzero";
    } else if (streq(mop, "bcopy")) {
        BENCHMP(init_loop, loop_bcopy, cleanup, 0, parallel, warmup,
                repetitions, &state);
        rw_count = 2.0;
        prefix = "libc-bcopy";
    } else if (streq(mop, "triad")) {
        BENCHMP(init_loop, f_triad, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 3.0;
        prefix = "triad";
    } else if (streq(mop, "rnd_rd")) {
        BENCHMP(init_loop, f_rnd_rd, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 1 / 4.0;
        prefix = "rnd_rd";
    } else if (streq(mop, "rnd_wr")) {
        BENCHMP(init_loop, f_rnd_wr, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 1 / 4.0;
        prefix = "rnd_wr";
    } else if (streq(mop, "add1")) {
        BENCHMP(init_loop, f_add1, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 1.0;
        prefix = "add1";
    } else if (streq(mop, "add2")) {
        BENCHMP(init_loop, f_add2, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 2.0;
        prefix = "add2";
    } else if (streq(mop, "mla")) {
        BENCHMP(init_loop, f_mla, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 4.0;
        prefix = "mla";
    } else if (streq(mop, "sum")) {
        BENCHMP(init_loop, f_sum, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 1.0;
        prefix = "sum";
    } else if (streq(mop, "dot")) {
        BENCHMP(init_loop, f_dot, cleanup, 0, parallel, warmup, repetitions,
                &state);
        rw_count = 2.0;
        prefix = "dot";
    } else {
        printf("unsupported micro-kernel: %s\n", mop);
    }
    char fname[256];
    snprintf(fname, sizeof(fname), "./bw_mem_core_%sreport.txt", core_list);

    return adjusted_bandwidth_simple(cost, nbytes, rw_count, fname,
                                     prefix.c_str());
}

#define UNROLLED_SUM()                                \
    {                                                 \
        register int i;                               \
        register int s0 = 0, s1 = 0, s2 = 0, s3 = 0;  \
        register int N = state->nbytes / sizeof(int); \
        register TYPE* a = state->buf;                \
        register TYPE* b = state->buf2;               \
        register TYPE* c = state->buf3;               \
        state->buf = state->buf2;                     \
        state->buf2 = state->buf3;                    \
        state->buf3 = a;                              \
                                                      \
        for (i = 0; i < N; i = i + 4) {               \
            s0 += a[i + 0];                           \
            s1 += a[i + 1];                           \
            s2 += a[i + 2];                           \
            s3 += a[i + 3];                           \
        }                                             \
        s = s0 + s1 + s2 + s3;                        \
    }

void f_sum(int iterations, void* cookie) {
    register int s;
    state_t* state = (state_t*)cookie;

    s = 0.0;
    while (iterations-- > 0) {
        UNROLLED_SUM()
    }
    keep_int((int)s);
}

void f_dot(int iterations, void* cookie) {
    register int s;
    state_t* state = (state_t*)cookie;

    s = 0.0;
    while (iterations-- > 0) {
        register int i;
        register int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        register int N = state->nbytes / sizeof(int);

        register TYPE* a = state->buf;
        register TYPE* b = state->buf2;
        register TYPE* c = state->buf3;
        state->buf = state->buf2;
        state->buf2 = state->buf3;
        state->buf3 = a;

        for (i = 0; i < N; i = i + 4) {
            s0 += a[i + 0] * b[i + 0];
            s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2];
            s3 += a[i + 3] * b[i + 3];
        }
        s = s0 + s1 + s2 + s3;
    }
    keep_int((int)s);
}

void init_loop(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;

    if (iterations)
        return;

    state->buf = (TYPE*)valloc_internal(state->nbytes);
    state->buf2_orig = NULL;
    state->buf3_orig = NULL;
    state->lastone = (TYPE*)state->buf + state->nbytes / sizeof(TYPE) - 1;

    if (!state->buf) {
        perror("malloc");
        exit(1);
    }
#if 0
    memset(state->buf, 3, state->nbytes);
#else
    for (size_t i = 0; i < state->nbytes / sizeof(TYPE); i++) {
        state->buf[i] = rand() % 255;
    }
#endif

    if (state->need_buf2 == 1) {
        state->buf2_orig = state->buf2 =
                (TYPE*)valloc_internal(state->nbytes + 2048);
        if (!state->buf2) {
            perror("malloc");
            exit(1);
        }
#if 0
    memset(state->buf2, 3, state->nbytes + 2048);
#else
        for (size_t i = 0; i < state->nbytes / sizeof(TYPE); i++) {
            state->buf2[i] = rand() % 255;
        }
#endif
        /* default is to have stuff unaligned wrt each other */
        /* XXX - this is not well tested or thought out */
        if (state->aligned) {
            char* tmp = (char*)state->buf2;

            tmp += 2048 - 128;
            state->buf2 = (TYPE*)tmp;
        }
    }

    if (state->need_buf3 == 1) {
        state->buf3_orig = state->buf3 =
                (TYPE*)valloc_internal(state->nbytes + 2048);
        if (!state->buf3) {
            perror("malloc");
            exit(1);
        }
#if 0
    memset(state->buf3, 3, state->nbytes + 2048);
#else
        for (size_t i = 0; i < state->nbytes / sizeof(TYPE); i++) {
            state->buf3[i] = rand() % 255;
        }
#endif
        /* default is to have stuff unaligned wrt each other */
        /* XXX - this is not well tested or thought out */
        if (state->aligned) {
            char* tmp = (char*)state->buf3;

            tmp += 2048 - 128;
            state->buf3 = (TYPE*)tmp;
        }
    }
}

void cleanup(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;

    if (iterations)
        return;

    free(state->buf);
    if (state->buf2_orig)
        free(state->buf2_orig);
    if (state->buf3_orig)
        free(state->buf3_orig);
}

void srd(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* lastone = state->lastone;
    register int sum = 0;

    while (iterations-- > 0) {
        register TYPE* p = state->buf;
        while (p <= lastone) {
            sum +=
            // clang-format off
#define	DOIT(i)	p[i]+
		DOIT(0) DOIT(4) DOIT(8) DOIT(12) DOIT(16) DOIT(20) DOIT(24)
		DOIT(28) DOIT(32) DOIT(36) DOIT(40) DOIT(44) DOIT(48) DOIT(52)
		DOIT(56) DOIT(60) DOIT(64) DOIT(68) DOIT(72) DOIT(76)
		DOIT(80) DOIT(84) DOIT(88) DOIT(92) DOIT(96) DOIT(100)
		DOIT(104) DOIT(108) DOIT(112) DOIT(116) DOIT(120)
            p[124];
            p += 128;
            // clang-format on
        }
    }
    keep_int(sum);
}
#undef DOIT

void swr(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* lastone = state->lastone;

    while (iterations-- > 0) {
        register TYPE* p = state->buf;
        while (p <= lastone) {
        // clang-format off
#define	DOIT(i)	p[i] = 33;
		DOIT(0) DOIT(4) DOIT(8) DOIT(12) DOIT(16) DOIT(20) DOIT(24)
		DOIT(28) DOIT(32) DOIT(36) DOIT(40) DOIT(44) DOIT(48) DOIT(52)
		DOIT(56) DOIT(60) DOIT(64) DOIT(68) DOIT(72) DOIT(76)
		DOIT(80) DOIT(84) DOIT(88) DOIT(92) DOIT(96) DOIT(100)
		DOIT(104) DOIT(108) DOIT(112) DOIT(116) DOIT(120) DOIT(124);
        p += 128;
            // clang-format on
        }
    }
}
#undef DOIT

void frdwr(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* lastone = state->lastone;
    register int sum0 = 0;
    register int sum1 = 0;
    register int sum2 = 0;
    register int sum3 = 0;
    while (iterations-- > 0) {
        register TYPE* p = state->buf;
        while (p <= lastone) {
        // clang-format off
#define DOIT(i)       \
    sum0 += p[i + 0]; p[i + 0] = 1;     \
    sum1 += p[i + 1]; p[i + 1] = 2;     \
    sum2 += p[i + 2]; p[i + 2] = 3;     \
    sum3 += p[i + 3]; p[i + 3] = 4;
		DOIT(0) DOIT(4) DOIT(8) DOIT(12) DOIT(16) DOIT(20) DOIT(24)
		DOIT(28) DOIT(32) DOIT(36) DOIT(40) DOIT(44) DOIT(48) DOIT(52)
		DOIT(56) DOIT(60) DOIT(64) DOIT(68) DOIT(72) DOIT(76)
		DOIT(80) DOIT(84) DOIT(88) DOIT(92) DOIT(96) DOIT(100)
		DOIT(104) DOIT(108) DOIT(112) DOIT(116) DOIT(120) DOIT(124);
        p += 128;
            // clang-format on
        }
    }
    keep_int(sum0 + sum1 + sum2 + sum3);
}
#undef DOIT

void scp(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* lastone = state->lastone;
    TYPE* p_save = NULL;

    while (iterations-- > 0) {
        register TYPE* p = state->buf;
        register TYPE* dst = state->buf2;
        while (p <= lastone) {
        // clang-format off
#define	DOIT(i)	dst[i] = p[i];
		DOIT(0) DOIT(4) DOIT(8) DOIT(12) DOIT(16) DOIT(20) DOIT(24)
		DOIT(28) DOIT(32) DOIT(36) DOIT(40) DOIT(44) DOIT(48) DOIT(52)
		DOIT(56) DOIT(60) DOIT(64) DOIT(68) DOIT(72) DOIT(76)
		DOIT(80) DOIT(84) DOIT(88) DOIT(92) DOIT(96) DOIT(100)
		DOIT(104) DOIT(108) DOIT(112) DOIT(116) DOIT(120) DOIT(124);
        p += 128;
        dst += 128;
            // clang-format on
        }
        p_save = p;
    }
    keep_pointer(p_save);
}
#undef DOIT

void fwr(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* lastone = state->lastone;
    TYPE* p_save = NULL;

    while (iterations-- > 0) {
        register TYPE* p = state->buf;
        while (p <= lastone) {
#if 0
        // clang-format off
#define	DOIT(i)	p[i]= 133 % (i+1);
		DOIT(0) DOIT(1) DOIT(2) DOIT(3) DOIT(4) DOIT(5) DOIT(6)
		DOIT(7) DOIT(8) DOIT(9) DOIT(10) DOIT(11) DOIT(12)
		DOIT(13) DOIT(14) DOIT(15) DOIT(16) DOIT(17) DOIT(18)
		DOIT(19) DOIT(20) DOIT(21) DOIT(22) DOIT(23) DOIT(24)
		DOIT(25) DOIT(26) DOIT(27) DOIT(28) DOIT(29) DOIT(30)
		DOIT(31) DOIT(32) DOIT(33) DOIT(34) DOIT(35) DOIT(36)
		DOIT(37) DOIT(38) DOIT(39) DOIT(40) DOIT(41) DOIT(42)
		DOIT(43) DOIT(44) DOIT(45) DOIT(46) DOIT(47) DOIT(48)
		DOIT(49) DOIT(50) DOIT(51) DOIT(52) DOIT(53) DOIT(54)
		DOIT(55) DOIT(56) DOIT(57) DOIT(58) DOIT(59) DOIT(60)
		DOIT(61) DOIT(62) DOIT(63) DOIT(64) DOIT(65) DOIT(66)
		DOIT(67) DOIT(68) DOIT(69) DOIT(70) DOIT(71) DOIT(72)
		DOIT(73) DOIT(74) DOIT(75) DOIT(76) DOIT(77) DOIT(78)
		DOIT(79) DOIT(80) DOIT(81) DOIT(82) DOIT(83) DOIT(84)
		DOIT(85) DOIT(86) DOIT(87) DOIT(88) DOIT(89) DOIT(90)
		DOIT(91) DOIT(92) DOIT(93) DOIT(94) DOIT(95) DOIT(96)
		DOIT(97) DOIT(98) DOIT(99) DOIT(100) DOIT(101) DOIT(102)
		DOIT(103) DOIT(104) DOIT(105) DOIT(106) DOIT(107)
		DOIT(108) DOIT(109) DOIT(110) DOIT(111) DOIT(112)
		DOIT(113) DOIT(114) DOIT(115) DOIT(116) DOIT(117)
		DOIT(118) DOIT(119) DOIT(120) DOIT(121) DOIT(122)
		DOIT(123) DOIT(124) DOIT(125) DOIT(126) DOIT(127);
        p += 128;
            // clang-format on
#else
#define DOIT(i) p[i] = 133;
            DOIT(0) DOIT(1) DOIT(2) DOIT(3) DOIT(4) DOIT(5) DOIT(6) DOIT(7);
            p += 8;
#endif
        }
        p_save = p;
    }
    keep_pointer(p_save);
}
#undef DOIT

#if 1
void frd(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register int sum0 = 0;
    register int sum1 = 0;
    register int sum2 = 0;
    register int sum3 = 0;
    register TYPE* lastone = state->lastone;

    while (iterations-- > 0) {
        register TYPE* p = state->buf;
        while (p <= lastone) {
        // clang-format off
#define	DOIT(i)	sum0 += p[i + 0];sum1 += p[i + 1];sum2 += p[i + 2];sum3 += p[i + 3];
		DOIT(0) DOIT(4) DOIT(8) DOIT(12) DOIT(16) DOIT(20) DOIT(24)
		DOIT(28) DOIT(32) DOIT(36) DOIT(40) DOIT(44) DOIT(48) DOIT(52)
		DOIT(56) DOIT(60) DOIT(64) DOIT(68) DOIT(72) DOIT(76)
		DOIT(80) DOIT(84) DOIT(88) DOIT(92) DOIT(96) DOIT(100)
		DOIT(104) DOIT(108) DOIT(112) DOIT(116) DOIT(120) DOIT(124)
		p += 128;
            // clang-format on
        }
    }
    keep_int(sum0 + sum1 + sum2 + sum3);
}
#undef DOIT
#else
void frd(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register int sum = 0;
    register TYPE* lastone = state->lastone;

    while (iterations-- > 0) {
        register TYPE* p = state->buf;
        while (p <= lastone) {
            // clang-format off
		sum +=
#define	DOIT(i)	p[i]+
		DOIT(0) DOIT(1) DOIT(2) DOIT(3) DOIT(4) DOIT(5) DOIT(6)
		DOIT(7) DOIT(8) DOIT(9) DOIT(10) DOIT(11) DOIT(12)
		DOIT(13) DOIT(14) DOIT(15) DOIT(16) DOIT(17) DOIT(18)
		DOIT(19) DOIT(20) DOIT(21) DOIT(22) DOIT(23) DOIT(24)
		DOIT(25) DOIT(26) DOIT(27) DOIT(28) DOIT(29) DOIT(30)
		DOIT(31) DOIT(32) DOIT(33) DOIT(34) DOIT(35) DOIT(36)
		DOIT(37) DOIT(38) DOIT(39) DOIT(40) DOIT(41) DOIT(42)
		DOIT(43) DOIT(44) DOIT(45) DOIT(46) DOIT(47) DOIT(48)
		DOIT(49) DOIT(50) DOIT(51) DOIT(52) DOIT(53) DOIT(54)
		DOIT(55) DOIT(56) DOIT(57) DOIT(58) DOIT(59) DOIT(60)
		DOIT(61) DOIT(62) DOIT(63) DOIT(64) DOIT(65) DOIT(66)
		DOIT(67) DOIT(68) DOIT(69) DOIT(70) DOIT(71) DOIT(72)
		DOIT(73) DOIT(74) DOIT(75) DOIT(76) DOIT(77) DOIT(78)
		DOIT(79) DOIT(80) DOIT(81) DOIT(82) DOIT(83) DOIT(84)
		DOIT(85) DOIT(86) DOIT(87) DOIT(88) DOIT(89) DOIT(90)
		DOIT(91) DOIT(92) DOIT(93) DOIT(94) DOIT(95) DOIT(96)
		DOIT(97) DOIT(98) DOIT(99) DOIT(100) DOIT(101) DOIT(102)
		DOIT(103) DOIT(104) DOIT(105) DOIT(106) DOIT(107)
		DOIT(108) DOIT(109) DOIT(110) DOIT(111) DOIT(112)
		DOIT(113) DOIT(114) DOIT(115) DOIT(116) DOIT(117)
		DOIT(118) DOIT(119) DOIT(120) DOIT(121) DOIT(122)
		DOIT(123) DOIT(124) DOIT(125) DOIT(126) p[127];
		p += 128;
            // clang-format on
        }
    }
    keep_int(sum);
}
#undef DOIT
#endif

void fcp(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* lastone = state->lastone;

    while (iterations-- > 0) {
        register TYPE* p = state->buf;
        register TYPE* dst = state->buf2;
        while (p <= lastone) {
        // clang-format off
#define	DOIT(i)	dst[i]=p[i];
		DOIT(0) DOIT(1) DOIT(2) DOIT(3) DOIT(4) DOIT(5) DOIT(6)
		DOIT(7) DOIT(8) DOIT(9) DOIT(10) DOIT(11) DOIT(12)
		DOIT(13) DOIT(14) DOIT(15) DOIT(16) DOIT(17) DOIT(18)
		DOIT(19) DOIT(20) DOIT(21) DOIT(22) DOIT(23) DOIT(24)
		DOIT(25) DOIT(26) DOIT(27) DOIT(28) DOIT(29) DOIT(30)
		DOIT(31) DOIT(32) DOIT(33) DOIT(34) DOIT(35) DOIT(36)
		DOIT(37) DOIT(38) DOIT(39) DOIT(40) DOIT(41) DOIT(42)
		DOIT(43) DOIT(44) DOIT(45) DOIT(46) DOIT(47) DOIT(48)
		DOIT(49) DOIT(50) DOIT(51) DOIT(52) DOIT(53) DOIT(54)
		DOIT(55) DOIT(56) DOIT(57) DOIT(58) DOIT(59) DOIT(60)
		DOIT(61) DOIT(62) DOIT(63) DOIT(64) DOIT(65) DOIT(66)
		DOIT(67) DOIT(68) DOIT(69) DOIT(70) DOIT(71) DOIT(72)
		DOIT(73) DOIT(74) DOIT(75) DOIT(76) DOIT(77) DOIT(78)
		DOIT(79) DOIT(80) DOIT(81) DOIT(82) DOIT(83) DOIT(84)
		DOIT(85) DOIT(86) DOIT(87) DOIT(88) DOIT(89) DOIT(90)
		DOIT(91) DOIT(92) DOIT(93) DOIT(94) DOIT(95) DOIT(96)
		DOIT(97) DOIT(98) DOIT(99) DOIT(100) DOIT(101) DOIT(102)
		DOIT(103) DOIT(104) DOIT(105) DOIT(106) DOIT(107)
		DOIT(108) DOIT(109) DOIT(110) DOIT(111) DOIT(112)
		DOIT(113) DOIT(114) DOIT(115) DOIT(116) DOIT(117)
		DOIT(118) DOIT(119) DOIT(120) DOIT(121) DOIT(122)
		DOIT(123) DOIT(124) DOIT(125) DOIT(126) DOIT(127)
		p += 128;
		dst += 128;
            // clang-format on
        }
    }
}
#undef DOIT

void loop_bzero(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* p = state->buf;
    register TYPE* dst = state->buf2;
    register size_t bytes = state->nbytes;

    while (iterations-- > 0) {
        bzero(p, bytes);
    }
}

void loop_bcopy(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* p = state->buf;
    register TYPE* dst = state->buf2;
    register size_t bytes = state->nbytes;

    volatile int res = 0;
    while (iterations-- > 0) {
        bcopy(p, dst, bytes);
        res += dst[0];
    }
    keep_int(res);
}

void f_triad(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register int N = state->nbytes / sizeof(TYPE);
    register TYPE* lastone = state->lastone;
    TYPE* p_save = NULL;
    register TYPE alpha = 3;
    while (iterations-- > 0) {
        register TYPE* a = state->buf;
        register TYPE* b = state->buf2;
        register TYPE* c = state->buf3;

        while (a < lastone) {
            *(c + 0) = *(a + 0) + alpha * (*(b + 0));
            *(c + 1) = *(a + 1) + alpha * (*(b + 1));
            *(c + 2) = *(a + 2) + alpha * (*(b + 2));
            *(c + 3) = *(a + 3) + alpha * (*(b + 3));

            a += 4;
            b += 4;
            c += 4;
        }
        p_save = c;
    }
    keep_int(p_save[33]);
}

void f_rnd_rd(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* lastone = state->lastone;
    register int sum = 0;

    while (iterations-- > 0) {
        register TYPE* p = state->buf;
        while (p <= lastone) {
            sum +=
            // clang-format off
#define	DOIT(i)	p[i]+
		DOIT(8) DOIT(101) DOIT(14) DOIT(44) DOIT(88) DOIT(13) DOIT(4) 
        DOIT(45) DOIT(61) DOIT(48) DOIT(69) DOIT(110) DOIT(20) DOIT(15) 
        DOIT(72) DOIT(67) DOIT(3) DOIT(28) DOIT(19) DOIT(99) DOIT(31) 
        DOIT(46) DOIT(112) DOIT(91) DOIT(113) DOIT(18) DOIT(24) DOIT(35) 
        DOIT(39) DOIT(1) DOIT(79) DOIT(115)
            p[124];
            p += 128;
            // clang-format on
        }
    }
    keep_int(sum);
}
#undef DOIT

void f_rnd_wr(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* lastone = state->lastone;

    while (iterations-- > 0) {
        register TYPE* p = state->buf;
        while (p <= lastone) {
        // clang-format off
#define	DOIT(i)	p[i] = 33;
		DOIT(8) DOIT(101) DOIT(14) DOIT(44) DOIT(88) DOIT(13) DOIT(4) 
        DOIT(45) DOIT(61) DOIT(48) DOIT(69) DOIT(110) DOIT(20) DOIT(15) 
        DOIT(72) DOIT(67) DOIT(3) DOIT(28) DOIT(19) DOIT(99) DOIT(31) 
        DOIT(46) DOIT(112) DOIT(91) DOIT(113) DOIT(18) DOIT(24) DOIT(35) 
        DOIT(39) DOIT(1) DOIT(79) DOIT(115);
        p += 128;
            // clang-format on
        }
    }
}
#undef DOIT

void f_add1(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register TYPE* lastone = state->lastone;
    register TYPE alpha = 3;
    while (iterations-- > 0) {
        register TYPE* a = state->buf;
        while (a <= lastone) {
        // clang-format off
#define	DOIT(i)	a[i] += alpha;
		DOIT(0) DOIT(1) DOIT(2) DOIT(3)
        a += 4;
            // clang-format on
        }
    }
    keep_pointer(state->buf);
}
#undef DOIT

void f_add2(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register int N = state->nbytes / sizeof(TYPE);
    register TYPE* lastone = state->lastone;
    register TYPE alpha = 3;
    while (iterations-- > 0) {
        register TYPE* a = state->buf;
        register TYPE* b = state->buf2;
        register int i = 0;
        while (a <= lastone) {
        // clang-format off
#define	DOIT(i)	b[i] = a[i] + alpha;
		DOIT(0) DOIT(1) DOIT(2) DOIT(3) DOIT(4) DOIT(5) DOIT(6)
		DOIT(7) DOIT(8) DOIT(9) DOIT(10) DOIT(11) DOIT(12)
		DOIT(13) DOIT(14) DOIT(15) DOIT(16) DOIT(17) DOIT(18)
		DOIT(19) DOIT(20) DOIT(21) DOIT(22) DOIT(23) DOIT(24)
		DOIT(25) DOIT(26) DOIT(27) DOIT(28) DOIT(29) DOIT(30)
		DOIT(31) DOIT(32) DOIT(33) DOIT(34) DOIT(35) DOIT(36)
		DOIT(37) DOIT(38) DOIT(39) DOIT(40) DOIT(41) DOIT(42)
		DOIT(43) DOIT(44) DOIT(45) DOIT(46) DOIT(47) DOIT(48)
		DOIT(49) DOIT(50) DOIT(51) DOIT(52) DOIT(53) DOIT(54)
		DOIT(55) DOIT(56) DOIT(57) DOIT(58) DOIT(59) DOIT(60)
		DOIT(61) DOIT(62) DOIT(63) DOIT(64) DOIT(65) DOIT(66)
		DOIT(67) DOIT(68) DOIT(69) DOIT(70) DOIT(71) DOIT(72)
		DOIT(73) DOIT(74) DOIT(75) DOIT(76) DOIT(77) DOIT(78)
		DOIT(79) DOIT(80) DOIT(81) DOIT(82) DOIT(83) DOIT(84)
		DOIT(85) DOIT(86) DOIT(87) DOIT(88) DOIT(89) DOIT(90)
		DOIT(91) DOIT(92) DOIT(93) DOIT(94) DOIT(95) DOIT(96)
		DOIT(97) DOIT(98) DOIT(99) DOIT(100) DOIT(101) DOIT(102)
		DOIT(103) DOIT(104) DOIT(105) DOIT(106) DOIT(107)
		DOIT(108) DOIT(109) DOIT(110) DOIT(111) DOIT(112)
		DOIT(113) DOIT(114) DOIT(115) DOIT(116) DOIT(117)
		DOIT(118) DOIT(119) DOIT(120) DOIT(121) DOIT(122)
		DOIT(123) DOIT(124) DOIT(125) DOIT(126) DOIT(127)
        a += 128;
        b += 128;
            // clang-format on
        }
    }
    keep_int(state->buf2[33]);
}
#undef DOIT

void f_mla(int iterations, void* cookie) {
    state_t* state = (state_t*)cookie;
    register int N = state->nbytes / sizeof(TYPE);
    register TYPE* lastone = state->lastone;
    TYPE* p_save = NULL;
    while (iterations-- > 0) {
        register TYPE* a = state->buf;
        register TYPE* b = state->buf2;
        register TYPE* c = state->buf3;
        register int i = 0;

        while (a < lastone) {
            *(c + 0) = *(c + 0) + (*(a + 0)) * (*(b + 0));
            *(c + 1) = *(c + 1) + (*(a + 1)) * (*(b + 1));
            *(c + 2) = *(c + 2) + (*(a + 2)) * (*(b + 2));
            *(c + 3) = *(c + 3) + (*(a + 3)) * (*(b + 3));

            a += 4;
            b += 4;
            c += 4;
        }
        p_save = c;
    }
    keep_int(p_save[33]);
}

float adjusted_bandwidth_simple(double secs, int bytes, double rw_count,
                                const char* fname, const char* prefix) {
#define MB (1000. * 1000.)
    FILE* ftiming = fopen(fname, "a");

    double mb;
    mb = bytes / MB;

    if (!ftiming)
        ftiming = stderr;

    if (mb < 1.) {
        fprintf(ftiming, "%10s%10d ", prefix, bytes);
        printf("%10d ", bytes);
    } else {
        fprintf(ftiming, "%10s%10d ", prefix, bytes);
        printf("%10d ", bytes);
    }
    if (mb / secs < 1.) {
        fprintf(ftiming, "%10.6f \n", rw_count * mb / secs);
        printf("%10.6f \n", rw_count * mb / secs);
    } else {
        fprintf(ftiming, "%10.1f \n", rw_count * mb / secs);
        printf("%10.1f \n", rw_count * mb / secs);
    }

    fclose(ftiming);
#undef MB
    return rw_count * mb / secs;
}

float mperf::cpu_dram_bandwidth() {
    constexpr size_t NR_WARMUP = 5, NR_RUNS = 40, NR_BYTES = 1024 * 1024 * 100;
#if 1
    std::unique_ptr<uint8_t[]> src{new uint8_t[NR_BYTES]()};
    std::unique_ptr<uint8_t[]> dst{new uint8_t[NR_BYTES]()};

    volatile uint8_t res = 0;
    for (size_t i = 0; i < NR_WARMUP; i++) {
        memcpy(dst.get(), src.get(), NR_BYTES);
        res += dst[0];
    }

    mperf::Timer timer;
    for (size_t i = 0; i < NR_RUNS; i++) {
        memcpy(dst.get(), src.get(), NR_BYTES);
        res += dst[0];
    }
    float used = timer.get_msecs() / NR_RUNS;
#else
    uint8_t* src = (uint8_t*)memalign(getpagesize(), NR_BYTES);
    uint8_t* dst = (uint8_t*)memalign(getpagesize(), NR_BYTES + 2048);
    memset(src, 10, NR_BYTES);
    memset(dst, 10, NR_BYTES);

    printf("warmup bcopy\n");
    volatile uint8_t res = 0;
    for (size_t i = 0; i < NR_WARMUP; i++) {
        bcopy(src, dst, NR_BYTES);
        res += dst[0];
    }

    Timer timer;
    for (size_t i = 0; i < NR_RUNS; i++) {
        bcopy(src, dst, NR_BYTES);
        res += dst[0];
    }
    float used = timer.get_msecs() / NR_RUNS;
#endif
    float gbps = 2 * NR_BYTES / (1024.0 * 1024.0 * 1024.0) * 1000 / used;
    printf("bandwidth: %f Gbps, \n", gbps);

    keep_int(res);

    return gbps;
}