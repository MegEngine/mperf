#define R2(S)   S;       S
#define R4(S)   R2(S);   R2(S)
#define R8(S)   R4(S);   R4(S)
#define R16(S)  R8(S);   R8(S)
#define R32(S)  R16(S);  R16(S)
#define R64(S)  R32(S);  R32(S)
#define R128(S) R64(S);  R64(S)
#define R256(S) R128(S); R128(S)
#define R512(S) R256(S); R256(S)

#define KERNEL1(a,b,c)   ((a) = (b) + (c))
#define KERNEL2(a,b,c)   ((a) = (a)*(b) + (c))

__kernel void pmu_test_kernel(const int nsize, const int trials,
                         __global float* A, __global int* params) {
    for (int j = 0; j < trials; ++j) {
        size_t total_thr = get_num_groups(0) * get_local_size(0);
        size_t elem_per_thr = (nsize + (total_thr - 1)) / total_thr;
        size_t blockOffset = get_group_id(0) * get_local_size(0);

        size_t start_idx = blockOffset + get_local_id(0);
        size_t end_idx = start_idx + elem_per_thr * total_thr;
        size_t stride_idx = total_thr;

        if (start_idx > nsize) {
            start_idx = nsize;
        }

        if (end_idx > nsize) {
            end_idx = nsize;
        }

        // A needs to be initilized to -1 coming in
        // And with alpha=2 and beta=1, A=-1 is preserved upon return
        float alpha = 2.0;
        float beta = 1.0;
        size_t i, j;
        for (i = start_idx; i < end_idx; i += stride_idx) {
            beta = 1.0;
            R512(KERNEL2(beta, A[i], alpha));
            A[i] = -beta;
        }
    }

    params[0] = sizeof(*A);
    params[1] = 2;
}
