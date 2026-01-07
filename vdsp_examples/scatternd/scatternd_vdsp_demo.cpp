/*
 * Minimal ScatterND (int8) demo for Synopsys VDSP style intrinsics.
 *
 * Goal:
 * - Provide a first, runnable skeleton in simulator similar to the RMSNorm "FIXPOINT_VECTOR" style:
 *   - scalar reference
 *   - vectorized kernel using vscatter + predicate tail handling
 *   - timing via TIMER0
 *
 * Supported in this initial version:
 * - indices: int32, shape [N, K]
 * - updates: int8, shape [N, slice_size] where slice_size = prod(output_shape[K..R-1])
 * - output:  int8, shape [output_shape...]
 * - reduction: "none" (overwrite). If indices repeat, "last write wins".
 *
 * Notes:
 * - Scatter is "irregular write". The only scalable SIMD form on VDSP is batching multiple writes
 *   via vscatter (addresses per lane). That is what this file demonstrates.
 * - For slice_size > 1 we vectorize inside the slice as well (still using vscatter, but with
 *   contiguous offsets).
 */
// CONFIDENTIAL/PROPRIETARY header intentionally omitted in this open demo.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vdsplib.h"

// Timer utilities (same style as your RMSNorm code)
#define INIT_TIMER0()  _sr(0xffffffff, 0x23); _sr(0x3, 0x22);
#define RESET_TIMER0() _sr(0x0, 0x21);
#define READ_TIMER0()  _lr(0x21)

// -----------------------------
// Small helpers (VDSP predicate + offsets)
// -----------------------------

static inline __attribute__((always_inline)) pvNx4 pred_n_lanes(int n) {
    return to_pvNx4(vvci_b() < n);
}

// Build an offsets vector for int8 scatter from a scalar offsets[] array.
// We keep this "dumb but safe" for v1: fill lane-by-lane in C.
static inline __attribute__((always_inline)) vNx4int_t pack_offsets_nx4(const int32_t* offsets, int n) {
    vNx4int_t v = (vNx4int_t)0;
    for (int i = 0; i < n; ++i) {
        v[i] = offsets[i];
    }
    return v;
}

// Load int8 values into a vNx4char_t from a scalar array.
static inline __attribute__((always_inline)) vNx4char_t pack_i8_nx4(const int8_t* values, int n) {
    vNx4char_t v = (vNx4char_t)0;
    for (int i = 0; i < n; ++i) {
        v[i] = values[i];
    }
    return v;
}

// -----------------------------
// ScatterND core (reference + VDSP)
// -----------------------------

static inline int32_t prod_i32(const int32_t* a, int n) {
    int32_t p = 1;
    for (int i = 0; i < n; ++i) p *= a[i];
    return p;
}

// Compute row-major strides for output shape [R]
// stride[d] = prod(shape[d+1..R-1])
static void compute_strides(const int32_t* shape, int R, int32_t* strides_out) {
    int32_t running = 1;
    for (int d = R - 1; d >= 0; --d) {
        strides_out[d] = running;
        running *= shape[d];
    }
}

// Scalar reference scatter_nd ("none" reduction).
// output must be pre-initialized (e.g., zeros or a copy of input).
static void scatternd_ref_i8(
    int8_t* out,
    const int32_t* out_shape, int R,
    const int32_t* indices, int N, int K,
    const int8_t* updates
) {
    int32_t strides[8];
    if (R > (int)(sizeof(strides) / sizeof(strides[0]))) {
        printf("R too large for demo\n");
        return;
    }
    compute_strides(out_shape, R, strides);

    const int32_t slice_size = prod_i32(&out_shape[K], R - K);

    for (int i = 0; i < N; ++i) {
        int32_t base = 0;
        for (int d = 0; d < K; ++d) {
            const int32_t idx = indices[i * K + d];
            base += idx * strides[d];
        }
        // overwrite
        memcpy(&out[base], &updates[i * slice_size], (size_t)slice_size);
    }
}

// -----------------------------
// Rank-5 "custom-op style" reference & vector versions (reduction = NONE only)
// This mirrors the nested loops shape of the code snippet you provided.
// -----------------------------

// Fixed rank for this demo
enum { RANK5 = 5 };

static inline void normalize_rank_dims(const int32_t* dims_in, int rank, int32_t* dims5_out) {
    // Fill missing trailing dims with 1 so we can keep a fixed 5D POS() implementation.
    for (int i = 0; i < RANK5; ++i) dims5_out[i] = 1;
    for (int i = 0; i < rank; ++i) dims5_out[i] = dims_in[i];
}

static inline int32_t pos5(const int32_t* dims, int d0, int d1, int d2, int d3, int d4) {
    // Row-major: ((((d0*D1 + d1)*D2 + d2)*D3 + d3)*D4 + d4)
    return ((((d0 * dims[1] + d1) * dims[2] + d2) * dims[3] + d3) * dims[4] + d4);
}

// indices tensor layout for this demo: [I0, I1, I2, I3, K]
static inline int32_t pos_idx5(const int32_t* idx_dims, int d0, int d1, int d2, int d3, int d4) {
    return ((((d0 * idx_dims[1] + d1) * idx_dims[2] + d2) * idx_dims[3] + d3) * idx_dims[4] + d4);
}

// updates tensor layout for this demo: [I0, I1, I2, I3, 1] (slice_size==1)
static inline int32_t pos_upd5(const int32_t* upd_dims, int d0, int d1, int d2, int d3, int d4) {
    return ((((d0 * upd_dims[1] + d1) * upd_dims[2] + d2) * upd_dims[3] + d3) * upd_dims[4] + d4);
}

// Reference implementation that mirrors your snippet structure:
// - Copy input -> output with 5 nested loops (intentionally "scalar/naive")
// - Update output at positions selected by indices (K in [1..rank])
// - reduction is NONE only (overwrite)
static void scatternd_custom_ref_i8_none(
    int rank,
    const int8_t* input,
    const int32_t* in_dims_in,           // [D0..D(rank-1)]
    const int64_t* indices,
    const int32_t* idx_dims,             // [I0..I3,K] (always rank-5 layout)
    const int8_t* updates,
    const int32_t* upd_dims,             // [I0..I3,slice_size] (always rank-5 layout)
    int8_t* out
) {
    int32_t in_dims[RANK5];
    normalize_rank_dims(in_dims_in, rank, in_dims);

    // Copy input -> output (5 nested loops)
    for (uint32_t dim_0 = 0; dim_0 < (uint32_t)in_dims[0]; dim_0++) {
        for (uint32_t dim_1 = 0; dim_1 < (uint32_t)in_dims[1]; dim_1++) {
            for (uint32_t dim_2 = 0; dim_2 < (uint32_t)in_dims[2]; dim_2++) {
                for (uint32_t dim_3 = 0; dim_3 < (uint32_t)in_dims[3]; dim_3++) {
                    for (uint32_t dim_4 = 0; dim_4 < (uint32_t)in_dims[4]; dim_4++) {
                        const int in_pos = pos5(in_dims, (int)dim_0, (int)dim_1, (int)dim_2, (int)dim_3, (int)dim_4);
                        const int out_pos = in_pos;
                        out[out_pos] = input[in_pos];
                    }
                }
            }
        }
    }

    const uint32_t K = (uint32_t)idx_dims[4];
    if (K == 0 || (int)K > rank) {
        printf("Invalid K=%u for rank=%d\n", (unsigned)K, rank);
        return;
    }

    // Generic update (matches your custom_op semantics):
    // - indices select the first K dims (dims 0..K-1)
    // - the remaining (rank-K) dims form a contiguous slice in row-major memory
    // - updates last dim is a flattened slice with the same row-major order
    //
    // Therefore the update can be done as:
    //   base = sum_{k=0..K-1} selected[k] * stride[k]
    //   for t in [0..slice_size): out[base + t] = updates[updates_base + t]
    // This is exactly equivalent to the nested loops + updates_dim++ in the snippet.
    int32_t strides[5];
    compute_strides(in_dims, RANK5, strides);
    const int32_t slice_size = upd_dims[4];

    for (uint32_t dim_0 = 0; dim_0 < (uint32_t)idx_dims[0]; dim_0++) {
        for (uint32_t dim_1 = 0; dim_1 < (uint32_t)idx_dims[1]; dim_1++) {
            for (uint32_t dim_2 = 0; dim_2 < (uint32_t)idx_dims[2]; dim_2++) {
                for (uint32_t dim_3 = 0; dim_3 < (uint32_t)idx_dims[3]; dim_3++) {
                    int32_t base = 0;
                    for (uint32_t k = 0; k < K; ++k) {
                        const int idx_pos = pos_idx5(idx_dims, (int)dim_0, (int)dim_1, (int)dim_2, (int)dim_3, (int)k);
                        const int32_t selected = (int32_t)indices[idx_pos];
                        base += selected * strides[k];
                    }

                    const int updates_base = pos_upd5(upd_dims, (int)dim_0, (int)dim_1, (int)dim_2, (int)dim_3, 0);
                    for (int32_t t = 0; t < slice_size; ++t) {
                        out[base + t] = updates[updates_base + t];  // reduction = NONE
                    }
                }
            }
        }
    }
}

// Vector version for the same "custom-op style" semantics (rank up to 5, K in [1..rank], reduction=none):
// - Copy input->output using memcpy (fast baseline)
// - Update:
//   - if slice_size == 1 (K == rank): use vscatter batching (random point writes)
//   - else: use memcpy for the contiguous slice copy (still much faster than nested loops)
__attribute__((noinline))
static void scatternd_custom_vdsp_i8_none(
    int rank,
    const int8_t* input,
    const int32_t* in_dims_in,
    const int64_t* indices,
    const int32_t* idx_dims,
    const int8_t* updates,
    const int32_t* upd_dims,
    int8_t* __vccm out
) {
    int32_t in_dims[RANK5];
    normalize_rank_dims(in_dims_in, rank, in_dims);

    const int32_t total = prod_i32(in_dims, RANK5);
    memcpy((void*)out, (const void*)input, (size_t)total);

    const int K = idx_dims[4];
    if (K <= 0 || K > rank) {
        printf("Invalid K=%d for rank=%d\n", K, rank);
        return;
    }

    // Precompute strides for output position computation
    int32_t strides[5];
    compute_strides(in_dims, RANK5, strides);

    const int lanes = _VDSP_NUM_8BIT_LANES;
    const int N = idx_dims[0] * idx_dims[1] * idx_dims[2] * idx_dims[3];

    const int slice_size = upd_dims[4];

    if (slice_size == 1) {
        // Random point updates: vscatter batching.
        // We treat indices as a flat AoS array of length N*K (K==rank here).
        for (int i = 0; i < N; i += lanes) {
            const int n = (N - i) > lanes ? lanes : (N - i);

            int32_t offs[_VDSP_NUM_8BIT_LANES];
            int8_t vals[_VDSP_NUM_8BIT_LANES];

            for (int l = 0; l < n; ++l) {
                const int row = i + l;
                int64_t d[5] = {0, 0, 0, 0, 0};
                for (int k = 0; k < K; ++k) {
                    d[k] = indices[row * K + k];
                }
                const int32_t base = (int32_t)(d[0] * strides[0] + d[1] * strides[1] + d[2] * strides[2] +
                                              d[3] * strides[3] + d[4] * strides[4]);
                offs[l] = base;
                vals[l] = updates[row];
            }

            const vNx4int_t vOffs = pack_offsets_nx4(offs, n);
            const vNx4char_t vVals = pack_i8_nx4(vals, n);
            vscatter(vVals, (int8_t __vccm*)out, vOffs, pred_n_lanes(n));
        }
        return;
    }

    // Slice copy (contiguous): compute base and memcpy slice.
    for (int row = 0; row < N; ++row) {
        int64_t d[5] = {0, 0, 0, 0, 0};
        for (int k = 0; k < K; ++k) {
            d[k] = indices[row * K + k];
        }
        const int32_t base = (int32_t)(d[0] * strides[0] + d[1] * strides[1] + d[2] * strides[2] +
                                      d[3] * strides[3] + d[4] * strides[4]);
        memcpy((void*)&out[base], (const void*)&updates[row * slice_size], (size_t)slice_size);
    }
}

// VDSP vectorized scatter_nd for int8, reduction="none".
// - Uses vscatter to batch irregular writes.
// - Handles tails via predicate.
// - For slice_size > 1, vectorizes within each slice (contiguous offsets).
//
// Safety assumptions for v1:
// - indices are in-range (no bounds check). Add checks if you need robustness.
// - output is in VCCM for best performance (can work in DDR but slower).
__attribute__((noinline))
static void scatternd_vdsp_i8(
    int8_t* __vccm out,
    const int32_t* out_shape, int R,
    const int32_t* __vccm indices, int N, int K,
    const int8_t* __vccm updates
) {
    int32_t strides[8];
    if (R > (int)(sizeof(strides) / sizeof(strides[0]))) {
        printf("R too large for demo\n");
        return;
    }
    compute_strides(out_shape, R, strides);

    const int32_t slice_size = prod_i32(&out_shape[K], R - K);

    // Lane count for int8 vectors on this target
    const int lanes = _VDSP_NUM_8BIT_LANES;

    if (slice_size == 1) {
        // Best case for vscatter batching: each update is a single element.
        for (int i = 0; i < N; i += lanes) {
            const int n = (N - i) > lanes ? lanes : (N - i);

            int32_t offs[_VDSP_NUM_8BIT_LANES];
            int8_t vals[_VDSP_NUM_8BIT_LANES];

            for (int l = 0; l < n; ++l) {
                const int row = i + l;
                int32_t base = 0;
                for (int d = 0; d < K; ++d) {
                    base += indices[row * K + d] * strides[d];
                }
                offs[l] = base;
                vals[l] = updates[row];
            }

            const vNx4int_t vOffs = pack_offsets_nx4(offs, n);
            const vNx4char_t vVals = pack_i8_nx4(vals, n);
            vscatter(vVals, (int8_t __vccm*)out, vOffs, pred_n_lanes(n));
        }
        return;
    }

    // General case: each update writes a contiguous slice of length slice_size.
    // Vectorize inside the slice (contiguous offsets), because that gives stable throughput.
    for (int i = 0; i < N; ++i) {
        int32_t base = 0;
        for (int d = 0; d < K; ++d) {
            base += indices[i * K + d] * strides[d];
        }

        // NOTE: ARC/VDSP toolchains often forbid address-space qualifiers on
        // automatic (local) variables. `updates` is already a __vccm pointer
        // as a parameter, so keep the local pointer unqualified.
        const int8_t* up = (const int8_t*)&updates[i * slice_size];

        int t = 0;
        for (; t + lanes <= slice_size; t += lanes) {
            int32_t offs[_VDSP_NUM_8BIT_LANES];
            int8_t vals[_VDSP_NUM_8BIT_LANES];
            for (int l = 0; l < lanes; ++l) {
                offs[l] = base + t + l;
                vals[l] = up[t + l];
            }
            const vNx4int_t vOffs = pack_offsets_nx4(offs, lanes);
            const vNx4char_t vVals = pack_i8_nx4(vals, lanes);
            vscatter(vVals, (int8_t __vccm*)out, vOffs);  // full lanes
        }

        const int tail = slice_size - t;
        if (tail) {
            int32_t offs[_VDSP_NUM_8BIT_LANES];
            int8_t vals[_VDSP_NUM_8BIT_LANES];
            for (int l = 0; l < tail; ++l) {
                offs[l] = base + t + l;
                vals[l] = up[t + l];
            }
            const vNx4int_t vOffs = pack_offsets_nx4(offs, tail);
            const vNx4char_t vVals = pack_i8_nx4(vals, tail);
            vscatter(vVals, (int8_t __vccm*)out, vOffs, pred_n_lanes(tail));
        }
    }
}

// -----------------------------
// Simple demo main (simulator)
// -----------------------------

int main(void) {
    printf("ScatterND demo started\n");
    INIT_TIMER0();

    // Run multiple (rank,K) cases to match custom_op behavior:
    // - idx_most_inner_dim (K) in {1..rank}
    // - reduction=none
    // For each case:
    //  - REF: 5-deep scalar loops copy + scalar update loops
    //  - VDSP: memcpy copy + vscatter or memcpy-slice update

    struct Case {
        int rank;
        int32_t dims[5];
        int K;
        int N;
        int repeats;
    };

    const Case cases[] = {
        // rank=3 example
        {3, {4, 16, 32, 1, 1}, 1, 256, 80},
        {3, {4, 16, 32, 1, 1}, 2, 512, 80},
        {3, {4, 16, 32, 1, 1}, 3, 2048, 80},
        // rank=5 example
        {5, {2, 2, 4, 4, 32}, 1, 64, 60},
        {5, {2, 2, 4, 4, 32}, 2, 128, 60},
        {5, {2, 2, 4, 4, 32}, 3, 256, 60},
        {5, {2, 2, 4, 4, 32}, 4, 512, 60},
        {5, {2, 2, 4, 4, 32}, 5, 4096, 60},
    };

    // IMPORTANT (toolchain constraint):
    // Many ARC/VDSP link scripts limit .vstack (VCCM stack) to a small size (e.g. 0x4000).
    // `__vccm_alloca` allocations accumulate until function returns, so allocating per-case in a loop
    // can easily blow .vstack. We therefore allocate ONE reusable VCCM output buffer here.
    int32_t max_total_io = 0;
    for (unsigned ci = 0; ci < (unsigned)(sizeof(cases) / sizeof(cases[0])); ++ci) {
        int32_t in_dims[5] = {1, 1, 1, 1, 1};
        for (int i = 0; i < 5; ++i) in_dims[i] = cases[ci].dims[i];
        const int32_t total_io = prod_i32(in_dims, RANK5);
        if (total_io > max_total_io) max_total_io = total_io;
    }
    // Allocate output in VCCM once and reuse for all cases.
    int8_t* out_v = (int8_t*)__vccm_alloca((size_t)max_total_io * sizeof(int8_t));
    if (!out_v) {
        printf("VCCM alloc failed for out_v (size=%d)\n", (int)max_total_io);
        return -1;
    }

    for (unsigned ci = 0; ci < (unsigned)(sizeof(cases) / sizeof(cases[0])); ++ci) {
        const Case tc = cases[ci];
        const int rank = tc.rank;
        const int K = tc.K;
        const int N = tc.N;
        const int REPEATS = tc.repeats;

        int32_t in_dims[5] = {1, 1, 1, 1, 1};
        for (int i = 0; i < 5; ++i) in_dims[i] = tc.dims[i];

        const int32_t total_io = prod_i32(in_dims, RANK5);
        const int32_t slice_size = prod_i32(&in_dims[K], RANK5 - K);

        // indices dims [1,1,1,N,K]
        int32_t idx_dims[5] = {1, 1, 1, N, K};
        // updates dims [1,1,1,N,slice_size]
        int32_t upd_dims[5] = {1, 1, 1, N, slice_size};

        printf("\nCase rank=%d K=%d N=%d slice_size=%d total_io=%d\n", rank, K, N, (int)slice_size, (int)total_io);

        // DDR buffers (reference path + vector inputs)
        int8_t* in_ref = (int8_t*)malloc((size_t)total_io * sizeof(int8_t));
        int8_t* out_ref = (int8_t*)malloc((size_t)total_io * sizeof(int8_t));
        int64_t* idx_ref = (int64_t*)malloc((size_t)N * K * sizeof(int64_t));
        int8_t* upd_ref = (int8_t*)malloc((size_t)N * (size_t)slice_size * sizeof(int8_t));
        if (!in_ref || !out_ref || !idx_ref || !upd_ref) {
            printf("DDR alloc failed for case %u\n", ci);
            free(in_ref);
            free(out_ref);
            free(idx_ref);
            free(upd_ref);
            return -1;
        }

        // Initialize deterministic random data
        srand(1 + (int)ci);
        for (int i = 0; i < total_io; ++i) {
            const int8_t v = (int8_t)((rand() % 255) - 128);
            in_ref[i] = v;
        }

        // indices in-range; updates random
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                const int64_t d = (int64_t)(rand() % in_dims[k]);
                idx_ref[n * K + k] = d;
            }
            for (int t = 0; t < slice_size; ++t) {
                const int8_t u = (int8_t)((rand() % 255) - 128);
                upd_ref[n * slice_size + t] = u;
            }
        }

        // Warmup
        scatternd_custom_ref_i8_none(rank, in_ref, in_dims, idx_ref, idx_dims, upd_ref, upd_dims, out_ref);
        scatternd_custom_vdsp_i8_none(rank, in_ref, in_dims, idx_ref, idx_dims, upd_ref, upd_dims, (int8_t __vccm*)out_v);

        // Time reference
        RESET_TIMER0();
        const uint32_t tr0 = READ_TIMER0();
        for (int it = 0; it < REPEATS; ++it) {
            scatternd_custom_ref_i8_none(rank, in_ref, in_dims, idx_ref, idx_dims, upd_ref, upd_dims, out_ref);
        }
        const uint32_t tr1 = READ_TIMER0();
        const uint32_t ref_cycles_total = (uint32_t)(tr1 - tr0);
        const uint32_t ref_avg = (uint32_t)(ref_cycles_total / (uint32_t)REPEATS);

        // Time VDSP
        RESET_TIMER0();
        const uint32_t tv0 = READ_TIMER0();
        for (int it = 0; it < REPEATS; ++it) {
            scatternd_custom_vdsp_i8_none(rank, in_ref, in_dims, idx_ref, idx_dims,
                                          upd_ref, upd_dims, (int8_t __vccm*)out_v);
        }
        const uint32_t tv1 = READ_TIMER0();
        const uint32_t vdsp_cycles_total = (uint32_t)(tv1 - tv0);
        const uint32_t vdsp_avg = (uint32_t)(vdsp_cycles_total / (uint32_t)REPEATS);

        printf("REF  cycles: %u (avg %u over %d)\n", (unsigned)ref_cycles_total, (unsigned)ref_avg, REPEATS);
        printf("VDSP cycles: %u (avg %u over %d)\n", (unsigned)vdsp_cycles_total, (unsigned)vdsp_avg, REPEATS);
        printf("Speedup: %.2fx\n", (double)ref_avg / (double)(vdsp_avg ? vdsp_avg : 1));

        // Correctness check (entire output)
        int mism = 0;
        for (int i = 0; i < total_io; ++i) {
            if (out_ref[i] != out_v[i]) {
                if (mism < 8) {
                    printf("mismatch at %d: ref=%d vdsp=%d\n", i, (int)out_ref[i], (int)out_v[i]);
                }
                mism++;
            }
        }
        printf("Check: %s (mismatches=%d)\n", mism ? "FAIL" : "OK", mism);

        // Free DDR buffers
        free(in_ref);
        free(out_ref);
        free(idx_ref);
        free(upd_ref);
    }

    printf("ScatterND demo ended\n");
    return 0;
}

