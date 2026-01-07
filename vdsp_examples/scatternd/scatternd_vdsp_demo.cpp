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
// - Update output at positions selected by indices (K == 5 => slice_size == 1)
// - reduction is NONE only (overwrite)
static void scatternd_rank5_ref_i8_none(
    const int8_t* input,
    const int32_t* in_dims,              // [D0..D4]
    const int64_t* indices,
    const int32_t* idx_dims,             // [I0..I3,K]
    const int8_t* updates,
    const int32_t* upd_dims,             // [I0..I3,1]
    int8_t* out
) {
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

    // Update (K == 5 => write single element)
    const uint32_t K = (uint32_t)idx_dims[4];
    if (K != 5) {
        printf("This demo ref implements K==5 only.\n");
        return;
    }

    for (uint32_t dim_0 = 0; dim_0 < (uint32_t)idx_dims[0]; dim_0++) {
        for (uint32_t dim_1 = 0; dim_1 < (uint32_t)idx_dims[1]; dim_1++) {
            for (uint32_t dim_2 = 0; dim_2 < (uint32_t)idx_dims[2]; dim_2++) {
                for (uint32_t dim_3 = 0; dim_3 < (uint32_t)idx_dims[3]; dim_3++) {
                    const int selected_d1_idx = pos_idx5(idx_dims, (int)dim_0, (int)dim_1, (int)dim_2, (int)dim_3, 0);
                    const int selected_d2_idx = pos_idx5(idx_dims, (int)dim_0, (int)dim_1, (int)dim_2, (int)dim_3, 1);
                    const int selected_d3_idx = pos_idx5(idx_dims, (int)dim_0, (int)dim_1, (int)dim_2, (int)dim_3, 2);
                    const int selected_d4_idx = pos_idx5(idx_dims, (int)dim_0, (int)dim_1, (int)dim_2, (int)dim_3, 3);
                    const int selected_d5_idx = pos_idx5(idx_dims, (int)dim_0, (int)dim_1, (int)dim_2, (int)dim_3, 4);

                    const int selected_d1 = (int)indices[selected_d1_idx];
                    const int selected_d2 = (int)indices[selected_d2_idx];
                    const int selected_d3 = (int)indices[selected_d3_idx];
                    const int selected_d4 = (int)indices[selected_d4_idx];
                    const int selected_d5 = (int)indices[selected_d5_idx];

                    const int out_pos = pos5(in_dims, selected_d1, selected_d2, selected_d3, selected_d4, selected_d5);
                    const int updates_pos = pos_upd5(upd_dims, (int)dim_0, (int)dim_1, (int)dim_2, (int)dim_3, 0);
                    out[out_pos] = updates[updates_pos];  // reduction = NONE
                }
            }
        }
    }
}

// Vector version for the same rank-5 / K==5 / reduction=none case:
// - Copy input->output using memcpy (fast baseline)
// - Update using vscatter batching (random point writes)
__attribute__((noinline))
static void scatternd_rank5_vdsp_i8_none(
    const int8_t* __vccm input,
    const int32_t* in_dims,
    const int64_t* __vccm indices,
    const int32_t* idx_dims,
    const int8_t* __vccm updates,
    const int32_t* upd_dims,
    int8_t* __vccm out
) {
    const int32_t total = prod_i32(in_dims, RANK5);
    memcpy((void*)out, (const void*)input, (size_t)total);

    const int K = idx_dims[4];
    if (K != 5) {
        printf("This demo VDSP implements K==5 only.\n");
        return;
    }

    // Precompute strides for output position computation
    int32_t strides[5];
    compute_strides(in_dims, RANK5, strides);

    const int lanes = _VDSP_NUM_8BIT_LANES;
    const int N = idx_dims[0] * idx_dims[1] * idx_dims[2] * idx_dims[3];

    // We treat indices as a flat AoS array of length N*K.
    // Each update corresponds to indices[n, 0..4] and updates[n,0].
    for (int i = 0; i < N; i += lanes) {
        const int n = (N - i) > lanes ? lanes : (N - i);

        int32_t offs[_VDSP_NUM_8BIT_LANES];
        int8_t vals[_VDSP_NUM_8BIT_LANES];

        for (int l = 0; l < n; ++l) {
            const int row = i + l;
            const int64_t d0 = indices[row * 5 + 0];
            const int64_t d1 = indices[row * 5 + 1];
            const int64_t d2 = indices[row * 5 + 2];
            const int64_t d3 = indices[row * 5 + 3];
            const int64_t d4 = indices[row * 5 + 4];

            const int32_t base =
                (int32_t)(d0 * strides[0] + d1 * strides[1] + d2 * strides[2] + d3 * strides[3] + d4 * strides[4]);
            offs[l] = base;
            vals[l] = updates[row];
        }

        const vNx4int_t vOffs = pack_offsets_nx4(offs, n);
        const vNx4char_t vVals = pack_i8_nx4(vals, n);
        vscatter(vVals, (int8_t __vccm*)out, vOffs, pred_n_lanes(n));
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

    // Rank-5 demo configuration (matches the "custom-op" loop shape):
    // input/output dims: [D0,D1,D2,D3,D4]
    // indices dims:       [I0,I1,I2,I3,K] with K==5 (slice_size==1)
    // updates dims:       [I0,I1,I2,I3,1]
    //
    // We choose sizes to:
    // - make the ref copy loop expensive (5 nested loops)
    // - keep buffers reasonable for VCCM
    const int32_t io_dims[RANK5] = {2, 4, 8, 8, 64};  // total = 32768
    const int32_t idx_dims[RANK5] = {1, 1, 1, 8192, 5};  // N = 8192 updates
    const int32_t upd_dims[RANK5] = {1, 1, 1, 8192, 1};
    const int REPEATS = 50;

    const int32_t total_io = prod_i32(io_dims, RANK5);
    const int N = idx_dims[0] * idx_dims[1] * idx_dims[2] * idx_dims[3];

    // VCCM buffers (vector path)
    int8_t* in_v = (int8_t*)__vccm_alloca((size_t)total_io * sizeof(int8_t));
    int8_t* out_v = (int8_t*)__vccm_alloca((size_t)total_io * sizeof(int8_t));
    int64_t* idx_v = (int64_t*)__vccm_alloca((size_t)N * 5 * sizeof(int64_t));
    int8_t* upd_v = (int8_t*)__vccm_alloca((size_t)N * sizeof(int8_t));

    if (!in_v || !out_v || !idx_v || !upd_v) {
        printf("VCCM alloc failed\n");
        return -1;
    }

    // DDR buffers (reference path)
    int8_t* in_ref = (int8_t*)malloc((size_t)total_io * sizeof(int8_t));
    int8_t* out_ref = (int8_t*)malloc((size_t)total_io * sizeof(int8_t));
    int64_t* idx_ref = (int64_t*)malloc((size_t)N * 5 * sizeof(int64_t));
    int8_t* upd_ref = (int8_t*)malloc((size_t)N * sizeof(int8_t));
    if (!in_ref || !out_ref || !idx_ref || !upd_ref) {
        printf("DDR alloc failed\n");
        free(in_ref);
        free(out_ref);
        free(idx_ref);
        free(upd_ref);
        return -1;
    }

    // Initialize deterministic random data (same content for ref & vector paths)
    srand(1);
    for (int i = 0; i < total_io; ++i) {
        const int8_t v = (int8_t)((rand() % 255) - 128);
        in_v[i] = v;
        in_ref[i] = v;
    }

    for (int n = 0; n < N; ++n) {
        // indices in-range for each of the 5 dims
        const int64_t d0 = (int64_t)(rand() % io_dims[0]);
        const int64_t d1 = (int64_t)(rand() % io_dims[1]);
        const int64_t d2 = (int64_t)(rand() % io_dims[2]);
        const int64_t d3 = (int64_t)(rand() % io_dims[3]);
        const int64_t d4 = (int64_t)(rand() % io_dims[4]);
        idx_v[n * 5 + 0] = d0;
        idx_v[n * 5 + 1] = d1;
        idx_v[n * 5 + 2] = d2;
        idx_v[n * 5 + 3] = d3;
        idx_v[n * 5 + 4] = d4;
        idx_ref[n * 5 + 0] = d0;
        idx_ref[n * 5 + 1] = d1;
        idx_ref[n * 5 + 2] = d2;
        idx_ref[n * 5 + 3] = d3;
        idx_ref[n * 5 + 4] = d4;

        const int8_t u = (int8_t)((rand() % 255) - 128);
        upd_v[n] = u;
        upd_ref[n] = u;
    }

    // Warmup
    scatternd_rank5_ref_i8_none(in_ref, io_dims, idx_ref, idx_dims, upd_ref, upd_dims, out_ref);
    scatternd_rank5_vdsp_i8_none((const int8_t __vccm*)in_v, io_dims, (const int64_t __vccm*)idx_v, idx_dims,
                                 (const int8_t __vccm*)upd_v, upd_dims, (int8_t __vccm*)out_v);

    // Time reference
    RESET_TIMER0();
    const uint32_t tr0 = READ_TIMER0();
    for (int it = 0; it < REPEATS; ++it) {
        scatternd_rank5_ref_i8_none(in_ref, io_dims, idx_ref, idx_dims, upd_ref, upd_dims, out_ref);
    }
    const uint32_t tr1 = READ_TIMER0();
    const uint32_t ref_cycles_total = (uint32_t)(tr1 - tr0);
    printf("REF  cycles: %u (avg %u over %d)\n",
           (unsigned)ref_cycles_total,
           (unsigned)(ref_cycles_total / (uint32_t)REPEATS),
           REPEATS);

    // Time VDSP
    RESET_TIMER0();
    const uint32_t tv0 = READ_TIMER0();
    for (int it = 0; it < REPEATS; ++it) {
        scatternd_rank5_vdsp_i8_none((const int8_t __vccm*)in_v, io_dims, (const int64_t __vccm*)idx_v, idx_dims,
                                     (const int8_t __vccm*)upd_v, upd_dims, (int8_t __vccm*)out_v);
    }
    const uint32_t tv1 = READ_TIMER0();
    const uint32_t vdsp_cycles_total = (uint32_t)(tv1 - tv0);
    printf("VDSP cycles: %u (avg %u over %d)\n",
           (unsigned)vdsp_cycles_total,
           (unsigned)(vdsp_cycles_total / (uint32_t)REPEATS),
           REPEATS);

    // Correctness check: compare entire output (same operation)
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

    // Print a few positions for quick sanity
    printf("Sample outputs (linear idx 0..15):\n");
    for (int j = 0; j < 16; ++j) {
        printf("%d ", (int)out_v[j]);
    }
    printf("\n");

    free(in_ref);
    free(out_ref);
    free(idx_ref);
    free(upd_ref);

    printf("ScatterND demo ended\n");
    return 0;
}

