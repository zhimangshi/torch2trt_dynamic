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

    // Example:
    // output shape [H, W] = [8, 16] -> R=2, total=128
    // indices [N, K] where K=1 means we write a full row slice of length W (slice_size=16).
    // This matches typical scatter_nd "update a row" use-case and shows slice vectorization.
    const int32_t out_shape[2] = {8, 16};
    const int R = 2;
    const int K = 1;
    const int N = 6;

    // Allocate in VCCM for performance.
    // NOTE: do not put `__vccm` on *local variable declarations*; keep it on
    // casts/usages instead.
    int8_t* out_v = (int8_t*)__vccm_alloca(out_shape[0] * out_shape[1] * sizeof(int8_t));
    int32_t* idx_v = (int32_t*)__vccm_alloca(N * K * sizeof(int32_t));
    int8_t* upd_v = (int8_t*)__vccm_alloca(N * out_shape[1] * sizeof(int8_t));

    if (!out_v || !idx_v || !upd_v) {
        printf("VCCM alloc failed\n");
        return -1;
    }

    // Allocate DDR buffers for scalar reference timing (avoid VCCM access effects).
    int8_t* out_ref = (int8_t*)malloc(out_shape[0] * out_shape[1] * sizeof(int8_t));
    int32_t* idx_ref = (int32_t*)malloc(N * K * sizeof(int32_t));
    int8_t* upd_ref = (int8_t*)malloc(N * out_shape[1] * sizeof(int8_t));
    if (!out_ref || !idx_ref || !upd_ref) {
        printf("DDR alloc failed\n");
        free(out_ref);
        free(idx_ref);
        free(upd_ref);
        return -1;
    }

    // Init output to zeros
    for (int i = 0; i < out_shape[0] * out_shape[1]; ++i) out_v[i] = 0;
    memset(out_ref, 0, out_shape[0] * out_shape[1] * sizeof(int8_t));

    // Fill indices: choose rows 0, 2, 3, 7, 2 (duplicate), 5
    idx_v[0] = 0;
    idx_v[1] = 2;
    idx_v[2] = 3;
    idx_v[3] = 7;
    idx_v[4] = 2;
    idx_v[5] = 5;
    for (int i = 0; i < N * K; ++i) idx_ref[i] = idx_v[i];

    // Fill updates: each update is a row slice (16 bytes)
    for (int n = 0; n < N; ++n) {
        for (int j = 0; j < out_shape[1]; ++j) {
            upd_v[n * out_shape[1] + j] = (int8_t)(n * 10 + j);
            upd_ref[n * out_shape[1] + j] = upd_v[n * out_shape[1] + j];
        }
    }

    // Run scalar reference with timing
    RESET_TIMER0();
    const uint32_t tr0 = READ_TIMER0();
    scatternd_ref_i8(out_ref, out_shape, R, idx_ref, N, K, upd_ref);
    const uint32_t tr1 = READ_TIMER0();
    printf("REF  cycles: %u\n", (unsigned)(tr1 - tr0));

    // Run VDSP kernel with timing
    RESET_TIMER0();
    const uint32_t t0 = READ_TIMER0();
    scatternd_vdsp_i8((int8_t __vccm*)out_v, out_shape, R, (const int32_t __vccm*)idx_v, N, K, (const int8_t __vccm*)upd_v);
    const uint32_t t1 = READ_TIMER0();
    printf("VDSP cycles: %u\n", (unsigned)(t1 - t0));

    int mism = 0;
    for (int i = 0; i < 128; ++i) {
        if (out_ref[i] != out_v[i]) {
            if (mism < 8) {
                printf("mismatch at %d: ref=%d vdsp=%d\n", i, (int)out_ref[i], (int)out_v[i]);
            }
            mism++;
        }
    }
    printf("Check: %s (mismatches=%d)\n", mism ? "FAIL" : "OK", mism);

    // Print one row to visualize last-write-wins for duplicate row=2
    printf("Row 2 after scatter (should match last update with index=2):\n");
    for (int j = 0; j < out_shape[1]; ++j) {
        printf("%d ", (int)out_v[2 * out_shape[1] + j]);
    }
    printf("\n");

    free(out_ref);
    free(idx_ref);
    free(upd_ref);

    printf("ScatterND demo ended\n");
    return 0;
}

