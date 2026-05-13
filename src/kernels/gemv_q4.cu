// gemv_q4.cu — Q4_K dequant-fused GEMV for Orin SM 8.7
// Matches llama.cpp's exact Q4_K block layout and dequant formula.

#include "jllm_kernels.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace jllm {

static constexpr int QK_K = 256;

static bool debug_kernels_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_DEBUG_KERNELS");
        return v && strcmp(v, "0") != 0;
    }();
    return enabled;
}

static bool fast_gemv_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_FAST_GEMV");
        return v && strcmp(v, "0") != 0;
    }();
    return enabled;
}

// Use raw uint16 instead of half2 to guarantee no padding
struct __attribute__((packed)) block_q4_K {
    uint16_t d_raw;       // FP16 super-block scale
    uint16_t dmin_raw;    // FP16 super-block min
    uint8_t  scales[12];  // packed 6-bit sub-block scales and mins
    uint8_t  qs[QK_K/2]; // 4-bit quants (128 bytes)
};
static_assert(sizeof(block_q4_K) == 144, "Q4_K block must be 144 bytes");

struct __attribute__((packed)) block_q5_K {
    uint16_t d_raw;
    uint16_t dmin_raw;
    uint8_t  scales[12];
    uint8_t  qh[QK_K/8];
    uint8_t  qs[QK_K/2];
};
static_assert(sizeof(block_q5_K) == 176, "Q5_K block must be 176 bytes");

struct __attribute__((packed)) block_q6_K {
    uint8_t  ql[QK_K/2];
    uint8_t  qh[QK_K/4];
    int8_t   scales[QK_K/16];
    uint16_t d_raw;
};
static_assert(sizeof(block_q6_K) == 210, "Q6_K block must be 210 bytes");

__host__ __device__ __forceinline__ float raw_fp16_to_float(uint16_t h) {
#ifdef __CUDA_ARCH__
    return __half2float(__ushort_as_half(h));
#else
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float result;
    if (exp == 0) {
        result = ldexpf((float)mant, -24);
    } else if (exp == 31) {
        result = mant ? NAN : INFINITY;
    } else {
        result = ldexpf((float)(mant + 1024), (int)exp - 25);
    }
    return sign ? -result : result;
#endif
}

__host__ __device__ __forceinline__ void get_scale_min_k4(
    int j, const uint8_t* q, uint8_t& d, uint8_t& m)
{
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

__global__ void gemv_q4k_kernel(
    half*              __restrict__ y,
    const block_q4_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K)
{
    const int row  = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    const block_q4_K* row_blocks = W + (int64_t)row * n_blocks;

    float acc = 0.0f;

    for (int b = lane; b < n_blocks; b += 32) {
        const block_q4_K& blk = row_blocks[b];

        float dall = raw_fp16_to_float(blk.d_raw);
        float dmin = raw_fp16_to_float(blk.dmin_raw);

        int k_base = b * QK_K;

        for (int il = 0; il < 4; il++) {
            int is = 2 * il;

            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk.scales, sc1, m1);
            get_scale_min_k4(is + 1, blk.scales, sc2, m2);

            float d1 = dall * sc1;
            float dm1 = dmin * m1;
            float d2 = dall * sc2;
            float dm2 = dmin * m2;

            const uint8_t* q = blk.qs + 32 * il;

            for (int l = 0; l < 32; l++) {
                int k_lo = k_base + 64 * il + l;
                int k_hi = k_base + 64 * il + l + 32;

                float w_lo = d1 * (q[l] & 0xF) - dm1;
                float w_hi = d2 * (q[l] >> 4)  - dm2;

                acc += w_lo * __half2float(x[k_lo]);
                acc += w_hi * __half2float(x[k_hi]);
            }
        }
    }

    // Warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0)
        y[row] = __float2half(acc);
}

__global__ void gemv_q5k_kernel(
    half*              __restrict__ y,
    const block_q5_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K)
{
    const int row  = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    const block_q5_K* row_blocks = W + (int64_t)row * n_blocks;

    float acc = 0.0f;

    for (int b = lane; b < n_blocks; b += 32) {
        const block_q5_K& blk = row_blocks[b];

        const float dall = raw_fp16_to_float(blk.d_raw);
        const float dmin = raw_fp16_to_float(blk.dmin_raw);
        const int k_base = b * QK_K;

        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (int il = 0; il < 4; il++) {
            const int is = 2 * il;
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk.scales, sc1, m1);
            get_scale_min_k4(is + 1, blk.scales, sc2, m2);

            const float d1 = dall * sc1;
            const float dm1 = dmin * m1;
            const float d2 = dall * sc2;
            const float dm2 = dmin * m2;

            const uint8_t* ql = blk.qs + 32 * il;
            const uint8_t* qh = blk.qh;

            for (int l = 0; l < 32; l++) {
                const int k_lo = k_base + 64 * il + l;
                const int k_hi = k_lo + 32;

                const float w_lo = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - dm1;
                const float w_hi = d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - dm2;

                acc += w_lo * __half2float(x[k_lo]);
                acc += w_hi * __half2float(x[k_hi]);
            }

            u1 <<= 2;
            u2 <<= 2;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0)
        y[row] = __float2half(acc);
}

__global__ void gemv_q6k_kernel(
    half*              __restrict__ y,
    const block_q6_K*  __restrict__ W,
    const half*        __restrict__ x,
    int M, int K)
{
    const int row  = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;

    const int n_blocks = K / QK_K;
    const block_q6_K* row_blocks = W + (int64_t)row * n_blocks;

    float acc = 0.0f;

    for (int b = lane; b < n_blocks; b += 32) {
        const block_q6_K& blk = row_blocks[b];
        const float d = raw_fp16_to_float(blk.d_raw);
        const int k_base = b * QK_K;

        const uint8_t* ql = blk.ql;
        const uint8_t* qh = blk.qh;
        const int8_t*  sc = blk.scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                const int k0 = k_base + n + l;
                acc += d * (float)sc[is + 0] * (float)q1 * __half2float(x[k0 +  0]);
                acc += d * (float)sc[is + 2] * (float)q2 * __half2float(x[k0 + 32]);
                acc += d * (float)sc[is + 4] * (float)q3 * __half2float(x[k0 + 64]);
                acc += d * (float)sc[is + 6] * (float)q4 * __half2float(x[k0 + 96]);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);

    if (lane == 0)
        y[row] = __float2half(acc);
}

// Debug kernel: print first few output values
__global__ void debug_print_half(const half* data, int n, const char* label) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[GPU %s] first 8: ", label);
        for (int i = 0; i < 8 && i < n; i++)
            printf("%.4f ", __half2float(data[i]));
        printf("\n");
    }
}

void gemv_q4(half* y, const void* W_q4, const half* scales, const half* x,
             int M, int K, int group_size, cudaStream_t stream) {
    (void)scales;
    (void)group_size;
    gemv_quant(y, W_q4, 12, x, M, K, stream);
}

static bool gemv_quant_gpu(half* y, const void* W, int ggml_type,
                           const half* x, int M, int K, cudaStream_t stream) {
    const void* W_device = resolve_mapped_weight_device_ptr(W);
    if (!W_device) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr,
                    "[GEMV] CUDA-visible mapped weight pointer is unavailable; "
                    "falling back to CPU reference GEMV\n");
            warned = true;
        }
        return false;
    }

    const int block = 128;
    const int grid = (M + 3) / 4;

    switch (ggml_type) {
        case 12:
            gemv_q4k_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q4_K*)W_device, x, M, K);
            break;
        case 13:
            gemv_q5k_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q5_K*)W_device, x, M, K);
            break;
        case 14:
            gemv_q6k_kernel<<<grid, block, 0, stream>>>(
                y, (const block_q6_K*)W_device, x, M, K);
            break;
        default:
            fprintf(stderr, "[GEMV] FATAL: unsupported GPU GGML type %d (M=%d K=%d)\n",
                    ggml_type, M, K);
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GEMV] GPU launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

__global__ void half_to_float_kernel(float* out, const half* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

static bool gemv_quant_cpu(std::vector<float>& h_y, const void* W, int ggml_type,
                           const std::vector<half>& h_x, int M, int K) {
    if (ggml_type != 12 && ggml_type != 13 && ggml_type != 14) {
        fprintf(stderr, "[GEMV] FATAL: unsupported GGML type %d (M=%d K=%d)\n",
                ggml_type, M, K);
        return false;
    }

    const int n_blocks = K / QK_K;
    h_y.assign(M, 0.0f);

    for (int row = 0; row < M; row++) {
        float acc = 0.0f;

        if (ggml_type == 12) {
            const block_q4_K* row_blocks = (const block_q4_K*)W + (int64_t)row * n_blocks;
            for (int b = 0; b < n_blocks; b++) {
                const block_q4_K& blk = row_blocks[b];
                const float dall = raw_fp16_to_float(blk.d_raw);
                const float dmin = raw_fp16_to_float(blk.dmin_raw);
                const uint8_t* q = blk.qs;
                int is = 0;
                int k = b * QK_K;

                for (int j = 0; j < QK_K; j += 64) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, blk.scales, sc1, m1);
                    get_scale_min_k4(is + 1, blk.scales, sc2, m2);
                    const float d1 = dall * sc1;
                    const float dm1 = dmin * m1;
                    const float d2 = dall * sc2;
                    const float dm2 = dmin * m2;
                    for (int l = 0; l < 32; l++) {
                        acc += (d1 * (q[l] & 0xF) - dm1) * __half2float(h_x[k + l]);
                    }
                    for (int l = 0; l < 32; l++) {
                        acc += (d2 * (q[l] >> 4) - dm2) * __half2float(h_x[k + 32 + l]);
                    }
                    q += 32;
                    is += 2;
                    k += 64;
                }
            }
        } else if (ggml_type == 13) {
            const block_q5_K* row_blocks = (const block_q5_K*)W + (int64_t)row * n_blocks;
            for (int b = 0; b < n_blocks; b++) {
                const block_q5_K& blk = row_blocks[b];
                const float dall = raw_fp16_to_float(blk.d_raw);
                const float dmin = raw_fp16_to_float(blk.dmin_raw);
                const uint8_t* ql = blk.qs;
                const uint8_t* qh = blk.qh;
                uint8_t u1 = 1, u2 = 2;
                int is = 0;
                int k = b * QK_K;

                for (int j = 0; j < QK_K; j += 64) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, blk.scales, sc1, m1);
                    get_scale_min_k4(is + 1, blk.scales, sc2, m2);
                    const float d1 = dall * sc1;
                    const float dm1 = dmin * m1;
                    const float d2 = dall * sc2;
                    const float dm2 = dmin * m2;
                    for (int l = 0; l < 32; l++) {
                        const float w = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - dm1;
                        acc += w * __half2float(h_x[k + l]);
                    }
                    for (int l = 0; l < 32; l++) {
                        const float w = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - dm2;
                        acc += w * __half2float(h_x[k + 32 + l]);
                    }
                    ql += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                    k += 64;
                }
            }
        } else {
            const block_q6_K* row_blocks = (const block_q6_K*)W + (int64_t)row * n_blocks;
            for (int b = 0; b < n_blocks; b++) {
                const block_q6_K& blk = row_blocks[b];
                const float d = raw_fp16_to_float(blk.d_raw);
                const uint8_t* ql = blk.ql;
                const uint8_t* qh = blk.qh;
                const int8_t* sc = blk.scales;
                int k_base = b * QK_K;

                for (int n = 0; n < QK_K; n += 128) {
                    for (int l = 0; l < 32; l++) {
                        const int is = l / 16;
                        const int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        const int k = k_base + n + l;
                        acc += d * (float)sc[is + 0] * (float)q1 * __half2float(h_x[k +  0]);
                        acc += d * (float)sc[is + 2] * (float)q2 * __half2float(h_x[k + 32]);
                        acc += d * (float)sc[is + 4] * (float)q3 * __half2float(h_x[k + 64]);
                        acc += d * (float)sc[is + 6] * (float)q4 * __half2float(h_x[k + 96]);
                    }
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
        }

        h_y[row] = acc;
    }

    return true;
}

void gemv_quant(half* y, const void* W, int ggml_type, const half* x,
                int M, int K, cudaStream_t stream) {
    if (fast_gemv_enabled()) {
        if (gemv_quant_gpu(y, W, ggml_type, x, M, K, stream)) {
            static int dbg_count = 0;
            if (debug_kernels_enabled() && dbg_count < 3) {
                cudaError_t err = cudaStreamSynchronize(stream);
                if (err != cudaSuccess) {
                    fprintf(stderr, "[GEMV] GPU execution failed: %s\n",
                            cudaGetErrorString(err));
                    cudaMemsetAsync(y, 0, M * sizeof(half), stream);
                    return;
                }
                half h_y[8], h_x[8];
                cudaMemcpy(h_y, y, 8 * sizeof(half), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_x, x, 8 * sizeof(half), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[GEMV-GPU #%d] type=%d M=%d K=%d out: ", dbg_count, ggml_type, M, K);
                for (int i = 0; i < 8 && i < M; i++) fprintf(stderr, "%.4f ", __half2float(h_y[i]));
                fprintf(stderr, " | in: ");
                for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", __half2float(h_x[i]));
                fprintf(stderr, "\n");
                dbg_count++;
            }
            return;
        }
    }

    cudaStreamSynchronize(stream);

    std::vector<half> h_x(K);
    std::vector<float> h_y_f32;
    std::vector<half> h_y(M);
    cudaMemcpy(h_x.data(), x, K * sizeof(half), cudaMemcpyDeviceToHost);

    if (!gemv_quant_cpu(h_y_f32, W, ggml_type, h_x, M, K)) {
        cudaMemsetAsync(y, 0, M * sizeof(half), stream);
        return;
    }

    for (int i = 0; i < M; i++) h_y[i] = __float2half(h_y_f32[i]);
    cudaMemcpy(y, h_y.data(), M * sizeof(half), cudaMemcpyHostToDevice);

    static int dbg_count = 0;
    if (debug_kernels_enabled() && dbg_count < 3) {
        cudaStreamSynchronize(stream);
        half h_y[8], h_x[8];
        cudaMemcpy(h_y, y, 8 * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_x, x, 8 * sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[GEMV #%d] type=%d M=%d K=%d out: ", dbg_count, ggml_type, M, K);
        for (int i = 0; i < 8 && i < M; i++) fprintf(stderr, "%.4f ", __half2float(h_y[i]));
        fprintf(stderr, " | in: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", __half2float(h_x[i]));
        fprintf(stderr, "\n");
        dbg_count++;
    }
}

void gemv_quant_f32(float* y, const void* W, int ggml_type, const half* x,
                    int M, int K, cudaStream_t stream) {
    if (fast_gemv_enabled()) {
        half* tmp = nullptr;
        cudaError_t err = cudaMalloc(&tmp, M * sizeof(half));
        if (err != cudaSuccess) {
            fprintf(stderr, "[GEMV] cudaMalloc tmp logits failed: %s\n",
                    cudaGetErrorString(err));
            cudaMemsetAsync(y, 0, M * sizeof(float), stream);
            return;
        }

        if (!gemv_quant_gpu(tmp, W, ggml_type, x, M, K, stream)) {
            cudaFree(tmp);
        } else {
            int block = 256;
            int grid = (M + block - 1) / block;
            half_to_float_kernel<<<grid, block, 0, stream>>>(y, tmp, M);
            err = cudaStreamSynchronize(stream);
            cudaFree(tmp);
            if (err == cudaSuccess) {
                return;
            }
            fprintf(stderr, "[GEMV] GPU FP32 output path failed: %s\n",
                    cudaGetErrorString(err));
            cudaMemsetAsync(y, 0, M * sizeof(float), stream);
            return;
        }
    }

    cudaStreamSynchronize(stream);

    std::vector<half> h_x(K);
    std::vector<float> h_y;
    cudaMemcpy(h_x.data(), x, K * sizeof(half), cudaMemcpyDeviceToHost);

    if (!gemv_quant_cpu(h_y, W, ggml_type, h_x, M, K)) {
        cudaMemsetAsync(y, 0, M * sizeof(float), stream);
        return;
    }

    cudaMemcpy(y, h_y.data(), M * sizeof(float), cudaMemcpyHostToDevice);
}

}  // namespace jllm
