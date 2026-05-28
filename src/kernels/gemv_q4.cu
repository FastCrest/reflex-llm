// Q4 GEMV/GEMM bridge into reflex-infer.

#include "jllm_kernels.h"

#include <reflex/infer.h>
#include <cstdio>

namespace jllm {
namespace {

reflex::infer::StreamHandle reflex_stream(cudaStream_t stream) {
    return reinterpret_cast<reflex::infer::StreamHandle>(stream);
}

const void* resolve_weight_device_ptr(const void* W) {
    if (const void* mapped = resolve_mapped_weight_device_ptr(W)) {
        return mapped;
    }

    cudaPointerAttributes attr{};
    cudaError_t err = cudaPointerGetAttributes(&attr, W);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }

#if CUDART_VERSION >= 10000
    if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
        return W;
    }
#else
    if (attr.memoryType == cudaMemoryTypeDevice) {
        return W;
    }
#endif

    return nullptr;
}

bool reflex_success(reflex::infer::Status status) {
    return status == reflex::infer::Status::Success;
}

void log_reflex_failure(const char* op, reflex::infer::Status status) {
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "[reflex-infer] %s failed with status %d\n",
                op, static_cast<int>(status));
        warned = true;
    }
}

}  // namespace

void gemv_q4(half* y, const void* W_q4, const half* scales, const half* x,
             int M, int K, int group_size, cudaStream_t stream) {
    (void)scales;
    (void)group_size;
    gemv_quant(y, W_q4, 12, x, M, K, stream);
}

void gemv_quant(half* y, const void* W, int ggml_type, const half* x,
                int M, int K, cudaStream_t stream) {
    reflex::infer::GemvQuantArgs args{};
    args.y = y;
    args.weights = W;
    args.weights_device = resolve_weight_device_ptr(W);
    args.ggml_type = ggml_type;
    args.x = x;
    args.M = M;
    args.K = K;
    args.stream = reflex_stream(stream);

    const auto status = reflex::infer::gemv_quant(args);
    if (!reflex_success(status)) {
        log_reflex_failure("gemv_quant", status);
    }
}

void gemv_quant_add(half* y, const void* W, int ggml_type, const half* x,
                    const half* residual, int M, int K, cudaStream_t stream) {
    reflex::infer::GemvQuantAddArgs args{};
    args.y = y;
    args.weights = W;
    args.weights_device = resolve_weight_device_ptr(W);
    args.ggml_type = ggml_type;
    args.x = x;
    args.residual = residual;
    args.M = M;
    args.K = K;
    args.stream = reflex_stream(stream);

    const auto status = reflex::infer::gemv_quant_add(args);
    if (!reflex_success(status)) {
        log_reflex_failure("gemv_quant_add", status);
    }
}

void gemv_quant_pair(
    half* y0, const void* W0, int ggml_type0, int M0,
    half* y1, const void* W1, int ggml_type1, int M1,
    const half* x, int K, cudaStream_t stream)
{
    reflex::infer::GemvQuantPairArgs args{};
    args.y0 = y0;
    args.weights0 = W0;
    args.weights0_device = resolve_weight_device_ptr(W0);
    args.ggml_type0 = ggml_type0;
    args.M0 = M0;
    args.y1 = y1;
    args.weights1 = W1;
    args.weights1_device = resolve_weight_device_ptr(W1);
    args.ggml_type1 = ggml_type1;
    args.M1 = M1;
    args.x = x;
    args.K = K;
    args.stream = reflex_stream(stream);

    const auto status = reflex::infer::gemv_quant_pair(args);
    if (!reflex_success(status)) {
        log_reflex_failure("gemv_quant_pair", status);
    }
}

void gemv_quant_triple(
    half* y0, const void* W0, int ggml_type0, int M0,
    half* y1, const void* W1, int ggml_type1, int M1,
    half* y2, const void* W2, int ggml_type2, int M2,
    const half* x, int K, cudaStream_t stream)
{
    reflex::infer::GemvQuantTripleArgs args{};
    args.y0 = y0;
    args.weights0 = W0;
    args.weights0_device = resolve_weight_device_ptr(W0);
    args.ggml_type0 = ggml_type0;
    args.M0 = M0;
    args.y1 = y1;
    args.weights1 = W1;
    args.weights1_device = resolve_weight_device_ptr(W1);
    args.ggml_type1 = ggml_type1;
    args.M1 = M1;
    args.y2 = y2;
    args.weights2 = W2;
    args.weights2_device = resolve_weight_device_ptr(W2);
    args.ggml_type2 = ggml_type2;
    args.M2 = M2;
    args.x = x;
    args.K = K;
    args.stream = reflex_stream(stream);

    const auto status = reflex::infer::gemv_quant_triple(args);
    if (!reflex_success(status)) {
        log_reflex_failure("gemv_quant_triple", status);
    }
}

void gemm_quant_batched(half* y, const void* W, int ggml_type, const half* x,
                        int M, int N, int K, cudaStream_t stream) {
    reflex::infer::GemmQuantBatchedArgs args{};
    args.y = y;
    args.weights = W;
    args.weights_device = resolve_weight_device_ptr(W);
    args.ggml_type = ggml_type;
    args.x = x;
    args.M = M;
    args.N = N;
    args.K = K;
    args.stream = reflex_stream(stream);

    const auto status = reflex::infer::gemm_quant_batched(args);
    if (!reflex_success(status)) {
        log_reflex_failure("gemm_quant_batched", status);
    }
}

void gemv_quant_f32(float* y, const void* W, int ggml_type, const half* x,
                    int M, int K, cudaStream_t stream) {
    reflex::infer::GemvQuantF32Args args{};
    args.y = y;
    args.weights = W;
    args.weights_device = resolve_weight_device_ptr(W);
    args.ggml_type = ggml_type;
    args.x = x;
    args.M = M;
    args.K = K;
    args.stream = reflex_stream(stream);

    const auto status = reflex::infer::gemv_quant_f32(args);
    if (!reflex_success(status)) {
        log_reflex_failure("gemv_quant_f32", status);
    }
}

bool dequant_embedding_row(half* dst, const void* W, int ggml_type,
                           int token_id, int hidden_dim, cudaStream_t stream) {
    reflex::infer::DequantEmbeddingRowArgs args{};
    args.dst = dst;
    args.weights = W;
    args.weights_device = resolve_weight_device_ptr(W);
    args.ggml_type = ggml_type;
    args.token_id = token_id;
    args.hidden_dim = hidden_dim;
    args.stream = reflex_stream(stream);

    const auto status = reflex::infer::dequant_embedding_row(args);
    if (!reflex_success(status)) {
        log_reflex_failure("dequant_embedding_row", status);
        return false;
    }
    return true;
}

}  // namespace jllm
