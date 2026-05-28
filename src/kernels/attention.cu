// Attention bridge into reflex-infer.

#include "jllm_kernels.h"

#include <reflex/infer.h>
#include <cstdio>

namespace jllm {
namespace {

reflex::infer::StreamHandle reflex_stream(cudaStream_t stream) {
    return reinterpret_cast<reflex::infer::StreamHandle>(stream);
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

void flash_attention_decode(
    half* output, const half* q, const void* k_cache, const void* v_cache,
    int n_heads, int n_kv_heads, int head_dim, int seq_len,
    float scale, bool kv_int8,
    const float* k_scales, const float* v_scales, cudaStream_t stream)
{
    reflex::infer::AttentionDecodeArgs args{};
    args.output = output;
    args.q = q;
    args.k_cache = k_cache;
    args.v_cache = v_cache;
    args.n_heads = n_heads;
    args.n_kv_heads = n_kv_heads;
    args.head_dim = head_dim;
    args.seq_len = seq_len;
    args.scale = scale;
    args.kv_int8 = kv_int8;
    args.k_scales = k_scales;
    args.v_scales = v_scales;
    args.stream = reflex_stream(stream);

    const auto status = reflex::infer::flash_attention_decode(args);
    if (!reflex_success(status)) {
        log_reflex_failure("flash_attention_decode", status);
    }
}

void flash_attention_prefill_batched(
    half* output, const half* q, const void* k_cache, const void* v_cache,
    int n_heads, int n_kv_heads, int head_dim,
    int N, int start_pos,
    float scale, bool kv_int8,
    const float* k_scales, const float* v_scales, cudaStream_t stream)
{
    reflex::infer::AttentionPrefillBatchedArgs args{};
    args.output = output;
    args.q = q;
    args.k_cache = k_cache;
    args.v_cache = v_cache;
    args.n_heads = n_heads;
    args.n_kv_heads = n_kv_heads;
    args.head_dim = head_dim;
    args.N = N;
    args.start_pos = start_pos;
    args.scale = scale;
    args.kv_int8 = kv_int8;
    args.k_scales = k_scales;
    args.v_scales = v_scales;
    args.stream = reflex_stream(stream);

    const auto status = reflex::infer::flash_attention_prefill_batched(args);
    if (!reflex_success(status)) {
        log_reflex_failure("flash_attention_prefill_batched", status);
    }
}

}  // namespace jllm
