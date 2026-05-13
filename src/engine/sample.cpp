// sample.cpp — Token sampling (top-k, top-p, temperature)
//
// Runs on CPU — logits are small (vocab_size floats) and sampling
// has branchy control flow that GPUs handle poorly.

#include "jllm_engine.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <cstdint>
#include <numeric>
#include <vector>
#include <random>

namespace jllm {

static thread_local std::mt19937 g_rng(42);

static bool finite_bits(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7F800000u) != 0x7F800000u;
}

static bool debug_kernels_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("JLLM_DEBUG_KERNELS");
        return v && strcmp(v, "0") != 0;
    }();
    return enabled;
}

struct TokenProb {
    int   id;
    float prob;
};

// Apply temperature scaling
static void apply_temperature(float* logits, int n, float temperature) {
    if (temperature <= 0.0f || temperature == 1.0f) return;
    float inv_t = 1.0f / temperature;
    for (int i = 0; i < n; i++)
        logits[i] *= inv_t;
}

// Apply repetition penalty
static void apply_repeat_penalty(float* logits, const int* recent_tokens,
                                  int n_recent, float penalty) {
    if (penalty <= 1.0f) return;
    for (int i = 0; i < n_recent; i++) {
        int tok = recent_tokens[i];
        if (logits[tok] > 0) logits[tok] /= penalty;
        else                 logits[tok] *= penalty;
    }
}

// Top-K: keep only the K highest logits
static int top_k_filter(TokenProb* candidates, int n, int k) {
    if (k >= n) return n;
    std::partial_sort(candidates, candidates + k, candidates + n,
                      [](const TokenProb& a, const TokenProb& b) {
                          return a.prob > b.prob;
                      });
    return k;
}

// Top-P (nucleus): keep tokens until cumulative probability > p
static int top_p_filter(TokenProb* candidates, int n, float p) {
    // Already sorted by probability (descending) from top-k
    float cumsum = 0.0f;
    int cutoff = n;
    for (int i = 0; i < n; i++) {
        cumsum += candidates[i].prob;
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }
    return cutoff;
}

// Sample from filtered distribution
static int sample_from(const TokenProb* candidates, int n) {
    // Compute total probability for renormalization
    float total = 0.0f;
    for (int i = 0; i < n; i++) total += candidates[i].prob;

    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(g_rng);

    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += candidates[i].prob;
        if (cumsum >= r) return candidates[i].id;
    }
    return candidates[n - 1].id;
}

// Greedy: just pick the highest logit
static int sample_greedy(const float* logits, int n) {
    int best = 0;
    float best_val = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (finite_bits(logits[i]) && logits[i] > best_val) {
            best = i;
            best_val = logits[i];
        }
    }
    return best;
}

// ── Public API ──────────────────────────────────────────────────────────

int sample_token(float* logits, int vocab_size, const GenParams& params,
                 const int* recent_tokens, int n_recent) {
    int invalid = 0;
    float min_val = FLT_MAX;
    float max_val_raw = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        if (!finite_bits(logits[i])) {
            logits[i] = -INFINITY;
            invalid++;
        } else {
            min_val = std::min(min_val, logits[i]);
            max_val_raw = std::max(max_val_raw, logits[i]);
        }
    }

    if (debug_kernels_enabled()) {
        int best[5] = {0, 0, 0, 0, 0};
        float vals[5] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (int i = 0; i < vocab_size; i++) {
            const float v = logits[i];
            for (int j = 0; j < 5; j++) {
                if (v > vals[j]) {
                    for (int k = 4; k > j; k--) {
                        vals[k] = vals[k - 1];
                        best[k] = best[k - 1];
                    }
                    vals[j] = v;
                    best[j] = i;
                    break;
                }
            }
        }
        fprintf(stderr,
                "[sample] logits min=%.4f max=%.4f invalid=%d top: %d=%.4f %d=%.4f %d=%.4f %d=%.4f %d=%.4f\n",
                min_val, max_val_raw, invalid,
                best[0], vals[0], best[1], vals[1], best[2], vals[2],
                best[3], vals[3], best[4], vals[4]);
    }

    // Greedy at temperature 0
    if (params.temperature <= 0.0f)
        return sample_greedy(logits, vocab_size);

    // Apply penalties
    apply_repeat_penalty(logits, recent_tokens, n_recent, params.repeat_penalty);
    apply_temperature(logits, vocab_size, params.temperature);

    // Softmax (on CPU — small array)
    float max_val = *std::max_element(logits, logits + vocab_size);
    if (!finite_bits(max_val)) return 0;

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    if (!finite_bits(sum) || sum <= 0.0f) return sample_greedy(logits, vocab_size);
    float inv_sum = 1.0f / sum;

    // Build candidates
    std::vector<TokenProb> candidates(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        candidates[i] = {i, logits[i] * inv_sum};
    }

    // Filter
    int n = top_k_filter(candidates.data(), vocab_size, params.top_k);
    n = top_p_filter(candidates.data(), n, params.top_p);

    return sample_from(candidates.data(), n);
}

}  // namespace jllm
