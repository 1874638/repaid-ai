#include "sampling.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

static void apply_repeat_frequency_presence(
    std::vector<float>& logits,
    const std::vector<int32_t>& recent,
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty
) {
    if (recent.empty()) return;

    // Count frequencies
    std::unordered_map<int32_t, int> freq;
    freq.reserve(recent.size());
    for (auto t : recent) ++freq[t];

    for (size_t i = 0; i < logits.size(); ++i) {
        auto it = freq.find((int32_t)i);
        if (it == freq.end()) continue;

        // repetition penalty: reduce or flip depending on sign
        if (repeat_penalty != 1.0f) {
            if (logits[i] > 0) logits[i] /= repeat_penalty;
            else               logits[i] *= repeat_penalty;
        }
        // frequency and presence: subtract proportional penalties
        logits[i] -= frequency_penalty * (float)it->second;
        logits[i] -= presence_penalty  * 1.0f;
    }
}

static int32_t greedy_argmax(const std::vector<float>& logits) {
    return (int32_t)std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

int32_t sample_next_token(
    const std::vector<float>& logits_in,
    const std::vector<int32_t>& recent_tokens,
    float temperature,
    int top_k,
    float top_p,
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty
) {
    // Copy logits to mutable
    std::vector<float> logits = logits_in;

    // Apply repetition/frequency/presence penalties
    apply_repeat_frequency_presence(logits, recent_tokens, repeat_penalty, frequency_penalty, presence_penalty);

    // If temperature == 0: greedy
    if (temperature <= 0.0f) {
        return greedy_argmax(logits);
    }

    // Apply temperature scaling and exponentiate to get unnormalized probs
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
    }

    // Top-k filter
    std::vector<int32_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (top_k > 0 && (int)indices.size() > top_k) {
        std::nth_element(indices.begin(), indices.begin() + top_k, indices.end(), [&](int a, int b) {
            return probs[a] > probs[b];
        });
        indices.resize(top_k);
    }

    // Gather top candidates
    std::vector<std::pair<int32_t, float>> cand;
    cand.reserve(indices.size());
    for (auto i : indices) cand.emplace_back(i, probs[i]);

    // Sort by prob desc
    std::sort(cand.begin(), cand.end(), [](auto& a, auto& b){ return a.second > b.second; });

    // Top-p filter
    if (top_p < 1.0f) {
        float cum = 0.0f;
        size_t cutoff = cand.size();
        for (size_t i = 0; i < cand.size(); ++i) {
            cum += cand[i].second;
            if (cum >= top_p) { cutoff = i + 1; break; }
        }
        cand.resize(std::max<size_t>(1, cutoff));
    }

    // Normalize probabilities
    float sum = 0.0f;
    for (auto& kv : cand) sum += kv.second;
    if (sum <= 0.0f) {
        // Fallback to greedy
        return greedy_argmax(logits);
    }
    for (auto& kv : cand) kv.second /= sum;

    // Sample
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float acc = 0.0f;
    for (auto& kv : cand) {
        acc += kv.second;
        if (r <= acc) return kv.first;
    }
    return cand.back().first;
}