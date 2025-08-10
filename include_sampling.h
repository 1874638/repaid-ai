#pragma once
#include <vector>
#include <cstdint>

// Apply penalties and sample next token from logits.
// - temperature: > 0.0. If == 0.0, do greedy.
// - top_k: 0 means disabled.
// - top_p: 1.0 means disabled.
// - repeat_penalty: >= 1.0. 1.0 means disabled.
// - frequency_penalty / presence_penalty: like OpenAI API.
// recent_tokens: context window of last tokens to penalize repetition.
int32_t sample_next_token(
    const std::vector<float>& logits,
    const std::vector<int32_t>& recent_tokens,
    float temperature,
    int top_k,
    float top_p,
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty
);