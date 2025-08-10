#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct GenerateParams {
    int n_ctx = 4096;
    int n_threads = 8;
    unsigned long long seed = 0; // 0 -> random
};

class ModelBackend {
public:
    virtual ~ModelBackend() = default;

    // Load model file and prepare context.
    virtual bool load(const std::string& model_path, const GenerateParams& params) = 0;

    // Tokenization and detokenization.
    virtual std::vector<int32_t> tokenize(const std::string& text, bool add_bos) = 0;
    virtual std::string detokenize(const std::vector<int32_t>& tokens) = 0;

    // Start a new generation session (clear kv cache if needed).
    virtual void reset() = 0;

    // Evaluate given tokens (append to context). After eval, logits() should reflect last token.
    virtual bool eval(const std::vector<int32_t>& tokens) = 0;

    // Return logits for the last position (size == vocab size).
    virtual std::vector<float> logits() = 0;

    // Convenient helpers.
    virtual int32_t vocab_size() const = 0;
    virtual int32_t bos_token() const = 0; // may return -1 if not used
    virtual int32_t eos_token() const = 0; // must be valid if model has EOS
};