#include "model_backend.h"
#ifdef USE_LLAMA
#include "llama.h"
#include <stdexcept>
#include <cstring>

class LlamaBackend final : public ModelBackend {
public:
    LlamaBackend() : model_(nullptr), ctx_(nullptr), n_vocab_(0), eos_(-1), bos_(-1) {}

    ~LlamaBackend() override {
        if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
        if (model_) { llama_free_model(model_); model_ = nullptr; }
        llama_backend_free();
    }

    bool load(const std::string& model_path, const GenerateParams& params) override {
        llama_backend_init();

        auto mparams = llama_model_default_params();
        model_ = llama_load_model_from_file(model_path.c_str(), mparams);
        if (!model_) return false;

        auto cparams = llama_context_default_params();
        cparams.n_ctx = params.n_ctx;
        cparams.seed  = (uint32_t)params.seed;
        cparams.n_threads = params.n_threads;

        ctx_ = llama_new_context_with_model(model_, cparams);
        if (!ctx_) return false;

        n_vocab_ = llama_n_vocab(model_);
        eos_ = llama_token_eos(model_);
        bos_ = llama_token_bos(model_);
        return true;
    }

    std::vector<int32_t> tokenize(const std::string& text, bool add_bos) override {
        // First, query how many tokens are needed
        int32_t n_est = (int32_t)text.size() + 8;
        std::vector<llama_token> tmp(n_est);
        int32_t n = llama_tokenize(model_, text.c_str(), (int32_t)text.size(), tmp.data(), (int32_t)tmp.size(), /*add_special*/ false, /*parse_special*/ true);
        if (n < 0) {
            // n is the required size (negative)
            tmp.resize(-n);
            n = llama_tokenize(model_, text.c_str(), (int32_t)text.size(), tmp.data(), (int32_t)tmp.size(), false, true);
        }
        std::vector<int32_t> out;
        out.reserve((add_bos && bos_ != -1) ? n + 1 : n);
        if (add_bos && bos_ != -1) out.push_back(bos_);
        for (int i = 0; i < n; ++i) out.push_back((int32_t)tmp[i]);
        return out;
    }

    std::string detokenize(const std::vector<int32_t>& tokens) override {
        // Convert tokens to string piece by piece
        std::string out;
        out.reserve(tokens.size() * 3);
        for (auto t : tokens) {
            const char* s = llama_token_to_piece(model_, (llama_token)t);
            if (s) out += s;
        }
        return out;
    }

    void reset() override {
        llama_kv_cache_clear(ctx_);
    }

    bool eval(const std::vector<int32_t>& tokens) override {
        if (tokens.empty()) return true;
        // Prepare batch for a single sequence
        llama_batch batch = llama_batch_init((int)tokens.size(), /*embd*/0, /*n_seq_max*/1);
        for (int i = 0; i < (int)tokens.size(); ++i) {
            batch.token[i] = (llama_token)tokens[i];
            batch.pos[i]   = pos_ + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == (int)tokens.size() - 1); // request logits for last token
        }
        pos_ += (int)tokens.size();
        int err = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        return err == 0;
    }

    std::vector<float> logits() override {
        const float* ptr = llama_get_logits(ctx_);
        std::vector<float> out(n_vocab_);
        std::memcpy(out.data(), ptr, sizeof(float) * n_vocab_);
        return out;
    }

    int32_t vocab_size() const override { return n_vocab_; }
    int32_t bos_token() const override { return bos_; }
    int32_t eos_token() const override { return eos_; }

private:
    llama_model*  model_;
    llama_context* ctx_;
    int n_vocab_;
    int32_t eos_;
    int32_t bos_;
    int pos_ = 0;
};

// Factory
#include <memory>
std::unique_ptr<ModelBackend> create_llama_backend() {
    return std::unique_ptr<ModelBackend>(new LlamaBackend());
}

#else
// Dummy to avoid linker error if llama is disabled
#include "model_backend.h"
#include <memory>
std::unique_ptr<ModelBackend> create_llama_backend() { return nullptr; }
#endif