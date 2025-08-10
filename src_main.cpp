#include "model_backend.h"
#include "sampling.h"
#include "chat_format.h"
#include <iostream>
#include <memory>
#include <deque>
#include <cstring>

// Forward decl (provided by llama backend cpp when USE_LLAMA=ON)
std::unique_ptr<ModelBackend> create_llama_backend();

struct CliArgs {
    std::string model;
    int n_threads = 8;
    int n_ctx = 4096;
    int max_new_tokens = 512;
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.95f;
    float repeat_penalty = 1.1f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    unsigned long long seed = 0;
};

static CliArgs parse_args(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        auto arg = std::string(argv[i]);
        auto next = [&](int& i){ return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (arg == "--model") a.model = next(i);
        else if (arg == "--threads") a.n_threads = std::stoi(next(i));
        else if (arg == "--ctx") a.n_ctx = std::stoi(next(i));
        else if (arg == "--max-tokens") a.max_new_tokens = std::stoi(next(i));
        else if (arg == "--temp") a.temperature = std::stof(next(i));
        else if (arg == "--top-k") a.top_k = std::stoi(next(i));
        else if (arg == "--top-p") a.top_p = std::stof(next(i));
        else if (arg == "--repeat-penalty") a.repeat_penalty = std::stof(next(i));
        else if (arg == "--freq-penalty") a.frequency_penalty = std::stof(next(i));
        else if (arg == "--presence-penalty") a.p Presence_penalty = std::stof(next(i)); // fix typo
        else if (arg == "--seed") a.seed = std::stoull(next(i));
    }
    return a;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    if (args.model.empty()) {
        std::cerr << "Usage: " << argv[0] << " --model /path/to/model.gguf [--threads N --ctx N --max-tokens N --temp F --top-k N --top-p F --repeat-penalty F --freq-penalty F --presence-penalty F --seed U64]\n";
        return 1;
    }

    auto backend = create_llama_backend();
    if (!backend) {
        std::cerr << "llama backend not available. Build with -DUSE_LLAMA=ON and add llama.cpp.\n";
        return 1;
    }

    GenerateParams gp;
    gp.n_ctx = args.n_ctx;
    gp.n_threads = args.n_threads;
    gp.seed = args.seed;

    if (!backend->load(args.model, gp)) {
        std::cerr << "Failed to load model: " << args.model << "\n";
        return 1;
    }

    std::vector<ChatMessage> history;
    history.push_back({"system", "You are a helpful assistant."});

    std::cout << "tokchat ready. Type your message and press Enter. Ctrl+C to quit.\n";

    std::deque<int32_t> recent; // track recent tokens for repetition penalties
    const size_t recent_max = 2048;

    std::string user_input;
    while (true) {
        std::cout << "\nUser> ";
        if (!std::getline(std::cin, user_input)) break;
        if (user_input == "/exit") break;
        if (user_input.empty()) continue;

        history.push_back({"user", user_input});

        // Build prompt and tokenize
        std::string prompt = build_prompt(history);
        auto prompt_tokens = backend->tokenize(prompt, /*add_bos*/ true);

        // Reset state and evaluate prompt
        backend->reset();
        // For long prompts, chunk to avoid huge single batch
        const int chunk = 512;
        for (size_t i = 0; i < prompt_tokens.size(); i += chunk) {
            size_t end = std::min(prompt_tokens.size(), i + chunk);
            std::vector<int32_t> span(prompt_tokens.begin() + i, prompt_tokens.begin() + end);
            if (!backend->eval(span)) {
                std::cerr << "\nEval error on prompt.\n";
                break;
            }
            for (auto t : span) {
                recent.push_back(t);
                if (recent.size() > recent_max) recent.pop_front();
            }
        }

        // Generate
        std::cout << "Assistant> " << std::flush;
        std::string partial_text;
        std::vector<int32_t> out_tokens;
        int32_t eos = backend->eos_token();

        for (int n = 0; n < args.max_new_tokens; ++n) {
            auto log = backend->logits();
            int32_t next = sample_next_token(
                log,
                std::vector<int32_t>(recent.begin(), recent.end()),
                args.temperature,
                args.top_k,
                args.top_p,
                args.repeat_penalty,
                args.frequency_penalty,
                args.presence_penalty
            );

            if (next == eos) {
                break;
            }

            // Append and stream
            out_tokens.push_back(next);
            // detokenize incrementally (naive; can be optimized with partial piece)
            std::string piece = backend->detokenize({next});
            std::cout << piece << std::flush;
            partial_text += piece;

            // Feed back for next token
            if (!backend->eval({next})) {
                std::cerr << "\nEval error during generation.\n";
                break;
            }
            recent.push_back(next);
            if (recent.size() > recent_max) recent.pop_front();

            // TODO: stop sequences check on partial_text (e.g., "</s>", or custom strings)
        }

        std::cout << std::endl;
        history.push_back({"assistant", partial_text});
    }

    return 0;
}