# tokchat (C++ token-based LLM chat skeleton)

This is a minimal C++ skeleton for building a token-based chat model:
- Uses llama.cpp as the inference backend (GGUF models).
- Implements: prompt formatting, tokenization, autoregressive generation loop, temperature/top-k/top-p sampling, repetition/frequency/presence penalties, stop-strings, and streaming output.
- Backend-agnostic design: you can replace llama.cpp with your own backend.

## Directory
- src/, include/: chat pipeline + sampling + llama backend
- third_party/llama.cpp: llama.cpp submodule (optional, see below)

## Prerequisites
- CMake ≥ 3.20
- C++17 or later
- (Optional) A GGUF model compatible with llama.cpp (e.g., LLaMA, Mistral, Qwen, etc.)
- (Optional) Build llama.cpp as a subdirectory to link against `llama` lib

## Getting started

1) Add llama.cpp as a submodule (optional but recommended):
   git submodule add https://github.com/ggerganov/llama.cpp third_party/llama.cpp
   git submodule update --init --recursive

2) Build:
   mkdir build && cd build
   cmake -DUSE_LLAMA=ON ..
   cmake --build . --config Release

3) Run:
   ./tokchat --model /path/to/model.gguf --threads 8 --ctx 4096 --top-k 40 --top-p 0.95 --temp 0.7 --repeat-penalty 1.1

   Then type your user message and press Enter to generate a streaming reply.

## Notes
- Prompt formatting here is simple and model-agnostic. If your model expects a specific chat template (e.g., Llama-2, ChatML, Qwen), adjust in `chat_format.cpp`.
- Sampling implements temperature, top-k, top-p, repetition penalty, frequency/presence penalties. Tune them per model.
- Stop sequences can be added easily (see TODO in main.cpp).
- The llama.cpp API changes over time; refer to their examples if a function signature differs.

## Replace backend
- Implement `include/model_backend.h` for your engine (ONNX Runtime, TensorRT, custom CUDA).
- Replace `src/llama_backend.cpp` or add another file; wire it in CMake.
