#pragma once
#include <string>
#include <vector>

// Simple role-based message
struct ChatMessage {
    std::string role;   // "system", "user", "assistant"
    std::string content;
};

// Build a prompt string from chat messages.
// Adjust this to the template your model expects (Llama-2, ChatML, Qwen, etc.)
std::string build_prompt(const std::vector<ChatMessage>& messages);