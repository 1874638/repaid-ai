#include "chat_format.h"
#include <sstream>

std::string build_prompt(const std::vector<ChatMessage>& messages) {
    // Example: Llama-2 style minimal template
    // [INST] <<SYS>> {system} <</SYS>> {user} [/INST]
    // {assistant}
    std::ostringstream oss;

    std::string system;
    for (auto& m : messages) {
        if (m.role == "system") {
            if (!system.empty()) system += "\n";
            system += m.content;
        }
    }

    bool first_user = true;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& m = messages[i];
        if (m.role == "user") {
            if (first_user) {
                first_user = false;
                oss << "[INST] ";
                if (!system.empty()) {
                    oss << "<<SYS>>\n" << system << "\n<</SYS>>\n";
                }
                oss << m.content << " [/INST]\n";
            } else {
                oss << "[INST] " << m.content << " [/INST]\n";
            }
        } else if (m.role == "assistant") {
            oss << m.content << "\n";
        }
    }

    return oss.str();
}