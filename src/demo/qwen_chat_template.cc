#include "demo/qwen_chat_template.h"

#include <cctype>

namespace feather {
namespace demo {

namespace {

std::string TrimAsciiWhitespace(const std::string& text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

std::string RenderMessage(const QwenChatMessage& message) {
    const std::string content = TrimAsciiWhitespace(message.content);
    if (message.role == "system" || message.role == "user") {
        return "<|im_start|>" + message.role + "\n" + content + "<|im_end|>\n";
    }
    if (message.role == "assistant") {
        return "<|im_start|>assistant\n" + content + "<|im_end|>\n";
    }
    return "";
}

}  // namespace

std::string RenderQwenChat(const std::vector<QwenChatMessage>& messages, bool add_generation_prompt) {
    if (messages.empty()) {
        return "";
    }
    std::string rendered;
    for (size_t index = 0; index < messages.size(); ++index) {
        if ((messages[index].role == "system" && index != 0) ||
            (messages[index].role != "system" && messages[index].role != "user" &&
             messages[index].role != "assistant")) {
            return "";
        }
        rendered += RenderMessage(messages[index]);
    }
    if (add_generation_prompt) {
        rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }
    return rendered;
}

std::string RenderQwenUserTurn(const std::string& content, bool add_generation_prompt) {
    return RenderQwenChat({{"user", content}}, add_generation_prompt);
}

std::string RenderQwenAssistantTurn(const std::string& content) {
    return RenderQwenChat({{"assistant", content}}, false);
}

}  // namespace demo
}  // namespace feather
