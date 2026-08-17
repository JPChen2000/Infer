#ifndef FEATHER_DEMO_QWEN_CHAT_TEMPLATE_H
#define FEATHER_DEMO_QWEN_CHAT_TEMPLATE_H

#include <string>
#include <vector>

namespace feather {
namespace demo {

struct QwenChatMessage {
    std::string role;
    std::string content;
};

std::string RenderQwenChat(const std::vector<QwenChatMessage>& messages, bool add_generation_prompt);
std::string RenderQwenUserTurn(const std::string& content, bool add_generation_prompt = true);
std::string RenderQwenAssistantTurn(const std::string& content);

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_QWEN_CHAT_TEMPLATE_H
