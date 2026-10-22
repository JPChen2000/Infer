// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
