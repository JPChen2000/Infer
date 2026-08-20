#ifndef FEATHER_DEMO_QWEN_DEMO_H
#define FEATHER_DEMO_QWEN_DEMO_H

#include <string>

namespace feather {
namespace demo {

// Buffers token-decoded bytes so callers only print complete UTF-8 sequences.
class QwenUtf8Stream {
   public:
    std::string Push(const std::string& chunk);
    std::string Finish();

   private:
    std::string pending_;
};

int RunQwenDemo(int argc, char** argv);

}  // namespace demo
}  // namespace feather

#endif  // FEATHER_DEMO_QWEN_DEMO_H
