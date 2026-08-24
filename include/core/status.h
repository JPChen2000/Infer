#ifndef FEATHER_CORE_STATUS_H
#define FEATHER_CORE_STATUS_H

#include <string>
#include <utility>

namespace feather {

enum class StatusCode {
    kOk = 0,
    kInvalidArgument,
    kNotFound,
    kShapeMismatch,
    kUnsupported,
    kBuildFailed,
    kExecutionFailed,
    kInternal,
};

class Status {
   public:
    Status() = default;
    explicit Status(StatusCode code) : code_(code) {}
    Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

    static Status Ok() { return Status(); }
    static Status Error(StatusCode code, std::string message) { return Status(code, std::move(message)); }

    bool ok() const { return code_ == StatusCode::kOk; }
    explicit operator bool() const { return ok(); }
    StatusCode code() const { return code_; }
    const std::string& message() const { return message_; }

   private:
    StatusCode code_{StatusCode::kOk};
    std::string message_;
};

}  // namespace feather

#endif  // FEATHER_CORE_STATUS_H
