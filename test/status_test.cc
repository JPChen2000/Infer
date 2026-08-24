#include <gtest/gtest.h>

#include "core/status.h"

namespace {

TEST(status_test, DefaultsToOk) {
    const feather::Status status;
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(status.code(), feather::StatusCode::kOk);
    EXPECT_TRUE(status.message().empty());
}

TEST(status_test, PreservesCodeAndDiagnostic) {
    const auto status = feather::Status::Error(feather::StatusCode::kNotFound, "tensor: input");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), feather::StatusCode::kNotFound);
    EXPECT_EQ(status.message(), "tensor: input");
}

}  // namespace
