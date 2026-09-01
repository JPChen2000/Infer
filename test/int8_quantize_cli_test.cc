#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "quant/static_quantizer.h"

namespace {

TEST(int8_quantize_cli_test, ScaleTableRejectsExtraColumns) {
    const std::string path = "/tmp/int8_quantize_cli_extra_columns.txt";
    std::ofstream file(path);
    file << "input 0.5 0 extra\n";
    file.close();
    feather::ActivationQuantizationTable table;
    std::vector<std::string> diagnostics;
    EXPECT_EQ(feather::LoadActivationQuantizationTable(path, &table, &diagnostics), -1);
    EXPECT_FALSE(diagnostics.empty());
}

}  // namespace
