#ifndef FEATHER_TEST_PYTHON_TEST_UTILS_H
#define FEATHER_TEST_PYTHON_TEST_UTILS_H

#include <cstdlib>
#include <filesystem>
#include <string>

namespace feather { namespace test {
inline std::string ShellQuote(const std::string& value) {
    std::string result = "'";
    for (const char ch : value) result += ch == '\'' ? "'\\''" : std::string(1, ch);
    return result + "'";
}
inline std::string PythonCommand() {
    const auto* python = std::getenv("FEATHER_TEST_PYTHON");
    return ShellQuote(python && *python ? python : "python3");
}
inline bool HasPythonModules(const std::string& modules) {
    const auto command = PythonCommand() + " -c " + ShellQuote("import " + modules) + " >/dev/null 2>&1";
    return std::system(command.c_str()) == 0;
}
inline std::filesystem::path OptionalModelPath(const char* variable, const std::filesystem::path& fallback) {
    const auto* path = std::getenv(variable);
    return path && *path ? std::filesystem::path(path) : fallback;
}
} }  // namespace feather::test
#endif
