#ifndef FEATHER_UTIL_LOGGER_H
#define FEATHER_UTIL_LOGGER_H

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace feather {

inline bool IsDebugLoggingEnabled() {
#ifdef FEATHER_DEBUG
    return true;
#else
    return false;
#endif
}

inline std::string CurrentDateTime() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    return buf;
}

template <typename... Args>
inline std::string FormatString(const std::string& format, Args... args) {
    size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

}  // namespace feather

#ifdef FEATHER_DEBUG
#define LOG(level, format, ...)                                                                                   \
    do {                                                                                                          \
        std::stringstream ss;                                                                                     \
        ss << ::feather::CurrentDateTime() << " [" << #level << "] "                                             \
           << ::feather::FormatString(format, ##__VA_ARGS__) << std::endl;                                       \
        std::cout << ss.str();                                                                                    \
    } while (0)
#define LOG_DEBUG(format, ...) LOG(DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) LOG(INFO, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) LOG(WARN, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) LOG(ERROR, format, ##__VA_ARGS__)
#else
#define LOG(level, format, ...)                                                                                   \
    do {                                                                                                          \
    } while (0)
#define LOG_DEBUG(format, ...)                                                                                    \
    do {                                                                                                          \
    } while (0)
#define LOG_INFO(format, ...)                                                                                     \
    do {                                                                                                          \
    } while (0)
#define LOG_WARN(format, ...)                                                                                     \
    do {                                                                                                          \
    } while (0)
#define LOG_ERROR(format, ...)                                                                                    \
    do {                                                                                                          \
    } while (0)
#endif

#define CHECK(expr)                                                                                                  \
    do {                                                                                                             \
        if (!(expr)) {                                                                                               \
            std::cerr << "CHECK failed: " << #expr << " in file " << __FILE__ << ", line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                            \
    } while (false)

#endif  // FEATHER_UTIL_LOGGER_H
