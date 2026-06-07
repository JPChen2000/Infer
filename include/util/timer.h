#ifndef FEATHER_UTIL_TIMER_H
#define FEATHER_UTIL_TIMER_H

#include "util/logger.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

class Timer {
   public:
    Timer() {
#ifdef FEATHER_DEBUG
        start_ = std::chrono::high_resolution_clock::now();
#endif
    }

    Timer(const std::string& name) {
#ifdef FEATHER_DEBUG
        start_ = std::chrono::high_resolution_clock::now();
        name_ = name;
#else
        (void)name;
#endif
    }

    ~Timer() {
#ifdef FEATHER_DEBUG
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = end - start_;
        long time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
        int32_t unit_type_idx = 0;
        while (time > 1000 && unit_type_idx < 3) {
            time /= 1000;
            ++unit_type_idx;
        }
        LOG_INFO("[Timer] %s cost %ld %s", name_.c_str(), time, unit_types_[unit_type_idx].c_str());
#endif
    }

   private:
#ifdef FEATHER_DEBUG
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::string name_;
    std::vector<std::string> unit_types_ = {"us", "ms", "s"};
#endif
};

class AutoTimer {
  public:
    AutoTimer() : m_timer() {}
    AutoTimer(const std::string& name) : m_timer(name) {}

  private:
    Timer m_timer;
};

#endif  // FEATHER_UTIL_TIMER_H
