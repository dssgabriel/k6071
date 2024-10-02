#pragma once

#include <hip/hip_runtime.h>

#include <cassert>
#include <chrono>
#include <cstddef>

constexpr int error_exit_code = -1;

/**
 * Checks if the provided error code is hipSuccess and if not, prints an error message to the standard error output and
 * terminates the program with an error code.
 **/
#define HIP_CHECK(condition)                                                                                           \
  {                                                                                                                    \
    const hipError_t error = condition;                                                                                \
    if (error != hipSuccess) {                                                                                         \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " << __FILE__ << ':' << __LINE__   \
                << std::endl;                                                                                          \
      std::exit(error_exit_code);                                                                                      \
    }                                                                                                                  \
  }

class Timer {
public:
  Timer() { start = std::chrono::high_resolution_clock::now(); }

  auto seconds() -> double {
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    return std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
  }

private:
  std::chrono::high_resolution_clock::time_point start;
};

template <class T>
auto check_result(T const* C, size_t extent) -> void {
  for (size_t i = 0; i < extent; ++i) {
    for (size_t j = 0; j < extent; ++j) {
      assert(C[i * extent + j] == 1);
    }
  }
}
