#pragma once

#include <cmath>
#include <chrono>
#include <cstddef>
#include <limits>

constexpr int error_exit_code = -1;

/**
 * Checks if the provided error code is hipSuccess and if not, prints an error message to the standard error output and
 * terminates the program with an error code.
 **/
#define HIP_CHECK(condition)                                                                                           \
  {                                                                                                                    \
    hipError_t const err = condition;                                                                                  \
    if (err != hipSuccess) {                                                                                           \
      std::cerr << "HIP Runtime error at: " << __FILE__ << ":" << __LINE__ << "\n"                                     \
                << "  " << hipGetErrorString(err) << "\n";                                                             \
      std::exit(error_exit_code);                                                                                      \
    }                                                                                                                  \
  }

#define CUDA_CHECK(condition)                                                                                          \
  {                                                                                                                    \
    cudaError_t const err = condition;                                                                                 \
    if (err != cudaSuccess) {                                                                                          \
      std::cerr << "CUDA Runtime error at: " << __FILE__ << ":" << __LINE__ << "\n"                                    \
                << "  " << cudaGetErrorString(err) << "\n";                                                            \
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
auto check_result(T* const C, size_t extent) -> bool {
  constexpr T tol = std::numeric_limits<T>::epsilon();
  constexpr T one = 1.0;
  for (size_t i = 0; i < extent * extent; ++i) {
    auto val = C[i];
    if (tol > std::abs(val - one)) {
      return false;
    }
  }
  return true;
}
