#include "utils.hpp"

#include <cuda.h>

#include <iostream>
#include <vector>

#ifdef RIGHT
#warning Using `Iterate::Right`
#endif

template <class T>
__global__ void gemm_kernel(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
#ifdef RIGHT
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
#else
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
#endif

  if (i < M && j < N) {
    T tmp = 0.0;
#ifdef RIGHT
    for (size_t k = 0; k < K; ++k) {
      tmp += A[k * M + i] * B[j * K + k];
    }
    C[j * M + i] = tmp;
#else
    for (size_t k = 0; k < K; ++k) {
      tmp += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = tmp;
#endif
  }
}

template <class T>
auto gemm(T* A, size_t A_extent0, size_t A_extent1, T* B, size_t B_extent1, T* C, int T1, int T2) -> void {
  size_t M = A_extent0;
  size_t K = A_extent1;
  size_t N = B_extent1;

  dim3 block_dim(T2 > 0 ? T2 : 16, T1 > 0 ? T1 : 16);
  dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);

  gemm_kernel<T><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

template <class T>
auto benchmark(size_t N, size_t R, int T1, int T2) -> void {
  size_t elems = N * N;
  size_t bytes = elems * sizeof(T);
  double flops = 2.0 * static_cast<double>(N * N * N * R);

  std::vector<T> h_A(elems, 1);
  std::vector<T> h_B(elems, 1);
  std::vector<T> h_C(elems);

  T* d_A{};
  T* d_B{};
  T* d_C{};
  CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  gemm(d_A, N, N, d_B, N, d_C, T1, T2);
  CUDA_CHECK(cudaDeviceSynchronize());
  Timer timer;
  for (size_t r = 0; r < R; r++) {
    gemm(d_A, N, N, d_B, N, d_C, T1, T2);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  double time = timer.seconds();

  std::cout << "Flops: " << flops << ", Bytes: " << sizeof(T) << ", GFlop/s: " << flops / time * 1.0e-9 << "\n";

  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

#ifdef CHECK
  check_result(h_C.data(), N);
#endif

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

auto main(int argc, char* argv[]) -> int {
  // Matrix dimensions
  size_t N = argc > 1 ? static_cast<size_t>(std::stoi(argv[1])) : 4096;
  // Benchmark repetitions
  size_t R = argc > 2 ? static_cast<size_t>(std::stoi(argv[2])) : 11;
  // Block sizes to use when running the HIP kernel
  int T1 = argc > 3 ? std::stoi(argv[3]) : -1;
  int T2 = argc > 4 ? std::stoi(argv[4]) : T1;

  benchmark<double>(N, R, T1, T2);
  benchmark<float>(N, R, T1, T2);

  return 0;
}
