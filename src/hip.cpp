#include "utils.hpp"

#include <iostream>

#ifdef RIGHT
#warning Using `Iterate::Right`
#endif

template <class T>
__global__ void gemm_kernel(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
#ifdef RIGHT
  size_t m = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = blockIdx.y * blockDim.y + threadIdx.y;
#else
  size_t m = blockIdx.y * blockDim.y + threadIdx.y;
  size_t n = blockIdx.x * blockDim.x + threadIdx.x;
#endif

  if (m < M && n < N) {
    T tmp = 0.0;
    for (size_t k = 0; k < K; ++k) {
      tmp += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = tmp;
  }
}

template <class T>
auto gemm(Kokkos::View<T**> A, Kokkos::View<T**> B, Kokkos::View<T**> C, int T1, int T2) -> void {
  size_t M = A.extent(0);
  size_t K = A.extent(1);
  size_t N = B.extent(1);

  dim3 block_dim(T2 > 0 ? T2 : 16, T1 > 0 ? T1 : 16);
  dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);

  gemm_kernel<T>
    <<<grid_dim, block_dim, 0, hipStreamDefault>>>(
      A.data(), B.data(), C.data(), M, N, K
    );
}

template <class T>
auto benchmark(size_t N, size_t R, int T1, int T2) -> void {
  double flops = 2.0 * static_cast<double>(N * N * N * R);

  Kokkos::View<T**> A("A", N, N);
  Kokkos::View<T**> B("B", N, N);
  Kokkos::View<T**> C("C", N, N);
  Kokkos::deep_copy(A, 1);
  Kokkos::deep_copy(B, 1);
  Kokkos::deep_copy(C, 1);

  gemm(A, B, C, T1, T2);
  Kokkos::fence();
  Kokkos::Timer timer;
  for (size_t r = 0; r < R; r++) {
    gemm(A, B, C, T1, T2);
  }
  Kokkos::fence();
  double time = timer.seconds();

  std::cout << "Flops: " << flops << ", Bytes: " << sizeof(T) << ", GFlop/s: " << flops / time * 1.0e-9 << "\n";
#ifdef CHECK
  check_result(h_C.data(), N);
#endif
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  {
    size_t N = argc > 1 ? static_cast<size_t>(std::stoi(argv[1])) : 4096;
    size_t R = argc > 2 ? static_cast<size_t>(std::stoi(argv[2])) : 11;
    int T1   = argc > 3 ? std::stoi(argv[3]) : -1;
    int T2   = argc > 4 ? std::stoi(argv[4]) : T1;

    benchmark<double>(N, R, T1, T2);
    benchmark<float>(N, R, T1, T2);
#if KOKKOS_VERSION >= 30700
    benchmark<Kokkos::Experimental::half_t>(N, R, T1, T2);
#endif
  }
  Kokkos::finalize();

  return 0;
}
