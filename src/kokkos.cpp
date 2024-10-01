#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>

#ifdef RIGHT
#warning Using `Iterate::Right`
using Iterate = Kokkos::Rank<2, Kokkos::Iterate::Right>;
#else
using Iterate = Kokkos::Rank<2, Kokkos::Iterate::Left>;
#endif

template <class T>
auto gemm(Kokkos::View<T**> A, Kokkos::View<T**> B, Kokkos::View<T**> C, int T1, int T2) -> void {
  using mdrange_policy = Kokkos::MDRangePolicy<Iterate>;
  size_t M             = A.extent(0);
  size_t K             = A.extent(1);
  size_t N             = B.extent(1);

  mdrange_policy p;
  // Is tile size specified?
  if (T1 > 0 && T2 > 0) {
    p = mdrange_policy({0, 0}, {M, N}, {T1, T2});
  } else {
    p = mdrange_policy({0, 0}, {M, N});
  }

  Kokkos::parallel_for(
    "AxB=C",
    p,
#ifdef RIGHT
    KOKKOS_LAMBDA(size_t n, size_t m) {
#else
    KOKKOS_LAMBDA(size_t m, size_t n) {
#endif
      T tmp = 0.0;
      for (size_t k = 0; k < K; ++k) {
        tmp += A(m, k) * B(k, n);
      }
      C(m, n) = tmp;
    }
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
