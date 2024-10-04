#include <Kokkos_Core.hpp>

#include <cmath>
#include <iostream>

#ifdef RIGHT
#warning Using `Iterate::Right`
using Iterate = Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>;
#else
using Iterate = Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>;
#endif

template <class T>
using V = Kokkos::View<T**, Kokkos::LayoutLeft>;

template <class T>
auto iter(V<T> A) -> void {
  size_t M = A.extent(0);
  size_t N = A.extent(1);
  Kokkos::MDRangePolicy<Iterate> p({0, 0}, {M, N});
  Kokkos::parallel_for(
    "A", p,
    #ifdef RIGHT
    KOKKOS_LAMBDA(size_t j, size_t i) {
    #else
    KOKKOS_LAMBDA(size_t i, size_t j) {
    #endif
      Kokkos::printf("A(%d, %d) = %lf\n", i, j, A(i, j));
    }
  );
}

template <class T>
auto benchmark(size_t N) -> void {
  V<T> A("A", N, N);
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < N; ++i) {
      A(i, j) = i * 10 + j;
    }
  }
  iter(A);
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  {
    size_t N = argc > 1 ? static_cast<size_t>(std::stoi(argv[1])) : 4096;
    benchmark<double>(N);
  }
  Kokkos::finalize();

  return 0;
}
