# Kokkos issue #6071

Investigate performance issues with `Iterate::Right` being slower than `Iterate::Left` when using `MDRangePolicy` on AMD HIP.

## Build

```sh
cmake -B <build_dir> -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc [-DK6071_ENABLE_RIGHT=ON]
cmake --build <build_dir> -j
```
