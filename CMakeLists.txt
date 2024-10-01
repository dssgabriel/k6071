cmake_minimum_required(VERSION 3.23)
project(K6071 LANGUAGES CXX)

add_subdirectory(kokkos)

option(K6071_ENABLE_RIGHT "Use Iterate::Right" OFF)
set(K6071_RIGHT ${K6071_ENABLE_RIGHT} CACHE BOOL "" FORCE)

add_executable(k6071)
target_sources(k6071 PRIVATE src/kokkos.cpp)
target_compile_features(k6071 PRIVATE cxx_std_20)
set_target_properties(k6071 PROPERTIES CXX_EXTENSIONS OFF)
if(K6071_RIGHT)
  target_compile_definitions(k6071 PRIVATE -DRIGHT)
endif()
target_compile_options(
  k6071
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
)
target_link_libraries(k6071 Kokkos::kokkos)
