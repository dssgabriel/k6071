cmake_minimum_required(VERSION 3.23)
project(K6071 LANGUAGES CXX)

# Enable color for Ninja output
set(CMAKE_COLOR_DIAGNOSTICS ON)

add_subdirectory(kokkos)

# Backend selection
option(K6071_HIP "Compile HIP version" OFF)
set(K6071_HIP ${K6071_ENABLE_HIP} CACHE BOOL "" FORCE)
option(K6071_ENABLE_CUDA "Compile CUDA version" OFF)
set(K6071_CUDA ${K6071_ENABLE_CUDA} CACHE BOOL "" FORCE)

# Enable iteration with Layout Right (row-major) or not
option(K6071_ENABLE_RIGHT "Use Iterate::Right" OFF)
set(K6071_RIGHT ${K6071_ENABLE_RIGHT} CACHE BOOL "" FORCE)

# Enable checking result
option(K6071_ENABLE_CHECK_RES "Check that computed result is correct" ON)
set(K6071_CHECK_RES ${K6071_ENABLE_CHECK_RES} CACHE BOOL "" FORCE)

add_library(utils INTERFACE)
target_sources(
  utils
  INTERFACE
  FILE_SET HEADERS
  BASE_DIRS ${PROJECT_SOURCE_DIR}/src
  FILES src/utils.hpp
)
target_compile_features(utils INTERFACE cxx_std_20)
set_target_properties(utils PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(
  utils
  INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
)

# Pure Kokkos reproducer
add_executable(kok)
target_sources(kok PRIVATE src/kokkos.cpp)
target_compile_features(kok PRIVATE cxx_std_20)
set_target_properties(kok PROPERTIES CXX_EXTENSIONS OFF)
if(K6071_RIGHT)
  target_compile_definitions(kok PRIVATE -DRIGHT)
endif()
if(K6071_CHECK_RES)
  target_compile_definitions(kok PRIVATE -DCHECK)
endif()
target_compile_options(
  kok
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
)
target_link_libraries(kok Kokkos::kokkos utils)

# If enabled, compile a pure HIP implementation
if(K6071_HIP)
  add_executable(hip)
  target_sources(hip PRIVATE src/hip.cpp)
  target_compile_features(hip PRIVATE cxx_std_20)
  set_target_properties(hip PROPERTIES CXX_EXTENSIONS OFF)
  if(K6071_RIGHT)
    target_compile_definitions(hip PRIVATE -DRIGHT)
  endif()
  if(K6071_CHECK_RES)
    target_compile_definitions(hip PRIVATE -DCHECK)
  endif()
  target_compile_options(
    hip
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wshadow
  )
  target_link_libraries(hip utils)
endif()

# If enabled, compile a pure CUDA implementation
if(K6071_CUDA)
  enable_language(CUDA)

  add_executable(cuda)
  target_sources(cuda PRIVATE src/cuda.cu)
  target_compile_features(cuda PRIVATE cxx_std_20)
  set_target_properties(cuda PROPERTIES CXX_EXTENSIONS OFF)
  set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(K6071_RIGHT)
    target_compile_definitions(cuda PRIVATE -DRIGHT)
  endif()
  if(K6071_CHECK_RES)
    target_compile_definitions(cuda PRIVATE -DCHECK)
  endif()
  target_compile_options(
    cuda
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wshadow
  )
  target_link_libraries(cuda utils)
endif()

# Test for understanding how MDRangePolicy and Iterate::* work.
add_executable(test)
target_sources(test PRIVATE src/test.cpp)
target_compile_features(test PRIVATE cxx_std_20)
set_target_properties(test PROPERTIES CXX_EXTENSIONS OFF)
if(K6071_RIGHT)
  target_compile_definitions(test PRIVATE -DRIGHT)
endif()
target_compile_options(
  test
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
)
target_link_libraries(test Kokkos::kokkos)
