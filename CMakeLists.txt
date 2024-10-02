cmake_minimum_required(VERSION 3.23)
project(K6071 LANGUAGES CXX)

add_subdirectory(kokkos)

option(K6071_ENABLE_RIGHT "Use Iterate::Right" OFF)
set(K6071_RIGHT ${K6071_ENABLE_RIGHT} CACHE BOOL "" FORCE)

option(K6071_ENABLE_CHECKS "Check that computed result is correct" OFF)
set(K6071_CHECKS ${K6071_ENABLE_CHECKS} CACHE BOOL "" FORCE)

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

add_executable(hip6071)
target_sources(hip6071 PRIVATE src/hip.cpp)
target_compile_features(hip6071 PRIVATE cxx_std_20)
set_target_properties(hip6071 PROPERTIES CXX_EXTENSIONS OFF)
if(K6071_RIGHT)
  target_compile_definitions(hip6071 PRIVATE -DRIGHT)
endif()
if(K6071_CHECKS)
  target_compile_definitions(hip6071 PRIVATE -DCHECK)
endif()
target_compile_options(
  hip6071
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
)
target_link_libraries(hip6071 utils)
