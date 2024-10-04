cmake_minimum_required(VERSION 3.23)
project(K6071 LANGUAGES CXX)

set(CMAKE_COLOR_DIAGNOSTICS ON)

add_subdirectory(kokkos)

# Backend selection
option(K6071_HIP "Compile HIP version" OFF)
set(K6071_HIP ${K6071_ENABLE_HIP} CACHE BOOL "" FORCE)
option(K6071_ENABLE_RIGHT "Use Iterate::Right" OFF)
set(K6071_RIGHT ${K6071_ENABLE_RIGHT} CACHE BOOL "" FORCE)

option(K6071_ENABLE_CHECK_RES "Check that computed result is correct" OFF)
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

endif()
