cmake_minimum_required(VERSION 3.20)
project(nir_c_runtime C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

add_compile_options(-Wall -Wextra -Werror)

add_library(nir_c_runtime
  src/json.c
  src/tensor.c
  src/graph.c
  src/ops.c
  src/runtime.c
)
target_include_directories(nir_c_runtime PUBLIC include)
target_link_libraries(nir_c_runtime PUBLIC m)

enable_testing()
add_subdirectory(tests)
