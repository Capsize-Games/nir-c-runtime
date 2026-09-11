cmake_minimum_required(VERSION 3.20)
project(nir_c_runtime C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# No sources yet -- scaffolding only, see README.md "Status". A future
# implementation pass adds src/*.c here and a proper test target under
# tests/ once there is something to build and validate against
# spikeforge's own reference NIR interpreter.
