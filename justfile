set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

setup:
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug

build:
    cmake --build build --parallel

test:
    ctest --test-dir build --output-on-failure

lint:
    cmake --build build --parallel

format:
    @echo "No repository formatter is configured."

typecheck:
    @echo "No separate C type checker is configured; compiler warnings are -Werror."

run:
    @echo "nir-c-runtime is a library; run the test executables through ctest."

clean:
    cmake --build build --target clean

docs:
    @echo "See README.md and CITATION.cff"

ci: setup lint typecheck test
