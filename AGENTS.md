# AGENTS.md — nir-c-runtime

Canonical, tool-neutral instructions for this repository.

**Manifest:** [`capsize.json`](capsize.json) · **Standards:** CAPSIZE 1.x

## What this repository is

`nir-c-runtime` is a portable C11 library for loading the supported subset of
NIR graph exports and running inference without Python or PyTorch. It is a
library boundary, not a replacement for the Python spikeforge trainer or for a
project-specific hardware port.

## Hard rules

1. Keep the public C API in `include/nir_c_runtime/` and the implementation in
   `src/`. Add a test or fixture when changing graph semantics or the ABI.
2. Use the existing CMake build. Do not introduce a second build system or
   vendor a Python runtime into this library.
3. Preserve the explicit unsupported-node errors. A graph kind that is not
   implemented must fail clearly rather than being silently reinterpreted.
4. Treat `build/` and `build-asan/` as generated directories. They are local
   build trees and must not be committed.
5. Keep the BSD-3-Clause license and existing attribution intact.
6. `tools/export_fixture.py` depends on spikeforge and is only needed when the
   committed reference fixtures must be regenerated; ordinary C tests do not
   require that Python environment.

## Task interface

Use `just` for the canonical local tasks:

```text
just setup    # configure a Debug CMake tree
just build   # compile the library and tests
just test    # run ctest
just ci      # build/check the configured tree and run tests
```

The repository has no separate C formatter or static type checker today;
`format` and `typecheck` are explicit no-op recipes. CMake's `-Wall -Wextra
-Werror` compilation is the current warning gate.

## Validation boundaries

This repository is public. CI must use GitHub-hosted runners, with Debug and
Release builds plus an AddressSanitizer/UBSan pass. Do not attach a workstation
runner or route pull-request jobs to a private machine. The owner authorized
public visibility and this CI migration on 2026-10-12.

The committed fixtures are small and hand-inspectable. The real-model timing
and bit-exact reference result described in `README.md` are evidence for the
current MVP, not a promise that every future graph or hardware target has been
validated.
