# Fleet audit — nir-c-runtime

Fleet-audit record for
[Capsize-Games/nir-c-runtime#7](https://github.com/Capsize-Games/nir-c-runtime/issues/7).
Docs only. Verified against `master` on 2026-10-13.

## 1. Identity, boundary, owner, runtime

- **Canonical identity.** GitHub `Capsize-Games/nir-c-runtime`
  (public, default branch `master` — this repository's default is
  not `main`); local checkout `Projects/spikeforge/nir-c-runtime`.
- **Owner and license.** Capsize LLC, BSD-3-Clause with existing
  attribution intact. Public history; no rewrite or deletion is
  authorized by this audit.
- **Product boundary.** A portable C11 library that loads the
  supported subset of NIR graph exports and runs inference without
  Python or PyTorch. It is a library boundary: not a replacement
  for the Python spikeforge trainer and not a project-specific
  hardware port. The public C API lives in
  `include/nir_c_runtime/`; the implementation lives in `src/`.
  Unsupported graph node kinds fail with explicit errors rather
  than silent reinterpretation.
- **Build/runtime matrix.** CMake `>=3.20`, C11, compiled with
  `-Wall -Wextra -Werror` as the warning gate. CI runs on
  GitHub-hosted Ubuntu workers: a Release/Debug test matrix plus
  an AddressSanitizer/UBSan pass, both via `ctest`.
- **Deployment channel.** Source library; there is no packaging or
  publishing workflow (the only workflow is `ci.yml`). Consumers
  take the source tree. The committed fixtures under
  `tests/fixtures/` are small and hand-inspectable; the real-model
  timing and bit-exact reference result in `README.md` are
  evidence for the current MVP, not a promise about every future
  graph or target.
- **Migration metadata.** `capsize.json` and `justfile` are
  present; `just` is the canonical local task runner.

## 2. Dependency provenance

- **Runtime dependencies: none beyond the C toolchain.** The
  library links only `m` (libm); `CMakeLists.txt` fetches no
  third-party code and there are no vendored dependencies.
- **Dev-only Python tool.** `tools/export_fixture.py` requires
  spikeforge and torch on the Python path, but it is used only to
  regenerate the committed reference fixtures. Ordinary C builds
  and `ctest` runs never need a Python environment.
- **Fixtures.** Produced from spikeforge's own reference
  `NirInterpreter` through the exporter above; committed to the
  repo and read by the C tests. No model weights, datasets, or
  credentials are fetched at build or test time.

## 3. Clean build/test path

From a clean checkout (no `build/` tree; `build/` and
`build-asan/` are gitignored generated directories and must not
be committed):

```bash
just setup   # configure a Debug CMake tree in build/
just build   # compile the library and tests
just test    # run ctest with output on failure
just ci      # setup + warning-gated build + tests
```

Raw equivalents are `cmake -S . -B build`,
`cmake --build build --parallel`, and
`ctest --test-dir build --output-on-failure`. There is no separate
C formatter or static type checker; `format` and `typecheck` are
explicit no-op recipes and the compiler warnings are the gate.
`ctest` registers the unit tests (`test_json`, `test_graph`,
`test_ops`, `test_conv_net`, `test_fixed_event`, `test_quantized`,
`test_workspace`, `test_tensor`), a JSON OOM sweep, and twelve
`test_graph_bias_*` cases.

## 4. Adopt / retain-local decision

**Retain local. Shared-code candidates: zero.**

- No Python or TypeScript dependency is proposed: adding one
  would violate this repository's own hard rules (single CMake
  build, no vendored Python runtime) and the audit's instruction
  not to force web abstractions into a native product.
- Language-appropriate sharing was considered and declined:
  there is no shared C++ header set, CMake toolchain module,
  asset template, or release automation in the fleet that this
  six-file C library needs. Its `CMakeLists.txt` files are
  already minimal and product-specific.
- Product behavior (graph semantics, ABI, fixture/regeneration
  loop) stays local. Any future extraction would need an exact,
  behavior-tested proposal with a clear owner and release
  cadence; none exists today.

No follow-up extraction issue is filed. If the fleet later gains
a shared CMake/C native template worth adopting, that adoption
would be its own focused change with before/after footprint
evidence — not part of this audit.
