# nir-c-runtime

[![Status: pre-1.0](https://img.shields.io/badge/status-pre--1.0-orange.svg)](#status)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://capsizegames.com/discord)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![C Standard: C11](https://img.shields.io/badge/C-C11-00599C.svg)](CMakeLists.txt)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Capsize-Games/nir-c-runtime/pulls)

A minimal C runtime for executing [NIR](https://github.com/neuromorphs/NIR)
(Neuromorphic Intermediate Representation) graphs — specifically, graphs
exported by [spikeforge](https://github.com/capsize-games/spikeforge) — on
embedded and edge targets where a Python/PyTorch runtime isn't an option.

## Why this exists

Spikeforge trains and exports spiking neural networks in Python, and can
export a trained topology to NIR. NIR is a portable graph format, but
nothing in the spikeforge ecosystem runs that graph without Python and
PyTorch. This project is the missing piece: a small, dependency-light C
implementation that loads a NIR-exported graph and its weights and runs
inference — the kind of thing a government or embedded-systems
buyer actually wants to see (C, not Python, on the target hardware), per
[MISRA-C](https://www.misra.org.uk/)/DO-178C-adjacent embedded conventions.

This is deliberately a separate repository from any project-specific
research (dust-robust lunar vision, etc.) — this runtime is generic to any
NIR-exported spikeforge model, not tied to one use case.

## Status

Working MVP. Loads a `graph_summary()`-shaped NIR JSON export and runs
inference for the exact node kinds spikeforge's `conv_net` topology emits:
`Input`, `Conv2d`, `Affine`, `Flatten`, `AvgPool2d`, `SumPool2d`, `LI`,
`Threshold`, `Scale`, `Delay`, `Output`. `LIF`/`IF`/`CubaLIF`/`Conv1d`/
`Linear` are not implemented; loading a graph that uses them fails with a
named "unsupported node kind" error rather than silently misinterpreting
them.

**Validated, not just "it runs":** `tests/test_conv_net.c` checks a small
topology against spikeforge's own reference `NirInterpreter` output
(`tools/export_fixture.py` produces both from one run). It was also checked
by hand against a real trained 128×128, 2-channel, 11-class `conv_net`
checkpoint (not committed here) — **bit-exact match** with the Python
reference interpreter (max abs diff `0.0`). All three test executables also
pass clean under AddressSanitizer + UBSan.

**Known limitation:** the convolution/pooling implementation is naive
nested loops, not vectorized or blocked. On the 128×128 real-model check
above it took ~148ms for 16 timesteps versus PyTorch's ~20ms on the same
machine — expected for a correctness-first MVP, not yet tuned, and the
real target hardware for this runtime (an MCU with no PyTorch alternative
at all) makes a different comparison than "faster than a desktop GPU
build" anyway. Worth optimizing before any latency claim ships.

## Scope

- [x] Parse a NIR JSON graph export for the node kinds above.
- [x] Load full-precision (float32) weights from the export.
- [x] Run a fixed-timestep forward pass, validated against spikeforge's own
      reference NIR interpreter.
- [ ] Load int8/uint8-quantized weights (matching spikeforge_targets'
      declared quantization schemes) — not yet started.
- [ ] No dynamic allocation after initialization / freestanding-C11
      compatibility for a real MCU target — the current implementation
      allocates a fresh tensor per node per step via malloc/calloc, which
      is fine for validation but not yet appropriate for a hard-realtime
      or freestanding embedded target.
- [ ] Vectorized/blocked conv2d (see the timing note above).

## Out of scope (for now)

- Training. This is inference-only.
- Any Curlee/JOE integration — tracked separately as a possible future
  target once [curlee#305](https://github.com/w4ffl35/curlee/issues/305)
  and [curlee#306](https://github.com/w4ffl35/curlee/issues/306) land.
- A specific hardware target. This targets portable C first; a specific
  MCU/board port is a later, separate scope once the core runtime exists.

## Building and testing

```sh
cmake -S . -B build
cmake --build build
cd build && ctest --output-on-failure
```

`tests/fixtures/` is committed (small, hand-inspectable); regenerate it
with `tools/export_fixture.py` (requires spikeforge on the Python path,
e.g. a lunar-snn-style venv with `spikeforge[nir]` installed) if the
reference interpreter's semantics ever change.

## Citing

If nir-c-runtime is useful in your research or deployment, please cite it —
see [CITATION.cff](CITATION.cff) (GitHub renders a "Cite this repository"
button from it automatically). To cite the spikeforge toolkit that trains
and exports the models this runtime executes, see
[spikeforge's own CITATION.cff](https://github.com/capsize-games/spikeforge/blob/main/CITATION.cff)
instead.

## License

BSD-3-Clause (matching spikeforge's own license) — see [LICENSE](LICENSE).
