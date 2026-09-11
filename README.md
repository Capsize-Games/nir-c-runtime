# nir-c-runtime

A minimal C runtime for executing [NIR](https://github.com/neuromorphs/NIR)
(Neuromorphic Intermediate Representation) graphs — specifically, graphs
exported by [spikeforge](https://github.com/capsize-games/spikeforge) — on
embedded and edge targets where a Python/PyTorch runtime isn't an option.

## Why this exists

Spikeforge trains and exports spiking neural networks in Python, and can
export a trained topology to NIR. NIR is a portable graph format, but
nothing in the spikeforge ecosystem runs that graph without Python and
PyTorch. This project is the missing piece: a small, dependency-light C
implementation that loads a NIR-exported graph and its quantized weights,
and runs inference — the kind of thing a government or embedded-systems
buyer actually wants to see (C, not Python, on the target hardware), per
[MISRA-C](https://www.misra.org.uk/)/DO-178C-adjacent embedded conventions.

This is deliberately a separate repository from any project-specific
research (dust-robust lunar vision, etc.) — this runtime is generic to any
NIR-exported spikeforge model, not tied to one use case.

## Status

Not yet started. Scaffolding only.

## Scope (planned)

- Parse a NIR JSON graph export (LIF/current-based LIF stages, linear/conv
  layers, the node kinds spikeforge actually emits).
- Load int8/uint8-quantized weights (matching spikeforge_targets' declared
  quantization schemes).
- Run a fixed-timestep forward pass with no dynamic memory allocation after
  initialization, no floating-point library dependencies beyond libc, and
  no OS assumptions beyond a C11 freestanding-compatible subset where
  feasible.
- Numerically validate against spikeforge's own reference NIR interpreter
  (Python) on a shared test-vector set, not just "it runs."

## Out of scope (for now)

- Training. This is inference-only.
- Any Curlee/JOE integration — tracked separately as a possible future
  target once [curlee#305](https://github.com/w4ffl35/curlee/issues/305)
  and [curlee#306](https://github.com/w4ffl35/curlee/issues/306) land.
- A specific hardware target. This targets portable C first; a specific
  MCU/board port is a later, separate scope once the core runtime exists.

## License

BSD-3-Clause (matching spikeforge's own license) — see [LICENSE](LICENSE).
