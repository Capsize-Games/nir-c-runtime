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
inference for `Input`, `Conv2d`, `Conv1d`, `Affine`, `Linear`, `Flatten`,
`AvgPool2d`, `SumPool2d`, `LI`, `LIF`, `IF`, `CubaLIF`, `Threshold`, `Scale`,
`Delay`, `ReLU`, and `Output`. Any other kind fails graph load with a named
"unsupported node kind" error rather than being silently misinterpreted.

**Validated, not just "it runs":** `tests/test_conv_net.c` and
`tests/test_neuron_kinds.c` check small topologies against spikeforge's own
reference `NirInterpreter` output (`tools/export_fixture.py` produces both
from one run) — **bit-exact matches** with the Python reference
interpreter (max abs diff `0.0` on every committed fixture, through both
the allocating and the workspace runtimes). The `conv_net` topology was
also checked by hand against a real trained 128×128, 2-channel, 11-class
checkpoint (not committed here) with the same bit-exact result. Every test
executable also passes clean under AddressSanitizer + UBSan.

LIF graphs default to NIR's hard reset: a spike stores `v_reset`. The graph
loader also accepts the explicit runtime extension `reset_mode: "subtractive"`,
which stores `membrane - v_threshold` immediately after a spike. BPTT uses the
surrogate derivative through both the spike and this subtractive reset; hard
reset keeps its existing zero-gradient reset branch. Graph-summary export
preserves the subtractive mode.

**Known limitation:** convolution uses portable 4×4 output tiles but is not
yet SIMD-vectorized. On the 128×128 real-model check above it took ~148ms for
16 timesteps versus PyTorch's ~20ms on the same machine — expected for a
correctness-first MVP, not yet tuned, and the real target hardware for this
runtime (an MCU with no PyTorch alternative at all) makes a different
comparison than "faster than a desktop GPU build" anyway. SIMD and
target-specific kernels remain future optimization work.

## Scope

- [x] Parse a NIR JSON graph export for the node kinds above.
- [x] Load full-precision (float32) weights from the export.
- [x] Run a fixed-timestep forward pass, validated against spikeforge's own
      reference NIR interpreter.
- [x] Load int8/uint8-quantized weights (matching spikeforge_targets'
      declared quantization schemes); values are dequantized at graph load.
- [ ] No dynamic allocation after initialization / freestanding-C11
      compatibility for a real MCU target — the current implementation
      allocates a fresh tensor per node per step via malloc/calloc, which
      is fine for validation but not yet appropriate for a hard-realtime
      or freestanding embedded target.
- [x] Caller-owned workspace execution for fixed-shape graph inference via
      `nir_runtime_run_workspace`; the legacy convenience API above remains
      allocation-based.
- [x] Blocked conv2d with portable 4×4 output tiles (see the timing note
      above); SIMD kernels remain target-specific future work.
- [x] Allocation-free fixed-point event-module ABI for bounded, single-output
      temporal detectors (`fixed_event.h`), deterministic guard-state/action
      helpers, and an 8,192-row exhaustive C/Curlee truth-table fingerprint.
- [x] C-side training math (`train.h`): named surrogate gradients
      (fast_sigmoid/atan, no silent default) with Threshold/LI/Affine/Linear
      backward ops, softmax cross-entropy, SGD/Adam optimizers over raw
      weight buffers, and a local STDP rule with a graph-bound adapter
      (`stdp_graph.h`) that trains a loaded Affine/Linear node's weights
      from recorded pre/post spikes.
- [x] C-side training IO: seeded RNG (`rng.h`), rate/latency/delta
      spike encoders (`encode.h`), IDX MNIST dataset loading (`data.h`),
      and versioned checkpoint save/load with weight accessors and
      summary export (`checkpoint.h`), all covered by C tests.
- [x] Backward ops for spatial/linear nodes (Conv2d, Conv1d, Flatten,
      AvgPool2d, SumPool2d, Scale) and spiking neurons (LIF/IF/CubaLIF
      with surrogate through the spike output, zero through the
      hard-reset branch), plus ReLU with zero derivative at and below zero.
- [x] Graph-level BPTT tape (`tape.h`): retains per-step activations
      and neuron state, backpropagates through Delay/LI recurrence,
      matches `nir_runtime_run` bit for bit, and finite-difference
      verifies smooth topologies including dense layers separated by ReLU.
- [x] Mini-batch training (`minibatch.h`): mean loss/gradient
      reduction over a batch-1 loop (no batch axis on the tensor
      ABI) plus seeded Fisher-Yates epoch shuffling.
- [x] Optimizer-state checkpoints: v2 `optimizers` section persists
      SGD velocity and Adam m/v/step per node for bit-identical
      resume; v1 files still load.
- [x] Weight initializers (`init.h`: Xavier/He/uniform from
      `nir_rng_t`), LR schedules (`schedule.h`: step/cosine/
      exponential mirroring torch), and extra losses (`loss.h`:
      MSE plus spike-count cross-entropy).
- [x] End-to-end C training driver (`driver.h`) plus a worked MNIST
      example (`examples/mnist_train`): loads IDX, encodes, runs
      the tape, steps SGD/Adam per batch, saves optimizer
      checkpoints — zero Python at train time (see Training).

## Training

The driver wires the tested units into one loop; no Python or
torch is involved after the fixtures are committed:

```text
IDX MNIST (data.h) → encode (encode.h: repeat/rate)
  → tape forward (tape.h, unprimed per sample)
  → cross-entropy (train.h) → tape backward
  → minibatch mean (minibatch.h) → SGD/Adam step
  → checkpoint with optimizer state (checkpoint.h)
```

Build and run the worked example (tiny 2×2, 3-sample fixture,
4-step LI network, 20 Adam epochs):

```sh
cmake -S . -B build
cmake --build build --parallel
./build/examples/mnist_train \
  examples/mnist_tiny_graph.json \
  tests/fixtures/mnist_tiny_images.idx \
  tests/fixtures/mnist_tiny_labels.idx \
  /tmp/mnist_tiny_checkpoint.json 20
```

Expected output (deterministic from seed 7): initial loss
~2.39 at 0.000 accuracy, falling to ~1.16 at 0.667 accuracy
by epoch 20, saving a v2 checkpoint with `step 40` (2 batches
× 20 epochs). `tests/test_driver.c` asserts the same run beats
its initial loss by 0.5 and resumes bit-identically from the
saved checkpoint.

Zero-Python proof: the example is a pure C binary — CI runs it
via ctest and again with `PATH=/nonexistent` (no `python` or
`python3` resolvable), and it still trains and saves. The only
Python in the repo (`tools/`) regenerates fixtures; training
and inference never invoke it.

## Out of scope (for now)

- General Curlee/JOE graph execution. The narrow, allocation-free event ABI is
  integrated as the reference path for
  [curlee#316](https://github.com/w4ffl35/curlee/issues/316); loading arbitrary
  NIR graphs into JOE remains future work.
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
