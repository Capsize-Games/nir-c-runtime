"""Export small NIR graphs + golden input/output pairs.

Dev-only tool: requires spikeforge on the Python path (run with an
environment with ``spikeforge[nir]`` installed, e.g.
``PYTHONPATH=<spikeforge-checkout> <venv>/bin/python
tools/export_fixture.py``). Not part of the C runtime itself -- this
only produces the test fixtures under tests/fixtures/ that the C
tests validate against.

Each fixture pairs a graph export with the reference
``NirInterpreter`` output on the same input -- the source of truth
for NIR node semantics this runtime implements. Small on purpose: the
fixtures are committed to the repo and read by humans debugging a
mismatch, not real accuracy benchmarks.

Fixtures:
- ``conv_net``: the original conv2d/LI/Threshold topology.
- ``lif_linear``: Linear (bias-free) + LIF chain from a hand-built
  spec through the real exporter (``to_nir``/``graph_summary``).
- ``cuba_conv``: Conv1d + CubaLIF + Linear + LIF through the same
  real exporter path.
- ``if_net``: a hand-built ``nir.IF`` graph (no topology preset emits
  IF today) run through the reference interpreter.
"""

import json
from pathlib import Path

import numpy as np
import torch

from spikeforge.nir_bridge.exporter import graph_summary
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology.builder import build_module
from spikeforge.topology.registry import build_topology
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SHAPE = (8, 8)
IN_CHANNELS = 2
NUM_CLASSES = 4
NUM_STEPS = 5
SEED = 0


def main() -> int:
    """Export every fixture pair into tests/fixtures/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_conv_net()
    export_lif_linear()
    export_cuba_conv()
    export_if_net()
    print(f"wrote fixtures to {OUT_DIR}")
    return 0


def export_conv_net() -> None:
    """Build a small conv_net, export its graph, and a golden run."""
    torch.manual_seed(SEED)
    spec, module = build_topology(
        "conv_net",
        {
            "in_channels": IN_CHANNELS, "channels": 2,
            "input_size": SHAPE, "num_classes": NUM_CLASSES,
        },
    )
    _randomize_weights(module, SEED)

    graph = to_nir(spec, module)
    summary = graph_summary(graph)

    generator = torch.Generator().manual_seed(SEED)
    spikes = (torch.rand(
        NUM_STEPS, 1, IN_CHANNELS, *SHAPE, generator=generator,
    ) < 0.3).float()

    result = NirInterpreter(graph).run(spikes)

    _write_json(OUT_DIR / "conv_net_graph.json", summary)
    _write_json(
        OUT_DIR / "conv_net_io.json",
        {
            "shape": [NUM_STEPS, IN_CHANNELS, *SHAPE],
            "input": spikes.squeeze(1).tolist(),
            "readout": result.readout.squeeze(0).tolist(),
        },
    )


def export_lif_linear() -> None:
    """Export a Linear/LIF chain with nonzero spike activity."""
    seed = 1
    torch.manual_seed(seed)
    spec = chain([
        Stage("fc1", "linear", {
            "in_features": 4, "out_features": 3, "bias": False,
        }),
        Stage("lif1", "leaky", {
            "beta": 0.9, "reset": "zero", "threshold": 1.0,
        }),
        Stage("fc2", "linear", {
            "in_features": 3, "out_features": 2,
        }),
        Stage("out", "leaky", {
            "beta": 0.9, "reset": "zero", "threshold": 1.0,
        }),
    ])
    spec.validate()
    module = build_module(spec)
    _randomize_weights(module, seed, scale=0.4)
    _export_spec_run(
        spec, module, seed, steps=6, frame_shape=(4,),
        spike_probability=0.5, stem="lif_linear",
    )


def export_cuba_conv() -> None:
    """Export a Conv1d/CubaLIF/Linear/LIF chain."""
    seed = 2
    torch.manual_seed(seed)
    spec = chain([
        Stage("conv1", "conv1d", {
            "in_channels": 2, "out_channels": 2, "kernel_size": 3,
            "padding": 1,
        }),
        Stage("lif1", "synaptic", {
            "beta": 0.9, "alpha": 0.8, "threshold": 1.0,
        }),
        Stage("flatten", "flatten", {}),
        Stage("fc", "linear", {
            "in_features": 16, "out_features": 2, "bias": False,
        }),
        Stage("out", "leaky", {
            "beta": 0.9, "reset": "zero", "threshold": 1.0,
        }),
    ])
    spec.validate()
    module = build_module(spec)
    _randomize_weights(module, seed, scale=0.2)
    _export_spec_run(
        spec, module, seed, steps=6, frame_shape=(2, 8),
        spike_probability=0.5, stem="cuba_conv",
    )


def export_if_net() -> None:
    """Export a hand-built IF graph with a crafted input train."""
    import nir

    steps = [
        [0.6, 1.5, 0.0],
        [0.6, 0.2, 0.4],
        [0.1, 0.9, 0.7],
        [0.0, 0.0, 0.0],
    ]
    graph = nir.NIRGraph(
        {
            "input": nir.Input({"input": None}),
            "if1": nir.IF(
                r=np.float32(1.0),
                v_threshold=np.float32(1.0),
                v_reset=np.float32(0.0),
            ),
            "output": nir.Output({"output": None}),
        },
        [("input", "if1"), ("if1", "output")],
        type_check=False,
    )
    frames = torch.tensor(steps, dtype=torch.float32).unsqueeze(1)
    result = NirInterpreter(graph).run(frames)
    _write_json(OUT_DIR / "if_net_graph.json", graph_summary(graph))
    _write_json(
        OUT_DIR / "if_net_io.json",
        {
            "shape": [len(steps), 3],
            "input": steps,
            "readout": result.readout.squeeze(0).tolist(),
        },
    )
    _require_activity("if_net", result.readout)


def _export_spec_run(
    spec, module, seed: int, steps: int,
    frame_shape: tuple, spike_probability: float, stem: str,
) -> None:
    """Export one spec-built graph and its golden run."""
    graph = to_nir(spec, module)
    summary = graph_summary(graph)
    generator = torch.Generator().manual_seed(seed)
    spikes = (torch.rand(
        steps, 1, *frame_shape, generator=generator,
    ) < spike_probability).float()
    result = NirInterpreter(graph).run(spikes)
    _require_activity(stem, result.readout)
    _write_json(OUT_DIR / f"{stem}_graph.json", summary)
    _write_json(
        OUT_DIR / f"{stem}_io.json",
        {
            "shape": [steps, *frame_shape],
            "input": spikes.squeeze(1).tolist(),
            "readout": result.readout.squeeze(0).tolist(),
        },
    )


def _require_activity(stem: str, readout: torch.Tensor) -> None:
    """Fail loudly when a fixture readout carries no spikes."""
    values = readout.detach().reshape(-1).tolist()
    if not any(value != 0.0 for value in values):
        raise SystemExit(f"fixture {stem} has an all-zero readout")
    print(f"{stem} readout: {values}")


def _randomize_weights(
    module: torch.nn.Module, seed: int, scale: float = 0.15
) -> None:
    """Replace every parameter with small seeded positive-leaning values.

    build_topology's default init is fine for training but not
    distinctive enough to catch a transposed-weight bug in the C
    runtime. Centered random weights (mean 0) mostly cancel out in a
    sum of ~18 terms and rarely cross the neurons' spike threshold,
    which makes a trivial all-zero readout too easy to match by
    accident on both sides; a positive-leaning range reliably drives
    real spikes so the fixture actually discriminates a broken port.
    The smaller neuron fixtures sum fewer terms per neuron, so they
    use a larger ``scale`` to reach the same spiking regime.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.rand(param.shape, generator=generator) * scale)


def _write_json(path: Path, data: object) -> None:
    """Write ``data`` as pretty-printed JSON."""
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
