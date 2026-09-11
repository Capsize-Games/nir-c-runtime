"""Export a small conv_net NIR graph + a golden input/output pair.

Dev-only tool: requires spikeforge on the Python path (run with
lunar-snn's venv, e.g. ~/Projects/999/.venv/bin/python, or any
environment with `spikeforge[nir]` installed). Not part of the C
runtime itself -- this only produces the test fixtures under
tests/fixtures/ that the C tests validate against.

Uses a small, randomly-initialized topology (not any trained
checkpoint) run through spikeforge's own reference NirInterpreter --
the source of truth for NIR node semantics this runtime implements.
Small on purpose: the fixture is committed to the repo and read by
humans debugging a mismatch, not a real accuracy benchmark.
"""

import json
from pathlib import Path

import torch

from spikeforge.nir_bridge.exporter import graph_summary
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology.registry import build_topology

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SHAPE = (8, 8)
IN_CHANNELS = 2
NUM_CLASSES = 4
NUM_STEPS = 5
SEED = 0


def main() -> int:
    """Build a small conv_net, export its graph, and a golden run."""
    torch.manual_seed(SEED)
    spec, module = build_topology(
        "conv_net",
        {
            "in_channels": IN_CHANNELS, "channels": 2,
            "input_size": SHAPE, "num_classes": NUM_CLASSES,
        },
    )
    _randomize_weights(module)

    graph = to_nir(spec, module)
    summary = graph_summary(graph)

    generator = torch.Generator().manual_seed(SEED)
    spikes = (torch.rand(
        NUM_STEPS, 1, IN_CHANNELS, *SHAPE, generator=generator,
    ) < 0.3).float()

    result = NirInterpreter(graph).run(spikes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(OUT_DIR / "conv_net_graph.json", summary)
    _write_json(
        OUT_DIR / "conv_net_io.json",
        {
            "shape": [NUM_STEPS, IN_CHANNELS, *SHAPE],
            "input": spikes.squeeze(1).tolist(),
            "readout": result.readout.squeeze(0).tolist(),
        },
    )
    print(f"wrote fixtures to {OUT_DIR}")
    return 0


def _randomize_weights(module: torch.nn.Module) -> None:
    """Replace every parameter with small seeded positive-leaning values.

    build_topology's default init is fine for training but not
    distinctive enough to catch a transposed-weight bug in the C
    runtime. Centered random weights (mean 0) mostly cancel out in a
    sum of ~18 terms and rarely cross the neurons' spike threshold,
    which makes a trivial all-zero readout too easy to match by
    accident on both sides; a positive-leaning range reliably drives
    real spikes so the fixture actually discriminates a broken port.
    """
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.rand(param.shape, generator=generator) * 0.15)


def _write_json(path: Path, data: object) -> None:
    """Write ``data`` as pretty-printed JSON."""
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
