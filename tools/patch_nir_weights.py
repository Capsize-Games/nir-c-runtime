"""Patch a graph_summary JSON with weights from a C checkpoint.

Stdlib-only reverse-bridge helper (#30): reads a `graph_summary`
document (spikeforge.nir_bridge.exporter.graph_summary) and a
`spikeforge-c-checkpoint` file, and writes a summary whose weighted
nodes carry the checkpoint's flat row-major weights re-nested to the
node's shape. The output feeds the same consumers as any summary --
notably a fresh `nir_graph_load`, or a NIR graph rebuild on the
Python side.

Name, kind, and shape mismatches fail loudly; nothing is silently
reinterpreted. Nodes the checkpoint does not mention keep their
summary weights.

Usage:
    patch_nir_weights.py summary.json checkpoint.json -o out.json
"""

import argparse
import json
import sys
from typing import Any


def nest(flat: list, shape: list) -> Any:
    """Re-nest a flat row-major list to `shape`."""
    if len(shape) == 1:
        if len(flat) != shape[0]:
            raise ValueError(f"weight has {len(flat)} values, not {shape}")
        return list(flat)
    stride = 1
    for dim in shape[1:]:
        stride *= dim
    if len(flat) != shape[0] * stride:
        raise ValueError(f"weight has {len(flat)} values, not {shape}")
    return [
        nest(flat[i * stride:(i + 1) * stride], shape[1:])
        for i in range(shape[0])
    ]


def patch(summary: dict, checkpoint: dict) -> dict:
    """Return a copy of `summary` with checkpoint weights applied."""
    if checkpoint.get("format") != "spikeforge-c-checkpoint":
        raise ValueError("not a spikeforge-c-checkpoint document")
    if checkpoint.get("version") != 1:
        raise ValueError(
            f"unsupported checkpoint version {checkpoint.get('version')!r}"
        )
    nodes = {node["name"]: node for node in summary["nodes"]}
    patched = 0
    for entry in checkpoint["nodes"]:
        name = entry["name"]
        if name not in nodes:
            raise ValueError(f"checkpoint names unknown node {name!r}")
        node = nodes[name]
        if node["kind"] != entry["kind"]:
            raise ValueError(
                f"node {name!r}: kind {entry['kind']!r} != "
                f"{node['kind']!r}"
            )
        node["params"]["weight"] = nest(entry["weight"], entry["shape"])
        bias = entry.get("bias")
        if bias is None:
            if node["params"].get("bias") is not None and node["kind"] in (
                "Affine",
            ):
                raise ValueError(f"node {name!r}: checkpoint drops a bias")
            node["params"]["bias"] = None
        else:
            node["params"]["bias"] = list(bias)
        patched += 1
    if patched == 0:
        raise ValueError("checkpoint carries no nodes")
    return summary


def main(argv: list) -> int:
    """Patch a summary file with a checkpoint file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="graph_summary JSON file")
    parser.add_argument("checkpoint", help="checkpoint JSON file")
    parser.add_argument("-o", "--output", required=True,
                        help="patched summary output path")
    args = parser.parse_args(argv)
    try:
        with open(args.summary, encoding="utf-8") as handle:
            summary = json.load(handle)
        with open(args.checkpoint, encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        result = patch(summary, checkpoint)
    except (OSError, ValueError, KeyError) as error:
        print(f"patch_nir_weights: {error}", file=sys.stderr)
        return 1
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
