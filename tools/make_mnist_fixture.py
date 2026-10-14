"""Write the tiny committed IDX fixtures for the C data test.

Stdlib-only (no torch, no datasets download): the pixels and labels
are hardcoded so regeneration is byte-stable. The C test
(tests/test_data.c) reads these with zero Python involvement.

Layout: 3 images of 2x2 pixels plus 3 labels.
"""

import struct
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

IMAGES = [
    [0, 85, 170, 255],
    [255, 170, 85, 0],
    [1, 2, 3, 4],
]
LABELS = [7, 3, 9]


def write_idx(path: Path, magic: int, dims: list, payload: bytes) -> None:
    """Write one IDX file with a big-endian header."""
    header = struct.pack(">" + "I" * (1 + len(dims)), magic, *dims)
    path.write_bytes(header + payload)


def main() -> int:
    """Write the image and label fixtures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat = bytes(px for image in IMAGES for px in image)
    write_idx(
        OUT_DIR / "mnist_tiny_images.idx", 2051,
        [len(IMAGES), 2, 2], flat,
    )
    write_idx(
        OUT_DIR / "mnist_tiny_labels.idx", 2049,
        [len(LABELS)], bytes(LABELS),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
