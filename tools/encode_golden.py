"""Emit golden spike trains for the C encoder test.

Stdlib-only: needs no torch, so it runs anywhere. It re-implements
the documented coding rules in float32 (via struct) on top of a pure
Python splitmix64, and writes tests/fixtures/encode_golden.json,
which tests/test_encode.c compares against bit-exactly.

Rule sources (see include/nir_c_runtime/encode.h):
- rate: spike iff u < clamp01(gain*x), one splitmix64-derived f32
  draw per (t, i), t-major -- the spikegen.rate/rate_conv Bernoulli
  rule. torch's own stream is MT19937, so these goldens pin the
  documented rule, not torch's stream (the equivalence contract the
  C header states).
- latency: spikegen.latency first-spike times, linear and log
  branches, rounded half-to-even like torch.round.
- delta: spikegen.delta over linspace(0, 1, steps) * pixel.
- random: rate_conv(rand * scale) with two draws per element.

The script refuses to emit goldens whose latency spike times sit
within 1e-4 of a .5 rounding boundary, so C/Python libm
differences cannot flip a bit.
"""

import json
import math
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
MASK = (1 << 64) - 1


def f32(value: float) -> float:
    """Round a float to float32 precision."""
    return struct.unpack("f", struct.pack("f", value))[0]


class SplitMix64:
    """Pure-Python splitmix64 matching src/rng.c."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK

    def next_u64(self) -> int:
        """Return the next 64 bits of the stream."""
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
        return value ^ (value >> 31)

    def next_f32(self) -> float:
        """Return a uniform float32 in [0, 1)."""
        bits = (self.next_u64() >> 40) & 0xFFFFFF
        return f32(bits * (1.0 / 16777216.0))


def clamp01(value: float) -> float:
    """Clamp a float32 value to [0, 1]."""
    return min(1.0, max(0.0, value))


def golden_rate(x, steps, gain, seed):
    """Bernoulli rate code bits, t-major."""
    rng = SplitMix64(seed)
    n = len(x)
    out = [0] * (steps * n)
    for time in range(steps):
        for i, pixel in enumerate(x):
            prob = clamp01(f32(gain * pixel))
            out[time * n + i] = 1 if rng.next_f32() < prob else 0
    return out


def check_margin(times, label):
    """Reject spike times too close to a rounding boundary."""
    for time in times:
        frac = abs(time - math.floor(time + 0.5))
        assert abs(frac - 0.5) > 1e-4, f"{label}: {time} near boundary"


def golden_latency_linear(x, steps, tau, thr, clip, norm, first=0):
    """Linear first-spike-time code bits, t-major."""
    plain = tau if not norm else float(steps - 1 - first)
    cap = f32(-plain * f32(thr - 1.0))
    times = []
    for pixel in x:
        time = f32(-plain * f32(pixel - 1.0))
        times.append(f32(min(time, cap) + first))
    check_margin(times, "latency-linear")
    return _scatter(times, x, steps, thr, clip)


def golden_latency_log(x, steps, tau, thr, norm, eps=1e-7):
    """Logarithmic first-spike-time code bits, t-major."""
    floor = thr + eps
    times = []
    for pixel in x:
        sat = max(pixel, floor)
        times.append(f32(tau * math.log(sat / (sat - thr))))
    if norm:
        peak = max(times)
        span = float(steps - 1)
        times = [f32(t * span / peak) for t in times] if peak > 0 else times
    check_margin(times, "latency-log")
    return _scatter(times, x, steps, thr, False)


def _scatter(times, x, steps, thr, clip):
    """Place one spike per pixel at its rounded spike time."""
    n = len(x)
    assert max(round(t) for t in times) <= steps - 1
    out = [0] * (steps * n)
    for i, time in enumerate(times):
        if clip and x[i] < thr:
            continue
        step = round(time)
        if step <= steps - 1:
            out[step * n + i] = 1
    return out


def golden_delta(x, steps, thr):
    """Delta code bits over a 0..1 ramp, t-major."""
    n = len(x)
    out = [0] * (steps * n)
    for time in range(steps):
        ramp = f32(time / (steps - 1)) if steps > 1 else 0.0
        prev = f32((time - 1) / (steps - 1)) if time > 0 else 0.0
        for i, pixel in enumerate(x):
            change = f32(f32(ramp * pixel) - f32(prev * pixel))
            assert abs(change - thr) > 1e-6, "delta near boundary"
            out[time * n + i] = 1 if change >= thr else 0
    return out


def golden_random(n, steps, scale, seed):
    """Noise-baseline bits, t-major."""
    rng = SplitMix64(seed)
    out = [0] * (steps * n)
    for time in range(steps):
        for i in range(n):
            magnitude = rng.next_f32()
            prob = clamp01(f32(scale * magnitude))
            out[time * n + i] = 1 if rng.next_f32() < prob else 0
    return out


def main() -> int:
    """Write the golden JSON fixture."""
    goldens = {
        "rate": {
            "x": [0.0, 0.25, 0.5, 0.75, 1.0],
            "steps": 8,
            "gain": 1.0,
            "seed": 12345,
            "bits": golden_rate(
                [0.0, 0.25, 0.5, 0.75, 1.0], 8, 1.0, 12345
            ),
        },
        "latency_linear": {
            "x": [0.0, 0.1, 0.55, 0.9, 1.0],
            "steps": 10,
            "tau": 5.0,
            "threshold": 0.01,
            "clip": False,
            "normalize": True,
            "linear": True,
            "bits": golden_latency_linear(
                [0.0, 0.1, 0.55, 0.9, 1.0], 10, 5.0, 0.01,
                False, True,
            ),
        },
        "latency_linear_clip": {
            "x": [0.0, 0.005, 0.55, 1.0],
            "steps": 10,
            "tau": 5.0,
            "threshold": 0.01,
            "clip": True,
            "normalize": True,
            "linear": True,
            "bits": golden_latency_linear(
                [0.0, 0.005, 0.55, 1.0], 10, 5.0, 0.01, True, True,
            ),
        },
        "latency_linear_plain": {
            "x": [0.2, 0.6, 1.0],
            "steps": 12,
            "tau": 2.0,
            "threshold": 0.01,
            "clip": False,
            "normalize": False,
            "linear": True,
            "first_spike_time": 1,
            "bits": golden_latency_linear(
                [0.2, 0.6, 1.0], 12, 2.0, 0.01, False, False, first=1,
            ),
        },
        "latency_log": {
            "x": [0.05, 0.3, 0.7, 1.0],
            "steps": 10,
            "tau": 5.0,
            "threshold": 0.01,
            "normalize": True,
            "linear": False,
            "bits": golden_latency_log(
                [0.05, 0.3, 0.7, 1.0], 10, 5.0, 0.01, True,
            ),
        },
        "delta": {
            "x": [0.0, 0.3, 1.0],
            "steps": 6,
            "threshold": 0.04,
            "bits": golden_delta([0.0, 0.3, 1.0], 6, 0.04),
        },
        "random": {
            "n": 4,
            "steps": 6,
            "scale": 0.5,
            "seed": 999,
            "bits": golden_random(4, 6, 0.5, 999),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "encode_golden.json", "w", encoding="utf-8") as handle:
        json.dump(goldens, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
