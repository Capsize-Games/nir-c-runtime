/* Spike encoders ported from spikeforge (#28).
 *
 * Mirrors `SpikeEncoder` (spikeforge/encoding/spike_encoder.py) on top
 * of snnTorch's `spikegen` semantics:
 *
 * - rate: clamp01(gain * x) Bernoulli sampling, i.e. `spikegen.rate`
 *   with gain=1 (`rate_conv`: clamp then `torch.bernoulli`). A spike
 *   fires iff a uniform draw u satisfies u < p. The draws come from
 *   `nir_rng_t` (splitmix64), NOT torch's MT19937, so bit equality
 *   with torch holds only given the same uniform stream; the rule
 *   above is the documented equivalence contract (draw k = t * n +
 *   i, consumed t-major, i-minor, one `nir_rng_next_f32` per draw).
 * - latency: first-spike-time code, `spikegen.latency` with the
 *   encoder's tau/threshold/clip/normalize/linear flags. Linear and
 *   log branches both implemented; spike times round half-to-even
 *   (`rintf`), matching `torch.round`.
 * - delta: `spikegen.delta` over a linspace(0, 1, steps) ramp times
 *   each pixel (padding=False, off_spike=False), threshold given as
 *   a fraction (`SpikeEncoder` divides its percent by 100).
 * - random: the noise baseline, `rate_conv(rand * scale)` with two
 *   RNG draws per element (magnitude, then threshold).
 * - repeat: the legacy raw path (`_repeat_pixels`): normalised
 *   pixels repeated across steps, no spikes.
 *
 * Outputs are time-major: out[t * n + i] is pixel i at step t,
 * matching the encoder's [T, B=1, N] layout flattened per sample.
 */
#ifndef NIR_C_RUNTIME_ENCODE_H
#define NIR_C_RUNTIME_ENCODE_H

#include <stddef.h>
#include <stdint.h>

#include "nir_c_runtime/rng.h"

#define NIR_LATENCY_EPSILON 1e-7f

typedef struct {
    size_t num_steps;
    float tau;
    float threshold;
    int clip;
    int normalize;
    int linear;
    size_t first_spike_time;
    float epsilon;
} nir_latency_params_t;

/* Bernoulli rate code of `x[0..n)` over `num_steps`. Consumes one RNG
 * draw per (t, i). Returns 0 and sets *error on bad sizes. */
int nir_encode_rate(
    const float *x, size_t n, size_t num_steps, float gain,
    nir_rng_t *rng, uint8_t *out, char **error
);

/* First-spike-time latency code of `x[0..n)`. Every input must lie
 * in [0, 1] (the torch side raises otherwise). Returns 0 and sets
 * *error on out-of-range input, bad parameters, or a spike time
 * that overflows `num_steps` (mirroring spikegen's no-bypass
 * raise). */
int nir_encode_latency(
    const float *x, size_t n, const nir_latency_params_t *params,
    uint8_t *out, char **error
);

/* Delta code of `x[0..n)` over `num_steps` with a fractional
 * change threshold. Returns 0 and sets *error on bad sizes. */
int nir_encode_delta(
    const float *x, size_t n, size_t num_steps, float threshold,
    uint8_t *out, char **error
);

/* Noise baseline: rate-coded uniform noise, input ignored.
 * Returns 0 and sets *error on bad sizes. */
int nir_encode_random(
    size_t n, size_t num_steps, float scale, nir_rng_t *rng,
    uint8_t *out, char **error
);

/* Repeat `x[0..n)` across `num_steps` into `out[t * n + i]`.
 * Returns 0 and sets *error on bad sizes. */
int nir_encode_repeat(
    const float *x, size_t n, size_t num_steps, float *out,
    char **error
);

#endif /* NIR_C_RUNTIME_ENCODE_H */
