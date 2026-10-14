/* Weight initializers for from-scratch C training (#45).
 *
 * Every weight in a C-loaded graph otherwise comes from a
 * Python-exported graph JSON; these fill a fresh buffer so a
 * zero-Python training run can start somewhere sane, then land
 * the buffer on a node via `nir_graph_set_weights`.
 *
 * Formulas mirror torch.nn.init exactly:
 *
 *   uniform:        U(low, high)
 *   xavier_uniform: U(-b, b), b = gain * sqrt(6 / (fan_in + fan_out))
 *   xavier_normal:  N(0, s),  s = gain * sqrt(2 / (fan_in + fan_out))
 *   he_uniform:     U(-b, b), b = gain * sqrt(3 / fan_in)
 *   he_normal:      N(0, s),  s = gain / sqrt(fan_in)
 *
 * Fan conventions (torch `_calculate_fan_in_and_fan_out`, which
 * `nir_init_fans` implements): a Linear/Affine weight laid out
 * [out_features, in_features] has fan_in = in_features and
 * fan_out = out_features; a convolution weight laid out
 * [out_channels, in_channels, *kernel] multiplies both by the
 * kernel's receptive-field size (kh * kw for Conv2d, k for
 * Conv1d). He follows torch's kaiming defaults -- mode 'fan_in'
 * (only fan_in enters) with an explicit gain, so ReLU nets pass
 * sqrt(2) (torch `calculate_gain('relu')`, equal to leaky_relu
 * with a = 0) and Xavier nets pass 1. Use `nir_init_fans` on the
 * node's weight shape rather than hand-counting.
 *
 * All draws come from `nir_rng_t`, so a seeded stream replays
 * bit-identically. Uniform draws consume one u64 each; normal
 * draws use Box-Muller pairs (two u64 per pair, the spare
 * discarded when `count` is odd), so the stream position after
 * a call depends on `count` -- reseed to replay a prefix.
 */
#ifndef NIR_C_RUNTIME_INIT_H
#define NIR_C_RUNTIME_INIT_H

#include <stddef.h>

#include "nir_c_runtime/rng.h"

/* Torch fan-in/fan-out for a weight shape: shape[0] is output
 * maps, shape[1] input maps, shape[2..] the kernel. Needs
 * ndim >= 2, no zero extents, and products that fit size_t. */
int nir_init_fans(
    const size_t *shape, size_t ndim, size_t *fan_in,
    size_t *fan_out
);

/* Fill `out` with U(low, high). Needs low < high, both finite. */
int nir_init_uniform(
    nir_rng_t *rng, float *out, size_t count, float low,
    float high
);

/* Xavier/Glorot initializers. Needs both fans nonzero and a
 * finite gain >= 0 (gain 0 yields all zeros). */
int nir_init_xavier_uniform(
    nir_rng_t *rng, float *out, size_t count, size_t fan_in,
    size_t fan_out, float gain
);
int nir_init_xavier_normal(
    nir_rng_t *rng, float *out, size_t count, size_t fan_in,
    size_t fan_out, float gain
);

/* He initializers (torch kaiming_* with mode 'fan_in'). Needs
 * fan_in nonzero and a finite gain >= 0. */
int nir_init_he_uniform(
    nir_rng_t *rng, float *out, size_t count, size_t fan_in,
    float gain
);
int nir_init_he_normal(
    nir_rng_t *rng, float *out, size_t count, size_t fan_in,
    float gain
);

#endif /* NIR_C_RUNTIME_INIT_H */
