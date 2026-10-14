/* Mini-batch training over batch-1 tensors (#42).
 *
 * Reduction convention (this is the documented contract; train.h
 * and data.h point here): the caller loops over one sample at a
 * time and reduces with a mean, matching torch's default
 * `reduction='mean'`:
 *
 *   batch loss = mean(sample losses)
 *   batch grad = mean(sample grads)
 *
 * Per batch the caller zeroes one accumulator per parameter
 * buffer (`nir_minibatch_zero`), adds each sample's gradient
 * (`nir_minibatch_accumulate`), scales the sums by 1/B
 * (`nir_minibatch_scale`), and only then takes the optimizer
 * step. Losses reduce the same way through
 * `nir_minibatch_mean_loss`. The optimizers in train.h always
 * see the mean gradient -- scale exactly once per batch.
 *
 * Within one sample the single-sample losses keep their own
 * rules: cross-entropy (train.h) and spike-count loss (loss.h)
 * are per-logit-vector, MSE (loss.h) already means over its
 * elements. Those per-sample values are what the batch mean
 * averages; nothing here re-weights them.
 *
 * Batch-axis decision (#42, decided): no batch axis on
 * `nir_tensor_t`. The accumulation loop above keeps the tensor
 * ABI and every inference call site unchanged, and the extra
 * adds are exact and cheap next to a forward/backward pass.
 * Revisit only with measured evidence that the loop -- not the
 * math under it -- is the bottleneck or the bug farm.
 *
 * Epoch shuffling is Fisher-Yates over an index array
 * (`nir_sampler_arange` + `nir_sampler_shuffle`), driven by the
 * seeded `nir_rng_t` stream so an epoch order replays exactly
 * from its seed.
 */
#ifndef NIR_C_RUNTIME_MINIBATCH_H
#define NIR_C_RUNTIME_MINIBATCH_H

#include <stddef.h>

#include "nir_c_runtime/rng.h"

/* Zero an accumulator of `n` floats. `accum` must hold `n`
 * floats; `n` may be 0. */
void nir_minibatch_zero(float *accum, size_t n);

/* Add `grad` into `accum` elementwise. Both buffers must hold
 * `n` floats with `n` nonzero. Fails (return 0, `accum`
 * untouched) on NULL buffers or non-finite data. */
int nir_minibatch_accumulate(
    float *accum, const float *grad, size_t n
);

/* Scale `accum` by `scale` elementwise -- the 1/B step that
 * turns accumulated sums into the mean gradient. Fails
 * (return 0, `accum` untouched) on a NULL buffer, `n` of 0,
 * or non-finite scale or data. */
int nir_minibatch_scale(float *accum, size_t n, float scale);

/* Mean of `n` per-sample losses, accumulated in float64 and
 * rounded once. Fails (return 0) on NULL buffers, `n` of 0,
 * or a non-finite entry. */
int nir_minibatch_mean_loss(
    const float *losses, size_t n, float *mean
);

/* Fill `indices` with 0..n-1. `indices` must hold `n`
 * entries; `n` may be 0. */
void nir_sampler_arange(size_t *indices, size_t n);

/* Fisher-Yates shuffle of `indices` in place, drawing one
 * `nir_rng_next_u64` per position from `i = n-1` down to 1
 * with `j = draw % (i + 1)`. A seed therefore replays the
 * same permutation on any platform. Fails (return 0) on a
 * NULL `rng`, or a NULL `indices` with `n` nonzero; `n` of
 * 0 or 1 is a successful no-op. */
int nir_sampler_shuffle(nir_rng_t *rng, size_t *indices, size_t n);

#endif /* NIR_C_RUNTIME_MINIBATCH_H */
