/* Loss functions beyond single-sample cross-entropy (#47).
 *
 * `nir_cross_entropy` (train.h) stays the classifier loss for
 * logits; these cover the other two common SNN recipes:
 *
 *   nir_mse_loss: mean-squared error with 'mean' reduction
 *   (loss = mean((pred - target)^2), grad = 2(pred - target)/n),
 *   mirroring torch F.mse_loss on one sample. Serves
 *   readout/membrane regression -- predicting continuous
 *   targets such as a decoded intensity or a target membrane
 *   trace -- where classification-style softmax is wrong.
 *
 *   nir_spike_count_loss: softmax cross-entropy over the
 *   per-class spike-count vector, mirroring torch
 *   F.cross_entropy(counts, target) on one sample. This is the
 *   decided spike-count form (cross-entropy over counts, not
 *   MSE on counts): it matches the snnTorch recipe of summing
 *   output spikes over the window and classifying the totals,
 *   and spikeforge's own averaged-readout-plus-cross_entropy
 *   training path is the same family. The caller accumulates
 *   the T-step window into `counts` (a plain sum; each step's
 *   spikes are 0/1, so counts are non-negative by
 *   construction) and backpropagates through the accumulation
 *   itself -- every timestep's spikes share their class's
 *   count gradient -- until the BPTT tape (#39) owns that.
 *
 * Both keep the single-sample-first convention: 1-D vectors,
 * mean reduction for MSE, batch reduction via the mean
 * convention in minibatch.h.
 */
#ifndef NIR_C_RUNTIME_LOSS_H
#define NIR_C_RUNTIME_LOSS_H

#include <stddef.h>

#include "nir_c_runtime/tensor.h"

/* MSE on one sample. `pred`/`target` must be 1-D with equal
 * nonzero size and finite data; `grad` needs capacity for the
 * vector and follows the cross-entropy aliasing rule (exact
 * aliasing with either input is fine, partial overlap fails). */
int nir_mse_loss(
    const nir_tensor_t *pred, const nir_tensor_t *target,
    float *loss, nir_tensor_t *grad
);

/* Softmax cross-entropy over spike counts: loss =
 * -log(softmax(counts)[target]), grad = softmax(counts) with
 * grad[target] -= 1. Same shape/aliasing rules as
 * `nir_cross_entropy`. */
int nir_spike_count_loss(
    const nir_tensor_t *counts, size_t target, float *loss,
    nir_tensor_t *grad
);

#endif /* NIR_C_RUNTIME_LOSS_H */
