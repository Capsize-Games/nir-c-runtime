/* C-side training math: surrogate-name resolution, cross-entropy
 * loss, SGD/Adam optimizers over raw weight buffers, and a local
 * STDP rule ported from spikeforge's STDPSynapse.
 *
 * This mirrors the torch training step (cross_entropy on logits,
 * then a torch.optim.Adam step) for single-step topologies whose
 * backward pass the caller composes from the layer backward ops in
 * ops.h. A graph-level tape retaining per-step activations for
 * backprop-through-time is future work; optimizers here operate on
 * caller-provided float buffers (e.g. a node's weight array).
 */
#ifndef NIR_C_RUNTIME_TRAIN_H
#define NIR_C_RUNTIME_TRAIN_H

#include <stddef.h>

#include "nir_c_runtime/ops.h"
#include "nir_c_runtime/tensor.h"

/* Resolve "fast_sigmoid" or "atan" to a surrogate kind. Any other
 * name (or NULL) fails with a malloc'd error naming it, mirroring
 * spikeforge's resolve_surrogate raising KeyError -- there is no
 * silent default surrogate. */
int nir_surrogate_from_name(
    const char *name, nir_surrogate_kind_t *out, char **error
);

/* Softmax cross-entropy for one logit vector, matching
 * torch.nn.functional.cross_entropy on a single sample: loss =
 * -log(softmax(logits)[target]), grad = softmax(logits) with
 * grad[target] -= 1. Batch reduction is the caller's job. */
int nir_cross_entropy(
    const nir_tensor_t *logits, size_t target, float *loss,
    nir_tensor_t *grad
);

/* SGD with optional momentum and L2 weight decay, matching
 * torch.optim.SGD with dampening=0 and nesterov=False:
 *   d = grad + weight_decay * w
 *   v = momentum * v + d
 *   w -= lr * v
 */
typedef struct {
    float lr;
    float momentum;
    float weight_decay;
    float *velocity;
    size_t count;
} nir_sgd_t;

/* lr must be finite and >= 0, momentum and weight_decay finite
 * and >= 0, count nonzero. The velocity buffer starts at zero. */
int nir_sgd_init(
    nir_sgd_t *opt, size_t count, float lr, float momentum,
    float weight_decay, char **error
);
int nir_sgd_step(
    nir_sgd_t *opt, float *weights, const float *grad, size_t count
);
void nir_sgd_free(nir_sgd_t *opt);

/* Adam with bias correction, matching torch.optim.Adam with
 * amsgrad=False:
 *   d = grad + weight_decay * w
 *   m = beta1 * m + (1 - beta1) * d
 *   v = beta2 * v + (1 - beta2) * d^2
 *   w -= lr * (m / (1 - beta1^t)) /
 *       (sqrt(v / (1 - beta2^t)) + eps)
 * where t counts steps from 1.
 */
typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float *m;
    float *v;
    size_t count;
    unsigned long step;
} nir_adam_t;

/* lr must be finite and >= 0, betas finite in [0, 1), eps finite
 * and > 0, weight_decay finite and >= 0, count nonzero. */
int nir_adam_init(
    nir_adam_t *opt, size_t count, float lr, float beta1,
    float beta2, float eps, float weight_decay, char **error
);
int nir_adam_step(
    nir_adam_t *opt, float *weights, const float *grad, size_t count
);
void nir_adam_free(nir_adam_t *opt);

/* Pair-based STDP with pre/post eligibility traces, ported from
 * spikeforge.memory.STDPSynapse (Song/Abbott rule). Weights are
 * [in_features][out_features], row-major pre-major, matching the
 * torch [in, out] layout and the spikes @ weight forward. */
typedef struct {
    size_t in_features;
    size_t out_features;
    float tau_pre;
    float tau_post;
    float a_plus;
    float a_minus;
    float w_min;
    float w_max;
    float *weight;
    float *pre_trace;
    float *post_trace;
} nir_stdp_t;

/* taus must be finite and > 0, w_min <= w_max, both sizes nonzero.
 * Weights and traces start at zero. */
int nir_stdp_init(
    nir_stdp_t *s, size_t in_features, size_t out_features,
    float tau_pre, float tau_post, float a_plus, float a_minus,
    float w_min, float w_max, char **error
);

/* Overwrite the weight matrix (in*out floats, pre-major). */
int nir_stdp_set_weight(nir_stdp_t *s, const float *weight);

/* One STDP step: decay both traces first (a spike reads the
 * *prior* step's trace), potentiate on post spikes landing on
 * lingering pre trace, depress on pre spikes landing on lingering
 * post trace, clamp to [w_min, w_max], then add this step's
 * spikes to the traces. */
int nir_stdp_step(
    nir_stdp_t *s, const float *pre_spikes, const float *post_spikes
);
void nir_stdp_free(nir_stdp_t *s);

#endif /* NIR_C_RUNTIME_TRAIN_H */
