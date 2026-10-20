/* Graph-level backprop-through-time tape (#39).
 *
 * The layer backward ops in ops.h cover one step of one node; the
 * caller would have to hand-compose them, and only for single-step
 * topologies. The tape closes that gap for a whole exported graph:
 * it runs a multi-step forward pass that retains every step's
 * per-node summed inputs, values, and neuron state, then walks the
 * graph backward in reverse topological order, unrolling Delay and
 * neuron state through time and accumulating parameter gradients
 * per node.
 *
 * The forward pass reuses the runtime's Delay-breaking scheduler
 * (an internal nir_runtime_t, never stepped, supplies the exact
 * order and cycle validation) and evaluates the same public layer
 * ops, so its readout matches nir_runtime_run bit for bit. One
 * deliberate difference: every tape forward starts from unprimed
 * temporal state (no retained membranes or delay buffers across
 * forwards), since a training step pairs one fresh forward with
 * one backward.
 *
 * Backward semantics:
 *
 * - The Output node's time-averaging backpropagates as grad/T
 *   into every step's output value.
 * - Delay nodes route their step-t value gradient to the
 *   contributing sources' step-(t-1) value gradients, mirroring
 *   the forward buffer refresh.
 * - LI nodes unroll exactly: each step's membrane gradient is
 *   the consumer gradient plus the next step's previous-state
 *   gradient.
 * - Hard-reset LIF, IF, and CubaLIF use the spiking-neuron
 *   convention in ops.h: the reset branch carries zero gradient.
 *   Subtractive-reset LIF also propagates through the immediate
 *   subtractive reset using the configured surrogate. The tape
 *   wires previous-state gradients into grad_stored where shapes
 *   agree. CubaLIF's synaptic-current recurrence has no gradient
 *   input slot on nir_op_cuba_backward, so cross-step synaptic
 *   credit is truncated as well.
 * - ReLU backpropagates grad_output where its retained summed input
 *   is strictly positive and zero at and below zero.
 *
 * The tape owns all retained tensors and gradient buffers; the
 * graph (and its parameter arrays) is borrowed and must outlive
 * the tape. Parameter gradients are exposed read-only for the
 * caller to feed nir_sgd_step/nir_adam_step. Inference APIs are
 * untouched: this module only adds new symbols.
 */
#ifndef NIR_C_RUNTIME_TAPE_H
#define NIR_C_RUNTIME_TAPE_H

#include <stddef.h>

#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/ops.h"
#include "nir_c_runtime/runtime.h"
#include "nir_c_runtime/tensor.h"

typedef struct {
    const nir_graph_t *graph; /* borrowed; must outlive the tape */
    /* Scheduler reuse: supplies the Delay-breaking topological
     * order and cycle validation. Never stepped. */
    nir_runtime_t sched;
    size_t num_steps;
    nir_surrogate_kind_t surrogate;
    float slope;
    /* Retained forward pass, indexed [step * node_count + node]:
     * summed inputs, node values, post-step neuron state (LI,
     * LIF, IF, CubaLIF), and CubaLIF synaptic current. Slots a
     * node never fills stay empty (data == NULL); an unprimed
     * Delay's step-0 value is one such slot, mirroring the
     * runtime leaving it unset. */
    nir_tensor_t *inputs;
    nir_tensor_t *values;
    nir_tensor_t *stored;
    nir_tensor_t *syn;
    /* Backward buffers, same indexing: dL/dvalue per step and
     * node, plus dL/dstored-state routed from the next step for
     * LI/LIF/IF/CubaLIF nodes. */
    nir_tensor_t *grads;
    nir_tensor_t *state_grads;
    /* Accumulated parameter gradients, indexed by node:
     * grad_weight for Affine/Linear/Conv2d/Conv1d (Conv weights
     * are flat vectors in params->weight order, as documented
     * on the layer backward ops), grad_bias for biased nodes,
     * grad_scale[i] for Scale nodes. Unweighted nodes keep an
     * empty grad_weight (data == NULL); unbiased nodes an empty
     * grad_bias; non-Scale nodes a zero grad_scale. */
    nir_tensor_t *grad_weight;
    nir_tensor_t *grad_bias;
    float *grad_scale;
    int forwarded;  /* a forward pass is currently retained */
    int have_grads; /* a backward pass has populated the grads */
} nir_tape_t;

/* Bind `graph` for `num_steps`-step runs: validate the graph
 * through the shared scheduler (unschedulable graphs fail here
 * with the runtime's cycle error), allocate the retained and
 * gradient buffers, and size the parameter-gradient slots from
 * the node parameters. `graph` must outlive `tape`.
 * `num_steps` must be nonzero; the surrogate/slope pair must
 * pass nir_surrogate_grad's validation (known kind, finite
 * positive slope). Returns 0 and sets *error on failure. */
int nir_tape_init(
    nir_tape_t *tape, const nir_graph_t *graph, size_t num_steps,
    nir_surrogate_kind_t surrogate, float slope, char **error
);

/* Free every retained and gradient buffer plus the internal
 * scheduler, and zero `tape` out. Safe on an already-freed or
 * zeroed tape. */
void nir_tape_free(nir_tape_t *tape);

/* Run `frames[0..num_steps)` through the graph from unprimed
 * temporal state, retaining every step's per-node inputs,
 * values, and neuron state, and return the time-averaged
 * Output-node value in `readout_out` (caller frees it with
 * nir_tensor_free), matching nir_runtime_run on a fresh
 * runtime. Any previously retained pass (and its gradients)
 * is discarded. Returns 0 and sets *error on failure,
 * including the runtime's own input/shape errors. */
int nir_tape_forward(
    nir_tape_t *tape, const nir_tensor_t *frames,
    nir_tensor_t *readout_out, char **error
);

/* Backpropagate `grad_readout` (dL/dreadout, shaped exactly as
 * the retained readout) through the retained pass: seed each
 * step's output gradient with grad/T, walk steps newest to
 * oldest and nodes in reverse topological order, unroll Delay
 * and neuron state across steps, and accumulate the parameter
 * gradients. Requires a retained forward pass; any previous
 * gradients are zeroed first, so backward may run again after
 * a new forward (or with a new grad_readout). Returns 0 and
 * sets *error on failure. */
int nir_tape_backward(
    nir_tape_t *tape, const nir_tensor_t *grad_readout,
    char **error
);

/* Read-only parameter gradients after a successful backward.
 * The weight gradient for an Affine/Linear/Conv2d/Conv1d
 * node, or NULL for any other kind, an out-of-range index,
 * or a tape with no gradients. */
const nir_tensor_t *nir_tape_weight_grad(
    const nir_tape_t *tape, size_t node_index
);

/* The bias gradient for a biased Affine/Conv2d/Conv1d node,
 * else NULL under the same conditions as above. */
const nir_tensor_t *nir_tape_bias_grad(
    const nir_tape_t *tape, size_t node_index
);

/* The accumulated scale gradient for a Scale node
 * (sum over steps of sum(grad_output * input), per
 * nir_op_scale_backward). Writes *out and returns 1, or
 * returns 0 for any other kind, an out-of-range index, a
 * tape with no gradients, or a NULL out. */
int nir_tape_scale_grad(
    const nir_tape_t *tape, size_t node_index, float *out
);

/* dL/dframes[step]: the retained input node's value gradient
 * at that step, or NULL for an out-of-range step or a tape
 * with no gradients. */
const nir_tensor_t *nir_tape_input_grad(
    const nir_tape_t *tape, size_t step
);

#endif /* NIR_C_RUNTIME_TAPE_H */
