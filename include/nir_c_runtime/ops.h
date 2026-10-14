/* Per-node-kind operations, matching spikeforge's reference NIR
 * interpreter (spikeforge/nir_bridge/ops_neuron.py, ops_linear.py)
 * numerically -- see each function's doc comment for the exact
 * formula it mirrors.
 */
#ifndef NIR_C_RUNTIME_OPS_H
#define NIR_C_RUNTIME_OPS_H

#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/tensor.h"

/* For *_into calls, output storage must have capacity for the result. Spatial
 * operators and affine require storage disjoint from their inputs; elementwise
 * operators permit an exact data-pointer alias but reject partial overlap.
 * Flatten supports overlapping storage through a move-safe copy. */

/* 2D convolution, zero-padded, matching torch.nn.functional.conv2d.
 * `input` is (in_channels, H, W); the result is
 * (out_channels, out_H, out_W). */
nir_tensor_t nir_op_conv2d(
    const nir_conv2d_params_t *params, const nir_tensor_t *input
);
int nir_op_conv2d_into(
    const nir_conv2d_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
);

/* y = W @ x + b, matching torch.nn.functional.linear. `input` is
 * (in_features,); the result is (out_features,). */
nir_tensor_t nir_op_affine(
    const nir_affine_params_t *params, const nir_tensor_t *input
);
int nir_op_affine_into(
    const nir_affine_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
);

/* y = W @ x, matching torch.nn.functional.linear with no bias (the
 * NIR Linear node carries no bias term). `input` is (in_features,);
 * the result is (out_features,). */
nir_tensor_t nir_op_linear(
    const nir_linear_params_t *params, const nir_tensor_t *input
);
int nir_op_linear_into(
    const nir_linear_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
);

/* 1D convolution, zero-padded, matching torch.nn.functional.conv1d.
 * `input` is (in_channels, L); the result is (out_channels, out_L).
 * Grouped convolution is rejected at graph load (only groups=1, the
 * only setting the exporter emits, is accepted). */
nir_tensor_t nir_op_conv1d(
    const nir_conv1d_params_t *params, const nir_tensor_t *input
);
int nir_op_conv1d_into(
    const nir_conv1d_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
);

/* Reshape to a 1D tensor over the same (already contiguous) data --
 * this runtime's Flatten always covers every non-batch dimension,
 * matching every conv_net export's start_dim=1, end_dim=-1. */
nir_tensor_t nir_op_flatten(const nir_tensor_t *input);
int nir_op_flatten_into(const nir_tensor_t *input, nir_tensor_t *output);

/* Average pooling with padded cells counted in the divisor
 * (count_include_pad=True), matching torch.nn.functional.avg_pool2d. */
nir_tensor_t nir_op_avgpool2d(
    const nir_pool2d_params_t *params, const nir_tensor_t *input
);
int nir_op_avgpool2d_into(
    const nir_pool2d_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
);

/* Sum pooling: the windowed sum, zero-padded -- mathematically
 * avg_pool2d(...) * (kernel_h * kernel_w), matching spikeforge's own
 * apply_sumpool. */
nir_tensor_t nir_op_sumpool2d(
    const nir_pool2d_params_t *params, const nir_tensor_t *input
);
int nir_op_sumpool2d_into(
    const nir_pool2d_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
);

/* One zero-order-hold membrane-integration step:
 *   decay = exp(-1 / tau)
 *   v = v_leak + (v_prev - v_leak) * decay + r * (1 - decay) * input
 * `v_prev` may be NULL on a node's first step, treated as zeros. The
 * result is the new membrane (also this node's output value, per LI's
 * NIR semantics -- the caller retains it as the next step's v_prev). */
nir_tensor_t nir_op_li(
    const nir_li_params_t *params,
    const nir_tensor_t *input,
    const nir_tensor_t *v_prev
);
int nir_op_li_into(
    const nir_li_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *v_prev, nir_tensor_t *output
);

/* One leaky integrate-and-fire step: the LI zero-order-hold
 * integration above, then a strict-threshold hard reset --
 *   spike = membrane > v_threshold ? 1.0 : 0.0
 *   stored = spike ? v_reset : membrane
 * matching apply_lif in ops_neuron.py. `v_prev` may be NULL on a
 * node's first step (treated as zeros). The node's output value is
 * the spike train (`spike_out`); `stored_out` is the post-reset
 * membrane the caller retains as the next step's v_prev. There is no
 * allocating form: a two-output op cannot return one tensor. The two
 * outputs must not overlap each other; either may exactly alias an
 * input, following the elementwise rule above. */
int nir_op_lif_into(
    const nir_lif_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *v_prev, nir_tensor_t *spike_out,
    nir_tensor_t *stored_out
);

/* One integrate-and-fire step without leak:
 *   membrane = v_prev + r * input
 * followed by the same strict-threshold hard reset as LIF, matching
 * apply_if in ops_neuron.py. Output/state conventions are the same
 * as `nir_op_lif_into`. */
int nir_op_if_into(
    const nir_if_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *v_prev, nir_tensor_t *spike_out,
    nir_tensor_t *stored_out
);

/* One current-based LIF step: a zero-order-hold synaptic current
 *   syn = syn_prev * syn_decay + w_in * (1 - syn_decay) * input
 * feeds a zero-order-hold membrane
 *   membrane = v_leak + (mem_prev - v_leak) * mem_decay
 *              + r * (1 - mem_decay) * syn
 * followed by the same strict-threshold hard reset as LIF, matching
 * apply_cuba in ops_neuron.py. `syn_prev`/`mem_prev` may both be NULL
 * on a node's first step (treated as zeros); they prime together, so
 * passing one without the other is rejected. The node's output value
 * is the spike train; `syn_out`/`stored_out` are the next step's
 * synaptic current and post-reset membrane. The three outputs must
 * not overlap each other; any may exactly alias an input. */
int nir_op_cuba_into(
    const nir_cuba_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *syn_prev, const nir_tensor_t *mem_prev,
    nir_tensor_t *spike_out, nir_tensor_t *syn_out,
    nir_tensor_t *stored_out
);

/* Strict Heaviside: 1.0 where input > threshold, else 0.0. */
nir_tensor_t nir_op_threshold(
    const nir_threshold_params_t *params, const nir_tensor_t *input
);
int nir_op_threshold_into(
    const nir_threshold_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
);

/* out = input * scale, elementwise. */
nir_tensor_t nir_op_scale(
    const nir_scale_params_t *params, const nir_tensor_t *input
);
int nir_op_scale_into(
    const nir_scale_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
);

#endif /* NIR_C_RUNTIME_OPS_H */
