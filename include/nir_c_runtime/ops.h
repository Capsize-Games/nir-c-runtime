/* Per-node-kind operations, matching spikeforge's reference NIR
 * interpreter (spikeforge/nir_bridge/ops_neuron.py, ops_linear.py)
 * numerically -- see each function's doc comment for the exact
 * formula it mirrors.
 */
#ifndef NIR_C_RUNTIME_OPS_H
#define NIR_C_RUNTIME_OPS_H

#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/tensor.h"

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
