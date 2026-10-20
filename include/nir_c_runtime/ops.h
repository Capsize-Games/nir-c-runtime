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

/* Elementwise rectified linear unit: y = max(0, input). */
nir_tensor_t nir_op_relu(const nir_tensor_t *input);
int nir_op_relu_into(const nir_tensor_t *input, nir_tensor_t *output);

/* Training: single-step backward ops. The forward spike
 * nonlinearities stay strict Heavisides; the backward pass below
 * substitutes a smooth surrogate derivative, mirroring snnTorch's
 * `surrogate` module (see spikeforge/introspection/surrogate.py).
 * Names resolve through nir_surrogate_from_name in train.h, which
 * rejects unknown names instead of silently defaulting. */

typedef enum {
    NIR_SURROGATE_FAST_SIGMOID,
    NIR_SURROGATE_ATAN
} nir_surrogate_kind_t;

/* Surrogate derivative dS/dx at x. fast_sigmoid with slope s:
 * s / (1 + |s*x|)^2 (snnTorch default s = 25). atan with
 * steepness a: (a/2) / (1 + (pi*a*x/2)^2) (snnTorch default
 * a = 2). Returns 0 unless `kind` is known and `slope` is
 * finite and positive. */
int nir_surrogate_grad(
    nir_surrogate_kind_t kind, float slope, float x, float *out
);

/* Backward through Threshold: forward is the strict Heaviside,
 * and grad_input[i] = grad_output[i] * surr(membrane[i] -
 * threshold). `membrane` is the retained pre-threshold input.
 * Shapes must match; `grad_input` may exactly alias
 * `grad_output` (elementwise rule). */
int nir_op_threshold_backward(
    const nir_threshold_params_t *params, const nir_tensor_t *membrane,
    const nir_tensor_t *grad_output, nir_surrogate_kind_t surrogate,
    float slope, nir_tensor_t *grad_input
);

/* Backward through ReLU: grad_input[i] is grad_output[i] when
 * input[i] > 0, and zero otherwise (including input exactly zero).
 * Shapes must match; grad_input may exactly alias grad_output. */
int nir_op_relu_backward(
    const nir_tensor_t *input, const nir_tensor_t *grad_output,
    nir_tensor_t *grad_input
);

/* Backward through one LI step: with decay = exp(-1 / tau),
 * grad_input = grad_output * r * (1 - decay) and grad_prev =
 * grad_output * decay. `grad_prev` may be NULL when the
 * previous-state gradient is unneeded. Either output may exactly
 * alias `grad_output`; the two outputs must not overlap each
 * other. */
int nir_op_li_backward(
    const nir_li_params_t *params, const nir_tensor_t *grad_output,
    nir_tensor_t *grad_input, nir_tensor_t *grad_prev
);

/* Backward through y = Wx + b: grad_weight += outer(grad_output,
 * input), grad_bias += grad_output, grad_input = W^T grad_output.
 * Weight and bias gradients accumulate (the caller zeroes them);
 * `grad_bias` may be NULL when the node has no bias. All three
 * outputs must be disjoint from the inputs and from each other. */
int nir_op_affine_backward(
    const nir_affine_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *grad_output, nir_tensor_t *grad_weight,
    nir_tensor_t *grad_bias, nir_tensor_t *grad_input
);

/* Backward through y = Wx: as above without the bias term. */
int nir_op_linear_backward(
    const nir_linear_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *grad_output, nir_tensor_t *grad_weight,
    nir_tensor_t *grad_input
);

/* Backward through Conv2d: grad_input via transposed convolution
 * (zero-filled, then each grad_output element scattered back
 * through the kernel), grad_weight += the input-window outer sums,
 * grad_bias += grad_output summed over spatial positions.
 * grad_weight is a flat [out*in*kernel_h*kernel_w] vector in the
 * same linear order as params->weight (a 4D shape would exceed
 * NIR_TENSOR_MAX_DIMS). Weight and bias gradients accumulate (the
 * caller zeroes them); grad_input is overwritten. grad_bias may be
 * NULL iff the node has no bias. All three outputs must be
 * disjoint from the inputs and from each other. */
int nir_op_conv2d_backward(
    const nir_conv2d_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *grad_output, nir_tensor_t *grad_weight,
    nir_tensor_t *grad_bias, nir_tensor_t *grad_input
);

/* Backward through Conv1d: as above in one spatial dimension, with
 * grad_weight a flat [out*in*kernel] vector. */
int nir_op_conv1d_backward(
    const nir_conv1d_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *grad_output, nir_tensor_t *grad_weight,
    nir_tensor_t *grad_bias, nir_tensor_t *grad_input
);

/* Backward through Flatten: copy the flat gradient back into the
 * retained input's shape (grad_input takes input's shape). Only
 * shapes are read from `input`. Overlap-safe through a move-safe
 * copy, like the forward. */
int nir_op_flatten_backward(
    const nir_tensor_t *input, const nir_tensor_t *grad_output,
    nir_tensor_t *grad_input
);

/* Backward through AvgPool2d (count-include-pad): grad_input is
 * zero-filled to the retained input's shape, then each
 * grad_output element is scattered as grad/area over its window
 * (padded cells receive nothing, and the divisor stays the full
 * window area, matching the forward). grad_input is overwritten
 * and must be disjoint from the inputs. */
int nir_op_avgpool2d_backward(
    const nir_pool2d_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *grad_output, nir_tensor_t *grad_input
);

/* Backward through SumPool2d: as above with unit weight per
 * window cell. */
int nir_op_sumpool2d_backward(
    const nir_pool2d_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *grad_output, nir_tensor_t *grad_input
);

/* Backward through Scale: grad_input = grad_output * scale,
 * following the elementwise aliasing rule. The node's scale is a
 * fixed graph constant, so the scale gradient is opt-in: when
 * grad_scale is non-NULL, *grad_scale += sum(grad_output * input)
 * (accumulating, mirroring affine_backward) and `input` must be a
 * valid same-shape tensor; when grad_scale is NULL, `input` may be
 * NULL and only grad_input is written. */
int nir_op_scale_backward(
    const nir_scale_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *grad_output, nir_tensor_t *grad_input,
    float *grad_scale
);

/* Spiking-neuron backward convention (LIF/IF/CubaLIF). The forward
 * hard reset (stored = spike ? v_reset : membrane) carries ZERO
 * gradient: grad_stored is accepted for tape wiring but
 * contributes nothing, so no gradient flows through the reset
 * branch. The learning signal flows through the spike output with
 * a surrogate derivative, d(spike)/d(membrane) =
 * surr(membrane - v_threshold), mirroring snnTorch's
 * straight-through spike gradient (the same fast_sigmoid/atan
 * derivatives as nir_surrogate_grad). This truncates BPTT through
 * the stored membrane across steps: within one step, only the
 * spike path carries credit to the input and previous state. The
 * membrane feeding the surrogate is recomputed from the retained
 * input and previous state, so the tape retains those rather than
 * a pre-reset membrane the forward never outputs. */

/* Backward through one LIF step: with decay = exp(-1 / tau) and
 * g = grad_spike * surr(membrane - v_threshold), grad_input = g *
 * r * (1 - decay) and grad_prev = g * decay. `v_prev` may be NULL
 * (treated as zeros, matching the forward); `grad_stored` may be
 * NULL (when non-NULL it must match the input shape and is
 * ignored per the reset convention above); `grad_prev` may be
 * NULL when the previous-state gradient is unneeded. Outputs
 * follow the elementwise rule against grad_spike and the retained
 * tensors, and must not overlap each other. */
int nir_op_lif_backward(
    const nir_lif_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *v_prev, const nir_tensor_t *grad_spike,
    const nir_tensor_t *grad_stored, nir_surrogate_kind_t surrogate,
    float slope, nir_tensor_t *grad_input, nir_tensor_t *grad_prev
);

/* Backward through one IF step: with membrane = v_prev + r *
 * input and g as above, grad_input = g * r and grad_prev = g.
 * Retention and aliasing conventions match lif_backward. */
int nir_op_if_backward(
    const nir_if_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *v_prev, const nir_tensor_t *grad_spike,
    const nir_tensor_t *grad_stored, nir_surrogate_kind_t surrogate,
    float slope, nir_tensor_t *grad_input, nir_tensor_t *grad_prev
);

/* Backward through one CubaLIF step: with g as above,
 * d(syn) = g * r * (1 - mem_decay), grad_mem_prev = g *
 * mem_decay, grad_input = d(syn) * w_in * (1 - syn_decay), and
 * grad_syn_prev = d(syn) * syn_decay. `syn_prev`/`mem_prev` prime
 * together (both NULL or both valid, matching the forward);
 * `grad_stored` follows the reset convention above; either state
 * gradient may be NULL when unneeded. */
int nir_op_cuba_backward(
    const nir_cuba_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *syn_prev, const nir_tensor_t *mem_prev,
    const nir_tensor_t *grad_spike, const nir_tensor_t *grad_stored,
    nir_surrogate_kind_t surrogate, float slope,
    nir_tensor_t *grad_input, nir_tensor_t *grad_syn_prev,
    nir_tensor_t *grad_mem_prev
);

#endif /* NIR_C_RUNTIME_OPS_H */
