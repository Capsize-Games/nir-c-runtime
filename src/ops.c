#include "nir_c_runtime/ops.h"

#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define NIR_CONV_OUTPUT_TILE 4u

static int shape_size(const size_t *shape, size_t ndim, size_t *size) {
    if (shape == NULL || ndim > NIR_TENSOR_MAX_DIMS) return 0;
    size_t n = 1;
    for (size_t i = 0; i < ndim; i++) {
        if (shape[i] != 0 && n > SIZE_MAX / shape[i]) return 0;
        n *= shape[i];
    }
    if (n > SIZE_MAX / sizeof(float)) return 0;
    *size = n;
    return 1;
}

static int checked_mul(size_t a, size_t b, size_t *out) {
    if (b != 0 && a > SIZE_MAX / b) return 0;
    *out = a * b;
    return 1;
}

static int fits_padded_long(size_t in, int padding) {
    return padding >= 0 && (long long)padding * 2 <= LONG_MAX &&
        in <= (size_t)(LONG_MAX - (long)((long long)padding * 2));
}

static int tensor_valid(const nir_tensor_t *t, size_t ndim) {
    size_t size;
    return t != NULL && t->data != NULL && t->ndim == ndim &&
        shape_size(t->shape, t->ndim, &size) && size == t->size &&
        t->capacity >= size;
}

static int tensor_same_shape(const nir_tensor_t *a, const nir_tensor_t *b) {
    if (a->ndim != b->ndim) return 0;
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

static int tensors_overlap(const nir_tensor_t *a, const nir_tensor_t *b) {
    uintptr_t ab = (uintptr_t)a->data, bb = (uintptr_t)b->data;
    size_t as = a->size * sizeof(float), bs = b->size * sizeof(float);
    if (as == 0 || bs == 0) return 0;
    return ab <= UINTPTR_MAX - as && bb <= UINTPTR_MAX - bs &&
        ab < bb + bs && bb < ab + as;
}

static int tensors_partially_overlap(const nir_tensor_t *a, const nir_tensor_t *b) {
    return a->data != b->data && tensors_overlap(a, b);
}

static int overlaps_output_shape(
    const nir_tensor_t *input, const nir_tensor_t *output,
    const size_t *shape, size_t ndim
) {
    size_t size;
    if (output == NULL || output->data == NULL ||
        !shape_size(shape, ndim, &size)) return 0;
    nir_tensor_t planned = *output;
    planned.size = size;
    return tensors_overlap(input, &planned);
}

static int conv_out_dim(
    size_t in, int padding, int dilation, size_t kernel, int stride,
    size_t *out
) {
    if (!fits_padded_long(in, padding) || dilation <= 0 ||
        kernel == 0 || kernel > (size_t)INT_MAX || stride <= 0) return 0;
    long long numerator = (long long)in + 2LL * padding -
        (long long)dilation * ((long long)kernel - 1) - 1;
    if (numerator < 0) return 0;
    *out = (size_t)(numerator / stride + 1);
    return 1;
}

static int prepare_output(
    nir_tensor_t *output, const size_t *shape, size_t ndim
) {
    if (output == NULL || output->data == NULL || ndim > NIR_TENSOR_MAX_DIMS ||
        shape == NULL) {
        return 0;
    }
    size_t output_shape[NIR_TENSOR_MAX_DIMS] = {0};
    for (size_t i = 0; i < ndim; i++) output_shape[i] = shape[i];
    size_t size;
    if (!shape_size(output_shape, ndim, &size)) return 0;
    if (output->capacity < size) {
        return 0;
    }
    output->ndim = ndim;
    memcpy(output->shape, output_shape, sizeof(output_shape));
    output->size = size;
    return 1;
}

int nir_op_conv2d_into(
    const nir_conv2d_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
) {
    if (params == NULL || !tensor_valid(input, 3) || params->weight == NULL ||
        params->in_channels != input->shape[0] || params->out_channels == 0 ||
        params->in_channels == 0) {
        return 0;
    }
    size_t weight_count;
    if (!checked_mul(params->out_channels, params->in_channels, &weight_count) ||
        !checked_mul(weight_count, params->kernel_h, &weight_count) ||
        !checked_mul(weight_count, params->kernel_w, &weight_count) ||
        weight_count > SIZE_MAX / sizeof(float)) return 0;
    size_t in_h = input->shape[1], in_w = input->shape[2];
    size_t out_h, out_w;
    if (!conv_out_dim(
        in_h, params->padding_h, params->dilation_h,
        params->kernel_h, params->stride_h, &out_h) ||
        !conv_out_dim(
        in_w, params->padding_w, params->dilation_w,
        params->kernel_w, params->stride_w, &out_w)) return 0;
    size_t shape[3] = {params->out_channels, out_h, out_w};
    if (output == input || overlaps_output_shape(input, output, shape, 3) ||
        !prepare_output(output, shape, 3)) {
        return 0;
    }

    for (size_t oc = 0; oc < params->out_channels; oc++) {
        float bias = params->bias != NULL ? params->bias[oc] : 0.0f;
        for (size_t oh0 = 0; oh0 < out_h; oh0 += NIR_CONV_OUTPUT_TILE) {
            size_t oh_end = oh0 + NIR_CONV_OUTPUT_TILE;
            if (oh_end > out_h) oh_end = out_h;
            for (size_t ow0 = 0; ow0 < out_w;
                 ow0 += NIR_CONV_OUTPUT_TILE) {
                size_t ow_end = ow0 + NIR_CONV_OUTPUT_TILE;
                if (ow_end > out_w) ow_end = out_w;
                for (size_t oh = oh0; oh < oh_end; oh++) {
                    for (size_t ow = ow0; ow < ow_end; ow++) {
                        float sum = bias;
                        for (size_t ic = 0; ic < params->in_channels; ic++) {
                            for (size_t kh = 0; kh < params->kernel_h; kh++) {
                                long ih = (long)(oh * params->stride_h)
                                    - params->padding_h
                                    + (long)kh * params->dilation_h;
                                if (ih < 0 || (size_t)ih >= in_h) continue;
                                for (size_t kw = 0; kw < params->kernel_w; kw++) {
                                    long iw = (long)(ow * params->stride_w)
                                        - params->padding_w
                                        + (long)kw * params->dilation_w;
                                    if (iw < 0 || (size_t)iw >= in_w) continue;
                                    size_t in_index = (ic * in_h + (size_t)ih)
                                        * in_w + (size_t)iw;
                                    size_t w_index =
                                        ((oc * params->in_channels + ic)
                                            * params->kernel_h + kh)
                                        * params->kernel_w + kw;
                                    sum += input->data[in_index]
                                        * params->weight[w_index];
                                }
                            }
                        }
                        output->data[(oc * out_h + oh) * out_w + ow] = sum;
                    }
                }
            }
        }
    }
    return 1;
}

nir_tensor_t nir_op_conv2d(
    const nir_conv2d_params_t *params, const nir_tensor_t *input
) {
    size_t shape[3] = {0, 0, 0};
    if (params != NULL && tensor_valid(input, 3)) {
        shape[0] = params->out_channels;
        (void)conv_out_dim(
            input->shape[1], params->padding_h, params->dilation_h,
            params->kernel_h, params->stride_h, &shape[1]
        );
        (void)conv_out_dim(
            input->shape[2], params->padding_w, params->dilation_w,
            params->kernel_w, params->stride_w, &shape[2]
        );
    }
    nir_tensor_t output = nir_tensor_alloc(shape, 3);
    if (output.data != NULL &&
        !nir_op_conv2d_into(params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

int nir_op_conv1d_into(
    const nir_conv1d_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
) {
    if (params == NULL || !tensor_valid(input, 2) || params->weight == NULL ||
        params->in_channels != input->shape[0] || params->out_channels == 0 ||
        params->in_channels == 0) {
        return 0;
    }
    size_t weight_count;
    if (!checked_mul(params->out_channels, params->in_channels, &weight_count) ||
        !checked_mul(weight_count, params->kernel, &weight_count) ||
        weight_count > SIZE_MAX / sizeof(float)) return 0;
    size_t in_len = input->shape[1];
    size_t out_len;
    if (!conv_out_dim(
        in_len, params->padding, params->dilation,
        params->kernel, params->stride, &out_len)) return 0;
    size_t shape[2] = {params->out_channels, out_len};
    if (output == input || overlaps_output_shape(input, output, shape, 2) ||
        !prepare_output(output, shape, 2)) {
        return 0;
    }

    for (size_t oc = 0; oc < params->out_channels; oc++) {
        float bias = params->bias != NULL ? params->bias[oc] : 0.0f;
        for (size_t ol = 0; ol < out_len; ol++) {
            float sum = bias;
            for (size_t ic = 0; ic < params->in_channels; ic++) {
                for (size_t k = 0; k < params->kernel; k++) {
                    long il = (long)(ol * (size_t)params->stride)
                        - params->padding
                        + (long)k * params->dilation;
                    if (il < 0 || (size_t)il >= in_len) continue;
                    size_t in_index = ic * in_len + (size_t)il;
                    size_t w_index = (oc * params->in_channels + ic)
                        * params->kernel + k;
                    sum += input->data[in_index] * params->weight[w_index];
                }
            }
            output->data[oc * out_len + ol] = sum;
        }
    }
    return 1;
}

nir_tensor_t nir_op_conv1d(
    const nir_conv1d_params_t *params, const nir_tensor_t *input
) {
    size_t shape[2] = {0, 0};
    if (params != NULL && tensor_valid(input, 2)) {
        shape[0] = params->out_channels;
        (void)conv_out_dim(
            input->shape[1], params->padding, params->dilation,
            params->kernel, params->stride, &shape[1]
        );
    }
    nir_tensor_t output = nir_tensor_alloc(shape, 2);
    if (output.data != NULL &&
        !nir_op_conv1d_into(params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

int nir_op_affine_into(
    const nir_affine_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
) {
    if (params == NULL || !tensor_valid(input, 1) ||
        input->size != params->in_features || params->weight == NULL ||
        params->bias == NULL) {
        return 0;
    }
    size_t weight_count;
    if (!checked_mul(params->out_features, params->in_features, &weight_count) ||
        weight_count > SIZE_MAX / sizeof(float) ||
        params->out_features > SIZE_MAX / sizeof(float)) return 0;
    size_t shape[1] = {params->out_features};
    if (output == input || overlaps_output_shape(input, output, shape, 1) ||
        !prepare_output(output, shape, 1)) {
        return 0;
    }
    for (size_t o = 0; o < params->out_features; o++) {
        float sum = params->bias[o];
        for (size_t i = 0; i < params->in_features; i++) {
            sum += params->weight[o * params->in_features + i]
                * input->data[i];
        }
        output->data[o] = sum;
    }
    return 1;
}

nir_tensor_t nir_op_affine(
    const nir_affine_params_t *params, const nir_tensor_t *input
) {
    if (params == NULL) return (nir_tensor_t){0};
    size_t shape[1] = {params->out_features};
    nir_tensor_t output = nir_tensor_alloc(shape, 1);
    if (output.data != NULL &&
        !nir_op_affine_into(params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

int nir_op_linear_into(
    const nir_linear_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
) {
    if (params == NULL || !tensor_valid(input, 1) ||
        input->size != params->in_features || params->weight == NULL) {
        return 0;
    }
    size_t weight_count;
    if (!checked_mul(params->out_features, params->in_features, &weight_count) ||
        weight_count > SIZE_MAX / sizeof(float) ||
        params->out_features > SIZE_MAX / sizeof(float)) return 0;
    size_t shape[1] = {params->out_features};
    if (output == input || overlaps_output_shape(input, output, shape, 1) ||
        !prepare_output(output, shape, 1)) {
        return 0;
    }
    for (size_t o = 0; o < params->out_features; o++) {
        float sum = 0.0f;
        for (size_t i = 0; i < params->in_features; i++) {
            sum += params->weight[o * params->in_features + i]
                * input->data[i];
        }
        output->data[o] = sum;
    }
    return 1;
}

nir_tensor_t nir_op_linear(
    const nir_linear_params_t *params, const nir_tensor_t *input
) {
    if (params == NULL) return (nir_tensor_t){0};
    size_t shape[1] = {params->out_features};
    nir_tensor_t output = nir_tensor_alloc(shape, 1);
    if (output.data != NULL &&
        !nir_op_linear_into(params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

int nir_op_flatten_into(const nir_tensor_t *input, nir_tensor_t *output) {
    if (!tensor_valid(input, input != NULL ? input->ndim : 0)) {
        return 0;
    }
    size_t shape[1] = {input->size};
    if (!prepare_output(output, shape, 1)) {
        return 0;
    }
    memmove(output->data, input->data, input->size * sizeof(float));
    return 1;
}

nir_tensor_t nir_op_flatten(const nir_tensor_t *input) {
    if (input == NULL) return (nir_tensor_t){0};
    size_t shape[1] = {input->size};
    nir_tensor_t output = nir_tensor_alloc(shape, 1);
    if (output.data != NULL && !nir_op_flatten_into(input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

static int pool2d_into(
    const nir_pool2d_params_t *params, const nir_tensor_t *input, float scale,
    nir_tensor_t *output
) {
    if (params == NULL || !tensor_valid(input, 3) || params->kernel_h <= 0 ||
        params->kernel_w <= 0 || params->stride_h <= 0 ||
        params->stride_w <= 0 || params->padding_h < 0 ||
        params->padding_w < 0) {
        return 0;
    }
    size_t channels = input->shape[0], in_h = input->shape[1],
           in_w = input->shape[2];
    if (!fits_padded_long(in_h, params->padding_h) ||
        !fits_padded_long(in_w, params->padding_w)) return 0;
    long long nh = (long long)in_h + 2LL * params->padding_h -
        params->kernel_h;
    long long nw = (long long)in_w + 2LL * params->padding_w -
        params->kernel_w;
    if (nh < 0 || nw < 0) return 0;
    size_t out_h = (size_t)(nh / params->stride_h + 1);
    size_t out_w = (size_t)(nw / params->stride_w + 1);
    size_t shape[3] = {channels, out_h, out_w};
    if (output == input || overlaps_output_shape(input, output, shape, 3) ||
        !prepare_output(output, shape, 3)) {
        return 0;
    }
    float area = (float)params->kernel_h * (float)params->kernel_w;

    for (size_t c = 0; c < channels; c++) {
        for (size_t oh = 0; oh < out_h; oh++) {
            for (size_t ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;
                for (int kh = 0; kh < params->kernel_h; kh++) {
                    long ih = (long)(oh * params->stride_h)
                        - params->padding_h + kh;
                    if (ih < 0 || (size_t)ih >= in_h) continue;
                    for (int kw = 0; kw < params->kernel_w; kw++) {
                        long iw = (long)(ow * params->stride_w)
                            - params->padding_w + kw;
                        if (iw < 0 || (size_t)iw >= in_w) continue;
                        sum += input->data[
                            (c * in_h + (size_t)ih) * in_w + (size_t)iw
                        ];
                    }
                }
                output->data[(c * out_h + oh) * out_w + ow] =
                    (sum / area) * scale;
            }
        }
    }
    return 1;
}

nir_tensor_t nir_op_avgpool2d(
    const nir_pool2d_params_t *params, const nir_tensor_t *input
) {
    size_t shape[3] = {0, 0, 0};
    if (params != NULL && tensor_valid(input, 3) && params->stride_h > 0 &&
        params->stride_w > 0 && params->kernel_h > 0 && params->kernel_w > 0 &&
        fits_padded_long(input->shape[1], params->padding_h) &&
        fits_padded_long(input->shape[2], params->padding_w)) {
        shape[0] = input->shape[0];
        long long nh = (long long)input->shape[1] + 2LL * params->padding_h - params->kernel_h;
        long long nw = (long long)input->shape[2] + 2LL * params->padding_w - params->kernel_w;
        if (nh >= 0 && nw >= 0) {
            shape[1] = (size_t)(nh / params->stride_h + 1);
            shape[2] = (size_t)(nw / params->stride_w + 1);
        }
    }
    nir_tensor_t output = nir_tensor_alloc(shape, 3);
    if (output.data != NULL &&
        !nir_op_avgpool2d_into(params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

int nir_op_avgpool2d_into(
    const nir_pool2d_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
) {
    return pool2d_into(params, input, 1.0f, output);
}

nir_tensor_t nir_op_sumpool2d(
    const nir_pool2d_params_t *params, const nir_tensor_t *input
) {
    size_t shape[3] = {0, 0, 0};
    if (params != NULL && tensor_valid(input, 3) && params->stride_h > 0 &&
        params->stride_w > 0 && params->kernel_h > 0 && params->kernel_w > 0 &&
        fits_padded_long(input->shape[1], params->padding_h) &&
        fits_padded_long(input->shape[2], params->padding_w)) {
        shape[0] = input->shape[0];
        long long nh = (long long)input->shape[1] + 2LL * params->padding_h - params->kernel_h;
        long long nw = (long long)input->shape[2] + 2LL * params->padding_w - params->kernel_w;
        if (nh >= 0 && nw >= 0) {
            shape[1] = (size_t)(nh / params->stride_h + 1);
            shape[2] = (size_t)(nw / params->stride_w + 1);
        }
    }
    nir_tensor_t output = nir_tensor_alloc(shape, 3);
    if (output.data != NULL &&
        !nir_op_sumpool2d_into(params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

int nir_op_sumpool2d_into(
    const nir_pool2d_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
) {
    if (params == NULL) {
        return 0;
    }
    if (params->kernel_h <= 0 || params->kernel_w <= 0) return 0;
    float area = (float)params->kernel_h * (float)params->kernel_w;
    return pool2d_into(params, input, area, output);
}

/* One zero-order-hold step of tau*v' = (v_leak - v) + r*current,
 * mirroring zoh_state in ops_neuron.py with identical association. */
static float zoh_step(
    float prev, float current, float tau, float r, float v_leak
) {
    float decay = expf(-1.0f / tau);
    return v_leak + (prev - v_leak) * decay
        + r * (1.0f - decay) * current;
}

/* The hard-reset (spike, stored) pair for one membrane value,
 * mirroring fire in ops_neuron.py. */
static void fire_reset(
    float membrane, float threshold, float reset,
    float *spike, float *stored
) {
    *spike = membrane > threshold ? 1.0f : 0.0f;
    *stored = *spike > 0.0f ? reset : membrane;
}

static int prev_matches(
    const nir_tensor_t *prev, const nir_tensor_t *input
) {
    return prev == NULL ||
        (tensor_valid(prev, input->ndim) &&
         tensor_same_shape(prev, input));
}

int nir_op_li_into(
    const nir_li_params_t *params,
    const nir_tensor_t *input,
    const nir_tensor_t *v_prev,
    nir_tensor_t *output
) {
    if (params == NULL || !tensor_valid(input, input != NULL ? input->ndim : 0) ||
        params->tau <= 0.0f || !isfinite(params->tau) ||
        !prev_matches(v_prev, input)) {
        return 0;
    }
    if (!prepare_output(output, input->shape, input->ndim) ||
        tensors_partially_overlap(input, output) ||
        (v_prev != NULL && tensors_partially_overlap(v_prev, output))) {
        return 0;
    }
    for (size_t i = 0; i < input->size; i++) {
        float prev = v_prev != NULL ? v_prev->data[i] : 0.0f;
        output->data[i] = zoh_step(
            prev, input->data[i], params->tau, params->r,
            params->v_leak
        );
    }
    return 1;
}

static int prepare_spiking_outputs(
    const nir_tensor_t *input, const nir_tensor_t *prev_a,
    const nir_tensor_t *prev_b, nir_tensor_t *spike_out,
    nir_tensor_t *stored_out
) {
    if (!prepare_output(spike_out, input->shape, input->ndim) ||
        !prepare_output(stored_out, input->shape, input->ndim) ||
        tensors_overlap(spike_out, stored_out) ||
        tensors_partially_overlap(input, spike_out) ||
        tensors_partially_overlap(input, stored_out) ||
        (prev_a != NULL && tensors_partially_overlap(prev_a, spike_out)) ||
        (prev_a != NULL && tensors_partially_overlap(prev_a, stored_out)) ||
        (prev_b != NULL && tensors_partially_overlap(prev_b, spike_out)) ||
        (prev_b != NULL && tensors_partially_overlap(prev_b, stored_out))) {
        return 0;
    }
    return 1;
}

int nir_op_lif_into(
    const nir_lif_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *v_prev, nir_tensor_t *spike_out,
    nir_tensor_t *stored_out
) {
    if (params == NULL || !tensor_valid(input, input != NULL ? input->ndim : 0) ||
        params->tau <= 0.0f || !isfinite(params->tau) ||
        !prev_matches(v_prev, input)) {
        return 0;
    }
    if (!prepare_spiking_outputs(
            input, v_prev, NULL, spike_out, stored_out)) {
        return 0;
    }
    for (size_t i = 0; i < input->size; i++) {
        float prev = v_prev != NULL ? v_prev->data[i] : 0.0f;
        float membrane = zoh_step(
            prev, input->data[i], params->tau, params->r,
            params->v_leak
        );
        fire_reset(
            membrane, params->v_threshold, params->v_reset,
            &spike_out->data[i], &stored_out->data[i]
        );
    }
    return 1;
}

int nir_op_if_into(
    const nir_if_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *v_prev, nir_tensor_t *spike_out,
    nir_tensor_t *stored_out
) {
    if (params == NULL || !tensor_valid(input, input != NULL ? input->ndim : 0) ||
        !prev_matches(v_prev, input)) {
        return 0;
    }
    if (!prepare_spiking_outputs(
            input, v_prev, NULL, spike_out, stored_out)) {
        return 0;
    }
    for (size_t i = 0; i < input->size; i++) {
        float prev = v_prev != NULL ? v_prev->data[i] : 0.0f;
        float membrane = prev + params->r * input->data[i];
        fire_reset(
            membrane, params->v_threshold, params->v_reset,
            &spike_out->data[i], &stored_out->data[i]
        );
    }
    return 1;
}

int nir_op_cuba_into(
    const nir_cuba_params_t *params, const nir_tensor_t *input,
    const nir_tensor_t *syn_prev, const nir_tensor_t *mem_prev,
    nir_tensor_t *spike_out, nir_tensor_t *syn_out,
    nir_tensor_t *stored_out
) {
    if (params == NULL || !tensor_valid(input, input != NULL ? input->ndim : 0) ||
        params->tau_syn <= 0.0f || !isfinite(params->tau_syn) ||
        params->tau_mem <= 0.0f || !isfinite(params->tau_mem) ||
        (syn_prev == NULL) != (mem_prev == NULL) ||
        !prev_matches(syn_prev, input) || !prev_matches(mem_prev, input)) {
        return 0;
    }
    if (!prepare_output(syn_out, input->shape, input->ndim) ||
        !prepare_spiking_outputs(
            input, syn_prev, mem_prev, spike_out, stored_out) ||
        tensors_overlap(syn_out, spike_out) ||
        tensors_overlap(syn_out, stored_out) ||
        tensors_partially_overlap(input, syn_out) ||
        (syn_prev != NULL && tensors_partially_overlap(syn_prev, syn_out)) ||
        (mem_prev != NULL && tensors_partially_overlap(mem_prev, syn_out))) {
        return 0;
    }
    for (size_t i = 0; i < input->size; i++) {
        float syn_previous = syn_prev != NULL ? syn_prev->data[i] : 0.0f;
        float mem_previous = mem_prev != NULL ? mem_prev->data[i] : 0.0f;
        float syn = zoh_step(
            syn_previous, input->data[i], params->tau_syn,
            params->w_in, 0.0f
        );
        float membrane = zoh_step(
            mem_previous, syn, params->tau_mem, params->r,
            params->v_leak
        );
        syn_out->data[i] = syn;
        fire_reset(
            membrane, params->v_threshold, params->v_reset,
            &spike_out->data[i], &stored_out->data[i]
        );
    }
    return 1;
}

nir_tensor_t nir_op_li(
    const nir_li_params_t *params,
    const nir_tensor_t *input,
    const nir_tensor_t *v_prev
) {
    if (input == NULL) return (nir_tensor_t){0};
    nir_tensor_t output = nir_tensor_alloc(input->shape, input->ndim);
    if (output.data != NULL && !nir_op_li_into(
            params, input, v_prev, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

int nir_op_threshold_into(
    const nir_threshold_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
) {
    if (params == NULL || !tensor_valid(input, input != NULL ? input->ndim : 0) ||
        !prepare_output(output, input->shape, input->ndim) ||
        tensors_partially_overlap(input, output)) {
        return 0;
    }
    for (size_t i = 0; i < input->size; i++) {
        output->data[i] = input->data[i] > params->threshold ? 1.0f : 0.0f;
    }
    return 1;
}

nir_tensor_t nir_op_threshold(
    const nir_threshold_params_t *params, const nir_tensor_t *input
) {
    if (input == NULL) return (nir_tensor_t){0};
    nir_tensor_t output = nir_tensor_alloc(input->shape, input->ndim);
    if (output.data != NULL && !nir_op_threshold_into(
            params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

int nir_op_scale_into(
    const nir_scale_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
) {
    if (params == NULL || !tensor_valid(input, input != NULL ? input->ndim : 0) ||
        !prepare_output(output, input->shape, input->ndim) ||
        tensors_partially_overlap(input, output)) {
        return 0;
    }
    for (size_t i = 0; i < input->size; i++) {
        output->data[i] = input->data[i] * params->scale;
    }
    return 1;
}

nir_tensor_t nir_op_scale(
    const nir_scale_params_t *params, const nir_tensor_t *input
) {
    if (input == NULL) return (nir_tensor_t){0};
    nir_tensor_t output = nir_tensor_alloc(input->shape, input->ndim);
    if (output.data != NULL && !nir_op_scale_into(
            params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}
