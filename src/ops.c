#include "nir_c_runtime/ops.h"

#include <math.h>
#include <string.h>

static size_t conv_out_dim(
    size_t in, int padding, int dilation, int kernel, int stride
) {
    long numerator = (long)in + 2L * padding - dilation * (kernel - 1) - 1;
    return (size_t)(numerator / stride + 1);
}

static int prepare_output(
    nir_tensor_t *output, const size_t *shape, size_t ndim
) {
    if (output == NULL || output->data == NULL || ndim > NIR_TENSOR_MAX_DIMS) {
        return 0;
    }
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    if (output->capacity < size) {
        return 0;
    }
    output->ndim = ndim;
    memset(output->shape, 0, sizeof(output->shape));
    for (size_t i = 0; i < ndim; i++) {
        output->shape[i] = shape[i];
    }
    output->size = size;
    return 1;
}

int nir_op_conv2d_into(
    const nir_conv2d_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
) {
    if (params == NULL || input == NULL || input->data == NULL ||
        params->weight == NULL) {
        return 0;
    }
    size_t in_h = input->shape[1], in_w = input->shape[2];
    size_t out_h = conv_out_dim(
        in_h, params->padding_h, params->dilation_h,
        (int)params->kernel_h, params->stride_h
    );
    size_t out_w = conv_out_dim(
        in_w, params->padding_w, params->dilation_w,
        (int)params->kernel_w, params->stride_w
    );
    size_t shape[3] = {params->out_channels, out_h, out_w};
    if (!prepare_output(output, shape, 3)) {
        return 0;
    }

    for (size_t oc = 0; oc < params->out_channels; oc++) {
        float bias = params->bias != NULL ? params->bias[oc] : 0.0f;
        for (size_t oh = 0; oh < out_h; oh++) {
            for (size_t ow = 0; ow < out_w; ow++) {
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
    return 1;
}

nir_tensor_t nir_op_conv2d(
    const nir_conv2d_params_t *params, const nir_tensor_t *input
) {
    size_t shape[3] = {0, 0, 0};
    if (params != NULL && input != NULL) {
        shape[0] = params->out_channels;
        shape[1] = conv_out_dim(
            input->shape[1], params->padding_h, params->dilation_h,
            (int)params->kernel_h, params->stride_h
        );
        shape[2] = conv_out_dim(
            input->shape[2], params->padding_w, params->dilation_w,
            (int)params->kernel_w, params->stride_w
        );
    }
    nir_tensor_t output = nir_tensor_alloc(shape, 3);
    if (output.data != NULL &&
        !nir_op_conv2d_into(params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

int nir_op_affine_into(
    const nir_affine_params_t *params, const nir_tensor_t *input,
    nir_tensor_t *output
) {
    if (params == NULL || input == NULL || input->data == NULL ||
        params->weight == NULL || params->bias == NULL) {
        return 0;
    }
    size_t shape[1] = {params->out_features};
    if (!prepare_output(output, shape, 1)) {
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
    size_t shape[1] = {params->out_features};
    nir_tensor_t output = nir_tensor_alloc(shape, 1);
    if (output.data != NULL &&
        !nir_op_affine_into(params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}

int nir_op_flatten_into(const nir_tensor_t *input, nir_tensor_t *output) {
    if (input == NULL || input->data == NULL) {
        return 0;
    }
    size_t shape[1] = {input->size};
    if (!prepare_output(output, shape, 1)) {
        return 0;
    }
    memcpy(output->data, input->data, input->size * sizeof(float));
    return 1;
}

nir_tensor_t nir_op_flatten(const nir_tensor_t *input) {
    size_t shape[1] = {input != NULL ? input->size : 0};
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
    if (params == NULL || input == NULL || input->data == NULL) {
        return 0;
    }
    size_t channels = input->shape[0], in_h = input->shape[1],
           in_w = input->shape[2];
    size_t out_h = (size_t)(
        ((long)in_h + 2L * params->padding_h - params->kernel_h)
            / params->stride_h + 1
    );
    size_t out_w = (size_t)(
        ((long)in_w + 2L * params->padding_w - params->kernel_w)
            / params->stride_w + 1
    );
    size_t shape[3] = {channels, out_h, out_w};
    if (!prepare_output(output, shape, 3)) {
        return 0;
    }
    float area = (float)(params->kernel_h * params->kernel_w);

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
    if (params != NULL && input != NULL) {
        shape[0] = input->shape[0];
        shape[1] = (size_t)(((long)input->shape[1] +
            2L * params->padding_h - params->kernel_h) /
            params->stride_h + 1);
        shape[2] = (size_t)(((long)input->shape[2] +
            2L * params->padding_w - params->kernel_w) /
            params->stride_w + 1);
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
    if (params != NULL && input != NULL) {
        shape[0] = input->shape[0];
        shape[1] = (size_t)(((long)input->shape[1] +
            2L * params->padding_h - params->kernel_h) /
            params->stride_h + 1);
        shape[2] = (size_t)(((long)input->shape[2] +
            2L * params->padding_w - params->kernel_w) /
            params->stride_w + 1);
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
    float area = (float)(params->kernel_h * params->kernel_w);
    return pool2d_into(params, input, area, output);
}

int nir_op_li_into(
    const nir_li_params_t *params,
    const nir_tensor_t *input,
    const nir_tensor_t *v_prev,
    nir_tensor_t *output
) {
    if (params == NULL || input == NULL || input->data == NULL ||
        params->tau <= 0.0f) {
        return 0;
    }
    if (!prepare_output(output, input->shape, input->ndim)) {
        return 0;
    }
    float decay = expf(-1.0f / params->tau);
    for (size_t i = 0; i < input->size; i++) {
        float prev = v_prev != NULL ? v_prev->data[i] : 0.0f;
        output->data[i] = params->v_leak
            + (prev - params->v_leak) * decay
            + params->r * (1.0f - decay) * input->data[i];
    }
    return 1;
}

nir_tensor_t nir_op_li(
    const nir_li_params_t *params,
    const nir_tensor_t *input,
    const nir_tensor_t *v_prev
) {
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
    if (params == NULL || input == NULL || input->data == NULL ||
        !prepare_output(output, input->shape, input->ndim)) {
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
    if (params == NULL || input == NULL || input->data == NULL ||
        !prepare_output(output, input->shape, input->ndim)) {
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
    nir_tensor_t output = nir_tensor_alloc(input->shape, input->ndim);
    if (output.data != NULL && !nir_op_scale_into(
            params, input, &output)) {
        nir_tensor_free(&output);
    }
    return output;
}
