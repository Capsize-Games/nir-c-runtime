#include "nir_c_runtime/tensor.h"

#include <stdlib.h>
#include <string.h>

nir_tensor_t nir_tensor_alloc(const size_t *shape, size_t ndim) {
    nir_tensor_t tensor;
    memset(&tensor, 0, sizeof(tensor));
    if (ndim > NIR_TENSOR_MAX_DIMS) {
        return tensor;
    }
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        tensor.shape[i] = shape[i];
        size *= shape[i];
    }
    tensor.ndim = ndim;
    tensor.size = size;
    tensor.capacity = size;
    tensor.data = (float *)calloc(size == 0 ? 1 : size, sizeof(float));
    tensor.owns_data = tensor.data != NULL;
    return tensor;
}

nir_tensor_t nir_tensor_clone(const nir_tensor_t *src) {
    nir_tensor_t out = nir_tensor_alloc(src->shape, src->ndim);
    if (out.data != NULL && src->data != NULL) {
        memcpy(out.data, src->data, src->size * sizeof(float));
    }
    return out;
}

int nir_tensor_bind(
    nir_tensor_t *tensor, float *data, size_t capacity,
    const size_t *shape, size_t ndim
) {
    if (tensor == NULL || data == NULL || shape == NULL ||
        ndim > NIR_TENSOR_MAX_DIMS) {
        return 0;
    }
    size_t size = 1;
    size_t bound_shape[NIR_TENSOR_MAX_DIMS] = {0};
    for (size_t i = 0; i < ndim; i++) {
        bound_shape[i] = shape[i];
        size *= bound_shape[i];
    }
    if (capacity < size) {
        return 0;
    }
    if (tensor->owns_data) {
        free(tensor->data);
    }
    memset(tensor, 0, sizeof(*tensor));
    tensor->data = data;
    tensor->ndim = ndim;
    memcpy(tensor->shape, bound_shape, sizeof(bound_shape));
    tensor->size = size;
    tensor->capacity = capacity;
    return 1;
}

int nir_tensor_copy_into(nir_tensor_t *out, const nir_tensor_t *src) {
    if (out == NULL || src == NULL || out->data == NULL || src->data == NULL ||
        out->capacity < src->size) {
        return 0;
    }
    out->ndim = src->ndim;
    memcpy(out->shape, src->shape, sizeof(out->shape));
    out->size = src->size;
    memcpy(out->data, src->data, src->size * sizeof(float));
    return 1;
}

void nir_tensor_free(nir_tensor_t *tensor) {
    if (tensor == NULL) {
        return;
    }
    if (tensor->owns_data) {
        free(tensor->data);
    }
    memset(tensor, 0, sizeof(*tensor));
}

int nir_tensor_add_inplace(nir_tensor_t *lhs, const nir_tensor_t *rhs) {
    if (lhs->size == rhs->size) {
        for (size_t i = 0; i < lhs->size; i++) {
            lhs->data[i] += rhs->data[i];
        }
        return 1;
    }
    if (rhs->size == 1) {
        for (size_t i = 0; i < lhs->size; i++) {
            lhs->data[i] += rhs->data[0];
        }
        return 1;
    }
    return 0;
}
