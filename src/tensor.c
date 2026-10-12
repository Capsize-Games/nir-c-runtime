#include "nir_c_runtime/tensor.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static int tensor_shape_size(
    const size_t *shape, size_t ndim, size_t *size
) {
    if (ndim > NIR_TENSOR_MAX_DIMS || (ndim != 0 && shape == NULL)) {
        return 0;
    }
    size_t product = 1;
    for (size_t i = 0; i < ndim; i++) {
        if (shape[i] != 0 && product > SIZE_MAX / shape[i]) return 0;
        product *= shape[i];
    }
    if (product > SIZE_MAX / sizeof(float)) return 0;
    *size = product;
    return 1;
}

static int tensor_is_valid(const nir_tensor_t *tensor) {
    size_t size;
    return tensor != NULL && tensor->data != NULL &&
        tensor_shape_size(tensor->shape, tensor->ndim, &size) &&
        size == tensor->size && tensor->capacity >= size;
}

nir_tensor_t nir_tensor_alloc(const size_t *shape, size_t ndim) {
    nir_tensor_t tensor;
    memset(&tensor, 0, sizeof(tensor));
    size_t size;
    if (!tensor_shape_size(shape, ndim, &size)) {
        return tensor;
    }
    for (size_t i = 0; i < ndim; i++) {
        tensor.shape[i] = shape[i];
    }
    tensor.ndim = ndim;
    tensor.size = size;
    tensor.capacity = size;
    tensor.data = (float *)calloc(size == 0 ? 1 : size, sizeof(float));
    tensor.owns_data = tensor.data != NULL;
    return tensor;
}

nir_tensor_t nir_tensor_clone(const nir_tensor_t *src) {
    nir_tensor_t empty = {0};
    if (!tensor_is_valid(src)) return empty;
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
    if (tensor == NULL || data == NULL) {
        return 0;
    }
    size_t size;
    if (!tensor_shape_size(shape, ndim, &size)) return 0;
    size_t bound_shape[NIR_TENSOR_MAX_DIMS] = {0};
    for (size_t i = 0; i < ndim; i++) {
        bound_shape[i] = shape[i];
    }
    if (capacity < size) {
        return 0;
    }
    int preserve_ownership = 0;
    if (tensor->owns_data && tensor->data != NULL) {
        uintptr_t base = (uintptr_t)tensor->data;
        uintptr_t requested = (uintptr_t)data;
        size_t original_capacity = tensor->capacity;
        size_t allocation_count = tensor->size == 0 ? 1 : tensor->size;
        if (allocation_count > SIZE_MAX / sizeof(float)) return 0;
        size_t bytes = allocation_count * sizeof(float);
        if (base <= UINTPTR_MAX - bytes && requested >= base && requested < base + bytes) {
            if (requested != base) return 0;
            preserve_ownership = 1;
        }
        if (!preserve_ownership && requested == base) preserve_ownership = 1;
        if (preserve_ownership && capacity > original_capacity) return 0;
        if (!preserve_ownership) free(tensor->data);
    }
    memset(tensor, 0, sizeof(*tensor));
    tensor->data = data;
    tensor->ndim = ndim;
    memcpy(tensor->shape, bound_shape, sizeof(bound_shape));
    tensor->size = size;
    tensor->capacity = capacity;
    tensor->owns_data = preserve_ownership;
    return 1;
}

int nir_tensor_copy_into(nir_tensor_t *out, const nir_tensor_t *src) {
    if (out == NULL || out->data == NULL || !tensor_is_valid(src) ||
        out->capacity < src->size) {
        return 0;
    }
    out->ndim = src->ndim;
    memcpy(out->shape, src->shape, sizeof(out->shape));
    out->size = src->size;
    memmove(out->data, src->data, src->size * sizeof(float));
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
    if (!tensor_is_valid(lhs) || !tensor_is_valid(rhs)) return 0;
    if (lhs->size == rhs->size) {
        uintptr_t lhs_begin = (uintptr_t)lhs->data;
        uintptr_t rhs_begin = (uintptr_t)rhs->data;
        size_t bytes = lhs->size * sizeof(float);
        if (lhs->data != rhs->data && bytes != 0 &&
            lhs_begin <= UINTPTR_MAX - bytes && rhs_begin <= UINTPTR_MAX - bytes &&
            lhs_begin < rhs_begin + bytes && rhs_begin < lhs_begin + bytes) {
            return 0;
        }
        for (size_t i = 0; i < lhs->size; i++) {
            lhs->data[i] += rhs->data[i];
        }
        return 1;
    }
    if (rhs->size == 1) {
        float scalar = rhs->data[0];
        for (size_t i = 0; i < lhs->size; i++) {
            lhs->data[i] += scalar;
        }
        return 1;
    }
    return 0;
}
