#include "nir_c_runtime/loss.h"

#include <math.h>
#include <stdint.h>

/* Exact aliasing is elementwise-safe; anything else that
 * overlaps is rejected, matching nir_cross_entropy. */
static int check_alias(
    const float *input, const float *other, size_t n
) {
    if (input == other) return 1;
    uintptr_t a = (uintptr_t)input;
    uintptr_t b = (uintptr_t)other;
    size_t bytes = n * sizeof(float);
    return !(a < b + bytes && b < a + bytes);
}

static int check_vector(
    const nir_tensor_t *t, const nir_tensor_t *grad
) {
    return t != NULL && t->data != NULL && t->ndim == 1 &&
        t->size > 0 && t->capacity >= t->size &&
        grad != NULL && grad->data != NULL &&
        grad->capacity >= t->size;
}

static int check_finite(const float *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(data[i])) return 0;
    }
    return 1;
}

static void shape_like(nir_tensor_t *grad, size_t n) {
    grad->ndim = 1;
    grad->shape[0] = n;
    grad->shape[1] = 0;
    grad->shape[2] = 0;
    grad->size = n;
}

static int check_mse(
    const nir_tensor_t *pred, const nir_tensor_t *target,
    const float *loss, const nir_tensor_t *grad
) {
    return check_vector(pred, grad) && target != NULL &&
        target->data != NULL && target->ndim == 1 &&
        target->size == pred->size &&
        target->capacity >= target->size && loss != NULL &&
        check_finite(pred->data, pred->size) &&
        check_finite(target->data, target->size) &&
        check_alias(pred->data, grad->data, pred->size) &&
        check_alias(target->data, grad->data, pred->size);
}

int nir_mse_loss(
    const nir_tensor_t *pred, const nir_tensor_t *target,
    float *loss, nir_tensor_t *grad
) {
    if (!check_mse(pred, target, loss, grad)) return 0;
    double total = 0.0;
    for (size_t i = 0; i < pred->size; i++) {
        double d = (double)pred->data[i] - (double)target->data[i];
        total += d * d;
    }
    shape_like(grad, pred->size);
    double n = (double)pred->size;
    for (size_t i = 0; i < pred->size; i++) {
        double d = (double)pred->data[i] - (double)target->data[i];
        grad->data[i] = (float)(2.0 * d / n);
    }
    *loss = (float)(total / n);
    return 1;
}

static double softmax_total(
    const nir_tensor_t *counts, float *peak
) {
    float p = counts->data[0];
    for (size_t i = 1; i < counts->size; i++) {
        if (counts->data[i] > p) p = counts->data[i];
    }
    *peak = p;
    double total = 0.0;
    for (size_t i = 0; i < counts->size; i++) {
        total += exp((double)counts->data[i] - (double)p);
    }
    return total;
}

static void softmax_into(
    const nir_tensor_t *counts, size_t target, nir_tensor_t *grad,
    float *loss
) {
    float peak = 0.0f;
    double total = softmax_total(counts, &peak);
    shape_like(grad, counts->size);
    for (size_t i = 0; i < counts->size; i++) {
        double p = exp((double)counts->data[i] - (double)peak);
        grad->data[i] = (float)(p / total -
            (i == target ? 1.0 : 0.0));
    }
    *loss = (float)-log(
        exp((double)counts->data[target] - (double)peak) / total
    );
}

int nir_spike_count_loss(
    const nir_tensor_t *counts, size_t target, float *loss,
    nir_tensor_t *grad
) {
    if (!check_vector(counts, grad) || target >= counts->size ||
        loss == NULL || !check_finite(counts->data, counts->size) ||
        !check_alias(counts->data, grad->data, counts->size)) {
        return 0;
    }
    softmax_into(counts, target, grad, loss);
    return 1;
}
