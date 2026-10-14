#include "nir_c_runtime/minibatch.h"

#include <math.h>

void nir_minibatch_zero(float *accum, size_t n) {
    for (size_t i = 0; i < n; i++) {
        accum[i] = 0.0f;
    }
}

static int check_finite(const float *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(data[i])) return 0;
    }
    return 1;
}

int nir_minibatch_accumulate(
    float *accum, const float *grad, size_t n
) {
    if (accum == NULL || grad == NULL || n == 0 ||
        !check_finite(accum, n) || !check_finite(grad, n)) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        accum[i] += grad[i];
    }
    return 1;
}

int nir_minibatch_scale(float *accum, size_t n, float scale) {
    if (accum == NULL || n == 0 || !isfinite(scale) ||
        !check_finite(accum, n)) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        accum[i] *= scale;
    }
    return 1;
}

int nir_minibatch_mean_loss(
    const float *losses, size_t n, float *mean
) {
    if (losses == NULL || mean == NULL || n == 0 ||
        !check_finite(losses, n)) {
        return 0;
    }
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        total += (double)losses[i];
    }
    *mean = (float)(total / (double)n);
    return 1;
}

void nir_sampler_arange(size_t *indices, size_t n) {
    for (size_t i = 0; i < n; i++) {
        indices[i] = i;
    }
}

int nir_sampler_shuffle(nir_rng_t *rng, size_t *indices, size_t n) {
    if (rng == NULL || (indices == NULL && n > 0)) {
        return 0;
    }
    for (size_t i = n; i > 1; i--) {
        size_t j = (size_t)(nir_rng_next_u64(rng) % i);
        size_t last = i - 1;
        size_t tmp = indices[last];
        indices[last] = indices[j];
        indices[j] = tmp;
    }
    return 1;
}
