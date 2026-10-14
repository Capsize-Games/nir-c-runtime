#include "nir_c_runtime/init.h"

#include <math.h>
#include <stdint.h>

static int field_size(
    const size_t *shape, size_t ndim, size_t *field
) {
    size_t f = 1;
    for (size_t i = 2; i < ndim; i++) {
        if (shape[i] == 0 || f > SIZE_MAX / shape[i]) {
            return 0;
        }
        f *= shape[i];
    }
    *field = f;
    return 1;
}

int nir_init_fans(
    const size_t *shape, size_t ndim, size_t *fan_in,
    size_t *fan_out
) {
    size_t field = 0;
    if (shape == NULL || ndim < 2 || fan_in == NULL ||
        fan_out == NULL || shape[0] == 0 || shape[1] == 0 ||
        !field_size(shape, ndim, &field) ||
        shape[1] > SIZE_MAX / field ||
        shape[0] > SIZE_MAX / field) {
        return 0;
    }
    *fan_in = shape[1] * field;
    *fan_out = shape[0] * field;
    return 1;
}

static int check_fill(
    const nir_rng_t *rng, const float *out, size_t count
) {
    return rng != NULL && out != NULL && count > 0;
}

int nir_init_uniform(
    nir_rng_t *rng, float *out, size_t count, float low,
    float high
) {
    if (!check_fill(rng, out, count) || !isfinite(low) ||
        !isfinite(high) || !(low < high)) {
        return 0;
    }
    double span = (double)high - (double)low;
    for (size_t i = 0; i < count; i++) {
        double u = nir_rng_next_f64(rng);
        out[i] = (float)((double)low + span * u);
    }
    return 1;
}

/* One standard-normal draw via Box-Muller. u1 == 0 would feed
 * log(0), so resample it; the stream stays deterministic. */
static double gaussian(nir_rng_t *rng) {
    double u1 = 0.0;
    do {
        u1 = nir_rng_next_f64(rng);
    } while (u1 == 0.0);
    double u2 = nir_rng_next_f64(rng);
    double r = sqrt(-2.0 * log(u1));
    return r * cos(2.0 * acos(-1.0) * u2);
}

static int check_gain(float gain) {
    return isfinite(gain) && gain >= 0.0f;
}

int nir_init_xavier_uniform(
    nir_rng_t *rng, float *out, size_t count, size_t fan_in,
    size_t fan_out, float gain
) {
    if (!check_fill(rng, out, count) || fan_in == 0 ||
        fan_out == 0 || !check_gain(gain)) {
        return 0;
    }
    double total = (double)fan_in + (double)fan_out;
    double bound = (double)gain * sqrt(6.0 / total);
    for (size_t i = 0; i < count; i++) {
        double u = nir_rng_next_f64(rng);
        out[i] = (float)(-bound + 2.0 * bound * u);
    }
    return 1;
}

int nir_init_xavier_normal(
    nir_rng_t *rng, float *out, size_t count, size_t fan_in,
    size_t fan_out, float gain
) {
    if (!check_fill(rng, out, count) || fan_in == 0 ||
        fan_out == 0 || !check_gain(gain)) {
        return 0;
    }
    double total = (double)fan_in + (double)fan_out;
    double std = (double)gain * sqrt(2.0 / total);
    for (size_t i = 0; i < count; i++) {
        out[i] = (float)(std * gaussian(rng));
    }
    return 1;
}

int nir_init_he_uniform(
    nir_rng_t *rng, float *out, size_t count, size_t fan_in,
    float gain
) {
    if (!check_fill(rng, out, count) || fan_in == 0 ||
        !check_gain(gain)) {
        return 0;
    }
    double bound = (double)gain * sqrt(3.0 / (double)fan_in);
    for (size_t i = 0; i < count; i++) {
        double u = nir_rng_next_f64(rng);
        out[i] = (float)(-bound + 2.0 * bound * u);
    }
    return 1;
}

int nir_init_he_normal(
    nir_rng_t *rng, float *out, size_t count, size_t fan_in,
    float gain
) {
    if (!check_fill(rng, out, count) || fan_in == 0 ||
        !check_gain(gain)) {
        return 0;
    }
    double std = (double)gain / sqrt((double)fan_in);
    for (size_t i = 0; i < count; i++) {
        out[i] = (float)(std * gaussian(rng));
    }
    return 1;
}
