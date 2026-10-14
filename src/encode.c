#include "nir_c_runtime/encode.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char **error, const char *message) {
    if (error == NULL || *error != NULL) {
        return;
    }
    size_t length = strlen(message) + 1;
    *error = (char *)malloc(length);
    if (*error != NULL) {
        memcpy(*error, message, length);
    }
}

static float clamp01(float v) {
    if (v < 0.0f) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    return v;
}

int nir_encode_rate(
    const float *x, size_t n, size_t num_steps, float gain,
    nir_rng_t *rng, uint8_t *out, char **error
) {
    if (x == NULL || rng == NULL || out == NULL || n == 0 ||
        num_steps == 0 || !isfinite(gain)) {
        set_error(error, "rate encode needs inputs, steps, and a gain");
        return 0;
    }
    for (size_t t = 0; t < num_steps; t++) {
        for (size_t i = 0; i < n; i++) {
            float p = clamp01(gain * x[i]);
            out[t * n + i] = nir_rng_next_f32(rng) < p ? 1 : 0;
        }
    }
    return 1;
}

static int latency_param_ok(
    const nir_latency_params_t *p, char **error
) {
    if (p == NULL || p->num_steps == 0 || !(p->tau > 0.0f) ||
        p->first_spike_time >= p->num_steps) {
        set_error(error, "latency encode needs steps, tau, spike time");
        return 0;
    }
    if (p->linear) {
        if (!(p->threshold >= 0.0f) || !(p->threshold <= 1.0f)) {
            set_error(error, "linear latency threshold must lie in [0, 1]");
            return 0;
        }
    } else if (!(p->threshold > 0.0f) || !(p->threshold < 1.0f)) {
        set_error(error, "log latency threshold must lie in (0, 1)");
        return 0;
    }
    return 1;
}

/* Linear branch: st = min(-tau*(x-1), -tau*(thr-1)) + first, with
 * tau replaced by steps-1-first when normalizing. */
static void latency_linear(
    const float *x, size_t n, const nir_latency_params_t *p, float *st
) {
    float tau = p->normalize
        ? (float)(p->num_steps - 1 - p->first_spike_time)
        : p->tau;
    float cap = -tau * (p->threshold - 1.0f);
    for (size_t i = 0; i < n; i++) {
        float time = -tau * (x[i] - 1.0f);
        st[i] = (time < cap ? time : cap) + (float)p->first_spike_time;
    }
}

/* Log branch: st = tau*ln(xc/(xc-thr)) (+ normalize rescale), with
 * inputs saturated at thr+eps. */
static void latency_log(
    const float *x, size_t n, const nir_latency_params_t *p, float *st
) {
    float floor = p->threshold + p->epsilon;
    float first = (float)p->first_spike_time;
    float max_shift = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float xc = x[i] < floor ? floor : x[i];
        st[i] = p->tau * logf(xc / (xc - p->threshold)) + first;
        if (st[i] - first > max_shift) {
            max_shift = st[i] - first;
        }
    }
    if (p->normalize) {
        float span = (float)(p->num_steps - 1) - first;
        for (size_t i = 0; i < n; i++) {
            st[i] = max_shift > 0.0f
                ? (st[i] - first) * span / max_shift + first
                : first;
        }
    }
}

int nir_encode_latency(
    const float *x, size_t n, const nir_latency_params_t *params,
    uint8_t *out, char **error
) {
    if (x == NULL || out == NULL || n == 0 ||
        !latency_param_ok(params, error)) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        if (!(x[i] >= 0.0f) || !(x[i] <= 1.0f)) {
            set_error(error, "latency input must lie in [0, 1]");
            return 0;
        }
    }
    float *st = (float *)malloc(n * sizeof(float));
    if (st == NULL) {
        set_error(error, "out of memory");
        return 0;
    }
    if (params->linear) {
        latency_linear(x, n, params, st);
    } else {
        latency_log(x, n, params, st);
    }
    /* spikegen raises when the latest spike overflows num_steps
     * (SpikeEncoder never bypasses); check before scattering. */
    double latest = 0.0;
    for (size_t i = 0; i < n; i++) {
        double rounded = rint((double)st[i]);
        if (rounded > latest) {
            latest = rounded;
        }
    }
    if (latest > (double)(params->num_steps - 1)) {
        free(st);
        set_error(error, "latency spike time exceeds num_steps");
        return 0;
    }
    memset(out, 0, params->num_steps * n);
    for (size_t i = 0; i < n; i++) {
        if (params->clip && x[i] < params->threshold) {
            continue;
        }
        size_t t = (size_t)rint((double)st[i]);
        if (t > params->num_steps - 1) {
            continue; /* rm_idx: drop out-of-range spikes */
        }
        out[t * n + i] = 1;
    }
    free(st);
    return 1;
}

int nir_encode_delta(
    const float *x, size_t n, size_t num_steps, float threshold,
    uint8_t *out, char **error
) {
    if (x == NULL || out == NULL || n == 0 || num_steps == 0 ||
        !isfinite(threshold)) {
        set_error(error, "delta encode needs inputs and steps");
        return 0;
    }
    /* linspace(0, 1, steps): t/(steps-1), or 0 for a single step. */
    for (size_t t = 0; t < num_steps; t++) {
        float ramp = num_steps > 1
            ? (float)t / (float)(num_steps - 1)
            : 0.0f;
        float prev = num_steps > 1 && t > 0
            ? (float)(t - 1) / (float)(num_steps - 1)
            : 0.0f;
        for (size_t i = 0; i < n; i++) {
            /* Separate statements: one expression could contract. */
            float level = ramp * x[i];
            float before = prev * x[i];
            out[t * n + i] = (level - before) >= threshold ? 1 : 0;
        }
    }
    return 1;
}

int nir_encode_random(
    size_t n, size_t num_steps, float scale, nir_rng_t *rng,
    uint8_t *out, char **error
) {
    if (rng == NULL || out == NULL || n == 0 || num_steps == 0 ||
        !isfinite(scale)) {
        set_error(error, "random encode needs steps, scale, and an RNG");
        return 0;
    }
    for (size_t t = 0; t < num_steps; t++) {
        for (size_t i = 0; i < n; i++) {
            float magnitude = nir_rng_next_f32(rng);
            float p = clamp01(scale * magnitude);
            out[t * n + i] = nir_rng_next_f32(rng) < p ? 1 : 0;
        }
    }
    return 1;
}

int nir_encode_repeat(
    const float *x, size_t n, size_t num_steps, float *out,
    char **error
) {
    if (x == NULL || out == NULL || n == 0 || num_steps == 0) {
        set_error(error, "repeat encode needs inputs and steps");
        return 0;
    }
    for (size_t t = 0; t < num_steps; t++) {
        memcpy(&out[t * n], x, n * sizeof(float));
    }
    return 1;
}
