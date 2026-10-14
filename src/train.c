#include "nir_c_runtime/train.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char **error, const char *message) {
    if (error == NULL) return;
    *error = NULL;
    if (message == NULL) return;
    size_t n = strlen(message) + 1;
    char *copy = (char *)malloc(n);
    if (copy != NULL) memcpy(copy, message, n);
    *error = copy;
}

static void set_error_arg(
    char **error, const char *before, const char *name,
    const char *after
) {
    if (error == NULL) return;
    *error = NULL;
    if (before == NULL || name == NULL || after == NULL) return;
    size_t n = strlen(before) + strlen(name) + strlen(after) + 1;
    char *copy = (char *)malloc(n);
    if (copy == NULL) return;
    copy[0] = '\0';
    strcat(copy, before);
    strcat(copy, name);
    strcat(copy, after);
    *error = copy;
}

int nir_surrogate_from_name(
    const char *name, nir_surrogate_kind_t *out, char **error
) {
    if (error != NULL) *error = NULL;
    if (out == NULL) {
        set_error(error, "surrogate output must not be null");
        return 0;
    }
    if (name == NULL) {
        set_error(error, "unknown surrogate: (null)");
        return 0;
    }
    if (strcmp(name, "fast_sigmoid") == 0) {
        *out = NIR_SURROGATE_FAST_SIGMOID;
        return 1;
    }
    if (strcmp(name, "atan") == 0) {
        *out = NIR_SURROGATE_ATAN;
        return 1;
    }
    set_error_arg(error, "unknown surrogate: ", name, "");
    return 0;
}

int nir_cross_entropy(
    const nir_tensor_t *logits, size_t target, float *loss,
    nir_tensor_t *grad
) {
    if (logits == NULL || logits->data == NULL || logits->ndim != 1 ||
        logits->size == 0 || logits->capacity < logits->size ||
        target >= logits->size || loss == NULL || grad == NULL ||
        grad->data == NULL || grad->capacity < logits->size) {
        return 0;
    }
    for (size_t i = 0; i < logits->size; i++) {
        if (!isfinite(logits->data[i])) return 0;
    }
    if (logits->data != grad->data) {
        uintptr_t lo = (uintptr_t)logits->data;
        uintptr_t go = (uintptr_t)grad->data;
        size_t bytes = logits->size * sizeof(float);
        if (lo < go + bytes && go < lo + bytes) return 0;
    }
    float peak = logits->data[0];
    for (size_t i = 1; i < logits->size; i++) {
        if (logits->data[i] > peak) peak = logits->data[i];
    }
    double total = 0.0;
    for (size_t i = 0; i < logits->size; i++) {
        total += exp((double)logits->data[i] - (double)peak);
    }
    grad->ndim = 1;
    grad->shape[0] = logits->size;
    grad->shape[1] = 0;
    grad->shape[2] = 0;
    grad->size = logits->size;
    for (size_t i = 0; i < logits->size; i++) {
        double p = exp((double)logits->data[i] - (double)peak) / total;
        grad->data[i] = (float)(p - (i == target ? 1.0 : 0.0));
    }
    *loss = (float)-log(
        exp((double)logits->data[target] - (double)peak) / total
    );
    return 1;
}

int nir_sgd_init(
    nir_sgd_t *opt, size_t count, float lr, float momentum,
    float weight_decay, char **error
) {
    if (error != NULL) *error = NULL;
    if (opt == NULL) {
        set_error(error, "optimizer output must not be null");
        return 0;
    }
    memset(opt, 0, sizeof(*opt));
    if (count == 0 || !isfinite(lr) || lr < 0.0f ||
        !isfinite(momentum) || momentum < 0.0f ||
        !isfinite(weight_decay) || weight_decay < 0.0f) {
        set_error(error, "invalid SGD hyperparameters");
        return 0;
    }
    if (count > SIZE_MAX / sizeof(float)) {
        set_error(error, "optimizer state is too large");
        return 0;
    }
    opt->velocity = (float *)calloc(count, sizeof(float));
    if (opt->velocity == NULL) {
        set_error(error, "out of memory");
        return 0;
    }
    opt->lr = lr;
    opt->momentum = momentum;
    opt->weight_decay = weight_decay;
    opt->count = count;
    return 1;
}

int nir_sgd_step(
    nir_sgd_t *opt, float *weights, const float *grad, size_t count
) {
    if (opt == NULL || opt->velocity == NULL ||
        opt->count != count || count == 0 || weights == NULL ||
        grad == NULL) {
        return 0;
    }
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(weights[i]) || !isfinite(grad[i])) return 0;
    }
    for (size_t i = 0; i < count; i++) {
        float d = grad[i] + opt->weight_decay * weights[i];
        opt->velocity[i] = opt->momentum * opt->velocity[i] + d;
        weights[i] -= opt->lr * opt->velocity[i];
    }
    return 1;
}

void nir_sgd_free(nir_sgd_t *opt) {
    if (opt == NULL) return;
    free(opt->velocity);
    memset(opt, 0, sizeof(*opt));
}

int nir_adam_init(
    nir_adam_t *opt, size_t count, float lr, float beta1,
    float beta2, float eps, float weight_decay, char **error
) {
    if (error != NULL) *error = NULL;
    if (opt == NULL) {
        set_error(error, "optimizer output must not be null");
        return 0;
    }
    memset(opt, 0, sizeof(*opt));
    if (count == 0 || !isfinite(lr) || lr < 0.0f ||
        !isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !isfinite(eps) || eps <= 0.0f || !isfinite(weight_decay) ||
        weight_decay < 0.0f) {
        set_error(error, "invalid Adam hyperparameters");
        return 0;
    }
    if (count > SIZE_MAX / sizeof(float)) {
        set_error(error, "optimizer state is too large");
        return 0;
    }
    opt->m = (float *)calloc(count, sizeof(float));
    opt->v = (float *)calloc(count, sizeof(float));
    if (opt->m == NULL || opt->v == NULL) {
        free(opt->m);
        free(opt->v);
        memset(opt, 0, sizeof(*opt));
        set_error(error, "out of memory");
        return 0;
    }
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->count = count;
    opt->step = 0;
    return 1;
}

int nir_adam_step(
    nir_adam_t *opt, float *weights, const float *grad, size_t count
) {
    if (opt == NULL || opt->m == NULL || opt->v == NULL ||
        opt->count != count || count == 0 || weights == NULL ||
        grad == NULL || opt->step == ULONG_MAX) {
        return 0;
    }
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(weights[i]) || !isfinite(grad[i])) return 0;
    }
    opt->step++;
    double bc1 = 1.0 - pow((double)opt->beta1, (double)opt->step);
    double bc2 = 1.0 - pow((double)opt->beta2, (double)opt->step);
    for (size_t i = 0; i < count; i++) {
        double d = (double)grad[i] +
            (double)opt->weight_decay * (double)weights[i];
        double m = (double)opt->beta1 * (double)opt->m[i] +
            (1.0 - (double)opt->beta1) * d;
        double v = (double)opt->beta2 * (double)opt->v[i] +
            (1.0 - (double)opt->beta2) * d * d;
        opt->m[i] = (float)m;
        opt->v[i] = (float)v;
        double denom = sqrt(v / bc2) + (double)opt->eps;
        weights[i] -= (float)((double)opt->lr * (m / bc1) / denom);
    }
    return 1;
}

void nir_adam_free(nir_adam_t *opt) {
    if (opt == NULL) return;
    free(opt->m);
    free(opt->v);
    memset(opt, 0, sizeof(*opt));
}

int nir_stdp_init(
    nir_stdp_t *s, size_t in_features, size_t out_features,
    float tau_pre, float tau_post, float a_plus, float a_minus,
    float w_min, float w_max, char **error
) {
    if (error != NULL) *error = NULL;
    if (s == NULL) {
        set_error(error, "STDP output must not be null");
        return 0;
    }
    memset(s, 0, sizeof(*s));
    if (in_features == 0 || out_features == 0 ||
        !isfinite(tau_pre) || tau_pre <= 0.0f || !isfinite(tau_post) ||
        tau_post <= 0.0f || !isfinite(a_plus) || !isfinite(a_minus) ||
        !isfinite(w_min) || !isfinite(w_max) || w_min > w_max ||
        in_features > SIZE_MAX / out_features ||
        in_features * out_features > SIZE_MAX / sizeof(float)) {
        set_error(error, "invalid STDP parameters");
        return 0;
    }
    s->weight = (float *)calloc(
        in_features * out_features, sizeof(float)
    );
    s->pre_trace = (float *)calloc(in_features, sizeof(float));
    s->post_trace = (float *)calloc(out_features, sizeof(float));
    if (s->weight == NULL || s->pre_trace == NULL ||
        s->post_trace == NULL) {
        nir_stdp_free(s);
        set_error(error, "out of memory");
        return 0;
    }
    s->in_features = in_features;
    s->out_features = out_features;
    s->tau_pre = tau_pre;
    s->tau_post = tau_post;
    s->a_plus = a_plus;
    s->a_minus = a_minus;
    s->w_min = w_min;
    s->w_max = w_max;
    return 1;
}

int nir_stdp_set_weight(nir_stdp_t *s, const float *weight) {
    if (s == NULL || s->weight == NULL || weight == NULL) {
        return 0;
    }
    size_t n = s->in_features * s->out_features;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(weight[i])) return 0;
    }
    memcpy(s->weight, weight, n * sizeof(float));
    return 1;
}

int nir_stdp_step(
    nir_stdp_t *s, const float *pre_spikes, const float *post_spikes
) {
    if (s == NULL || s->weight == NULL || s->pre_trace == NULL ||
        s->post_trace == NULL || pre_spikes == NULL ||
        post_spikes == NULL) {
        return 0;
    }
    size_t in = s->in_features;
    size_t out = s->out_features;
    for (size_t i = 0; i < in; i++) {
        if (!isfinite(pre_spikes[i])) return 0;
    }
    for (size_t o = 0; o < out; o++) {
        if (!isfinite(post_spikes[o])) return 0;
    }
    /* Decay first: this step's spikes read the prior step's traces. */
    float decay_pre = (float)exp(-1.0 / (double)s->tau_pre);
    float decay_post = (float)exp(-1.0 / (double)s->tau_post);
    for (size_t i = 0; i < in; i++) s->pre_trace[i] *= decay_pre;
    for (size_t o = 0; o < out; o++) s->post_trace[o] *= decay_post;
    for (size_t i = 0; i < in; i++) {
        for (size_t o = 0; o < out; o++) {
            float w = s->weight[i * out + o] +
                s->a_plus * s->pre_trace[i] * post_spikes[o] -
                s->a_minus * pre_spikes[i] * s->post_trace[o];
            if (w < s->w_min) w = s->w_min;
            if (w > s->w_max) w = s->w_max;
            s->weight[i * out + o] = w;
        }
    }
    for (size_t i = 0; i < in; i++) s->pre_trace[i] += pre_spikes[i];
    for (size_t o = 0; o < out; o++) s->post_trace[o] += post_spikes[o];
    return 1;
}

void nir_stdp_free(nir_stdp_t *s) {
    if (s == NULL) return;
    free(s->weight);
    free(s->pre_trace);
    free(s->post_trace);
    memset(s, 0, sizeof(*s));
}
