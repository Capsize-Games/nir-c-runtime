#include "nir_c_runtime/driver.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/checkpoint.h"
#include "nir_c_runtime/data.h"
#include "nir_c_runtime/encode.h"
#include "nir_c_runtime/minibatch.h"
#include "nir_c_runtime/schedule.h"

static void set_error(char **error, const char *msg) {
    if (error == NULL || *error != NULL) {
        return;
    }
    size_t n = strlen(msg) + 1;
    *error = (char *)malloc(n);
    if (*error != NULL) {
        memcpy(*error, msg, n);
    }
}

static int node_weights(
    nir_graph_t *g, size_t idx, float **w, size_t *nw,
    float **b, size_t *nb
) {
    nir_node_t *node = &g->nodes[idx];
    *w = NULL;
    *nw = 0;
    *b = NULL;
    *nb = 0;
    switch (node->kind) {
        case NIR_NODE_AFFINE:
            *w = node->params.affine.weight;
            *nw = node->params.affine.out_features *
                node->params.affine.in_features;
            *b = node->params.affine.bias;
            if (*b != NULL) {
                *nb = node->params.affine.out_features;
            }
            return 1;
        case NIR_NODE_LINEAR:
            *w = node->params.linear.weight;
            *nw = node->params.linear.out_features *
                node->params.linear.in_features;
            return 1;
        case NIR_NODE_CONV2D:
            *w = node->params.conv2d.weight;
            *nw = node->params.conv2d.out_channels *
                node->params.conv2d.in_channels *
                node->params.conv2d.kernel_h *
                node->params.conv2d.kernel_w;
            *b = node->params.conv2d.bias;
            if (*b != NULL) {
                *nb = node->params.conv2d.out_channels;
            }
            return 1;
        case NIR_NODE_CONV1D:
            *w = node->params.conv1d.weight;
            *nw = node->params.conv1d.out_channels *
                node->params.conv1d.in_channels *
                node->params.conv1d.kernel;
            *b = node->params.conv1d.bias;
            if (*b != NULL) {
                *nb = node->params.conv1d.out_channels;
            }
            return 1;
        default:
            return 0;
    }
}

static int opt_hyper_ok(
    const nir_driver_config_t *c, char **error
) {
    if (!isfinite(c->lr) || c->lr < 0.0f ||
        !isfinite(c->weight_decay) || c->weight_decay < 0.0f) {
        set_error(error, "driver needs lr and decay >= 0");
        return 0;
    }
    if (c->opt_kind == NIR_DRIVER_OPT_SGD) {
        if (!isfinite(c->momentum) || c->momentum < 0.0f) {
            set_error(error, "driver SGD needs momentum >= 0");
            return 0;
        }
        return 1;
    }
    if (c->opt_kind != NIR_DRIVER_OPT_ADAM) {
        set_error(error, "driver needs SGD or Adam");
        return 0;
    }
    if (!isfinite(c->beta1) || c->beta1 < 0.0f ||
        c->beta1 >= 1.0f || !isfinite(c->beta2) ||
        c->beta2 < 0.0f || c->beta2 >= 1.0f ||
        !isfinite(c->eps) || c->eps <= 0.0f) {
        set_error(error, "driver Adam needs betas, eps");
        return 0;
    }
    return 1;
}

static int sched_ok(
    const nir_driver_config_t *c, char **error
) {
    switch (c->sched) {
        case NIR_DRIVER_SCHED_NONE:
            return 1;
        case NIR_DRIVER_SCHED_STEP:
            if (!isfinite(c->sched_gamma) ||
                c->sched_gamma < 0.0f ||
                c->sched_step_size == 0) {
                set_error(error, "driver step schedule invalid");
                return 0;
            }
            return 1;
        case NIR_DRIVER_SCHED_COSINE:
            if (!isfinite(c->sched_eta_min) ||
                c->sched_eta_min < 0.0f ||
                c->sched_t_max == 0) {
                set_error(error, "driver cosine schedule invalid");
                return 0;
            }
            return 1;
        case NIR_DRIVER_SCHED_EXP:
            if (!isfinite(c->sched_gamma) ||
                c->sched_gamma < 0.0f) {
                set_error(error, "driver exp schedule invalid");
                return 0;
            }
            return 1;
        default:
            set_error(error, "driver schedule unknown");
            return 0;
    }
}

static int config_ok(
    const nir_driver_config_t *c, char **error
) {
    if (c == NULL || c->num_steps == 0 ||
        c->batch_size == 0) {
        set_error(error, "driver needs steps and batch");
        return 0;
    }
    if (c->encode != NIR_DRIVER_ENCODE_REPEAT &&
        c->encode != NIR_DRIVER_ENCODE_RATE) {
        set_error(error, "driver encode must be repeat/rate");
        return 0;
    }
    if (c->encode == NIR_DRIVER_ENCODE_RATE &&
        !isfinite(c->encode_gain)) {
        set_error(error, "driver rate encode needs a gain");
        return 0;
    }
    float dummy = 0.0f;
    if (!nir_surrogate_grad(c->surrogate, c->slope, 0.0f,
            &dummy)) {
        set_error(error, "driver surrogate/slope invalid");
        return 0;
    }
    return opt_hyper_ok(c, error) && sched_ok(c, error);
}

static void free_nodes(
    nir_driver_node_t *nodes, size_t count
) {
    if (nodes == NULL) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        free(nodes[i].accum_w);
        free(nodes[i].accum_b);
        if (nodes[i].sgd_w_live) {
            nir_sgd_free(&nodes[i].sgd_w);
        }
        if (nodes[i].sgd_b_live) {
            nir_sgd_free(&nodes[i].sgd_b);
        }
        if (nodes[i].adam_w_live) {
            nir_adam_free(&nodes[i].adam_w);
        }
        if (nodes[i].adam_b_live) {
            nir_adam_free(&nodes[i].adam_b);
        }
    }
    free(nodes);
}

static int init_one_sgd(
    nir_driver_t *d, size_t idx, char **error
) {
    nir_driver_node_t *n = &d->nodes[idx];
    const nir_driver_config_t *c = &d->config;
    if (!nir_sgd_init(&n->sgd_w, n->w_count, c->lr,
            c->momentum, c->weight_decay, error)) {
        return 0;
    }
    n->sgd_w_live = 1;
    if (!n->has_b) {
        return 1;
    }
    if (!nir_sgd_init(&n->sgd_b, n->b_count, c->lr,
            c->momentum, c->weight_decay, error)) {
        return 0;
    }
    n->sgd_b_live = 1;
    return 1;
}

static int init_one_adam(
    nir_driver_t *d, size_t idx, char **error
) {
    nir_driver_node_t *n = &d->nodes[idx];
    const nir_driver_config_t *c = &d->config;
    if (!nir_adam_init(&n->adam_w, n->w_count, c->lr,
            c->beta1, c->beta2, c->eps, c->weight_decay,
            error)) {
        return 0;
    }
    n->adam_w_live = 1;
    if (!n->has_b) {
        return 1;
    }
    if (!nir_adam_init(&n->adam_b, n->b_count, c->lr,
            c->beta1, c->beta2, c->eps, c->weight_decay,
            error)) {
        return 0;
    }
    n->adam_b_live = 1;
    return 1;
}

static int init_nodes(nir_driver_t *d, char **error) {
    size_t n = d->graph->node_count;
    d->nodes = (nir_driver_node_t *)calloc(
        n == 0 ? 1 : n, sizeof(nir_driver_node_t));
    if (d->nodes == NULL) {
        set_error(error, "out of memory");
        return 0;
    }
    d->node_count = n;
    for (size_t i = 0; i < n; i++) {
        float *w = NULL;
        float *b = NULL;
        size_t nw = 0;
        size_t nb = 0;
        if (!node_weights(d->graph, i, &w, &nw, &b,
                &nb)) {
            continue;
        }
        nir_driver_node_t *node = &d->nodes[i];
        node->has_w = 1;
        node->w_count = nw;
        node->accum_w = (float *)calloc(
            nw == 0 ? 1 : nw, sizeof(float));
        if (node->accum_w == NULL) {
            set_error(error, "out of memory");
            return 0;
        }
        if (b != NULL && nb > 0) {
            node->has_b = 1;
            node->b_count = nb;
            node->accum_b = (float *)calloc(nb,
                sizeof(float));
            if (node->accum_b == NULL) {
                set_error(error, "out of memory");
                return 0;
            }
        }
        int ok = d->config.opt_kind == NIR_DRIVER_OPT_SGD
            ? init_one_sgd(d, i, error)
            : init_one_adam(d, i, error);
        if (!ok) {
            return 0;
        }
    }
    return 1;
}

int nir_driver_init(
    nir_driver_t *driver, nir_graph_t *graph,
    const nir_driver_config_t *config, char **error
) {
    if (driver == NULL || graph == NULL) {
        set_error(error, "driver needs a graph");
        return 0;
    }
    memset(driver, 0, sizeof(*driver));
    if (!config_ok(config, error)) {
        return 0;
    }
    driver->graph = graph;
    driver->config = *config;
    driver->step = 0;
    if (!nir_tape_init(&driver->tape, graph,
            config->num_steps, config->surrogate,
            config->slope, error)) {
        return 0;
    }
    if (!init_nodes(driver, error)) {
        nir_tape_free(&driver->tape);
        free_nodes(driver->nodes, driver->node_count);
        memset(driver, 0, sizeof(*driver));
        return 0;
    }
    nir_rng_seed(&driver->shuffle_rng, config->seed);
    nir_rng_seed(&driver->encode_rng, config->seed + 1);
    driver->live = 1;
    return 1;
}

void nir_driver_free(nir_driver_t *driver) {
    if (driver == NULL) {
        return;
    }
    nir_tape_free(&driver->tape);
    free_nodes(driver->nodes, driver->node_count);
    memset(driver, 0, sizeof(*driver));
}

unsigned long nir_driver_step(const nir_driver_t *driver) {
    return driver == NULL ? 0 : driver->step;
}
