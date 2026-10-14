#include "nir_c_runtime/driver.h"

#include <stdint.h>
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

static void free_frames(
    nir_tensor_t *frames, size_t count
) {
    if (frames == NULL) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        nir_tensor_free(&frames[i]);
    }
}

static int frames_from_floats(
    const float *flat, size_t n, size_t rows, size_t cols,
    nir_tensor_t *frames, size_t steps, char **error
) {
    size_t shape[3] = {1, rows, cols};
    for (size_t t = 0; t < steps; t++) {
        frames[t] = nir_tensor_alloc(shape, 3);
        if (frames[t].data == NULL) {
            free_frames(frames, t);
            set_error(error, "out of memory");
            return 0;
        }
        memcpy(frames[t].data, flat + t * n,
            n * sizeof(float));
    }
    return 1;
}

static int frames_from_spikes(
    const uint8_t *spikes, size_t n, size_t rows,
    size_t cols, nir_tensor_t *frames, size_t steps,
    char **error
) {
    size_t shape[3] = {1, rows, cols};
    for (size_t t = 0; t < steps; t++) {
        frames[t] = nir_tensor_alloc(shape, 3);
        if (frames[t].data == NULL) {
            free_frames(frames, t);
            set_error(error, "out of memory");
            return 0;
        }
        for (size_t i = 0; i < n; i++) {
            frames[t].data[i] =
                spikes[t * n + i] ? 1.0f : 0.0f;
        }
    }
    return 1;
}

static int encode_repeat_frames(
    nir_driver_t *d, const float *norm, size_t n,
    size_t rows, size_t cols, nir_tensor_t *frames,
    char **error
) {
    size_t steps = d->config.num_steps;
    float *flat = (float *)malloc(
        steps * n * sizeof(float));
    if (flat == NULL) {
        set_error(error, "out of memory");
        return 0;
    }
    int ok = nir_encode_repeat(norm, n, steps, flat,
        error);
    if (ok) {
        ok = frames_from_floats(flat, n, rows, cols,
            frames, steps, error);
    }
    free(flat);
    return ok;
}

static int encode_rate_frames(
    nir_driver_t *d, const float *norm, size_t n,
    size_t rows, size_t cols, nir_tensor_t *frames,
    char **error
) {
    size_t steps = d->config.num_steps;
    uint8_t *spikes = (uint8_t *)malloc(steps * n);
    if (spikes == NULL) {
        set_error(error, "out of memory");
        return 0;
    }
    int ok = nir_encode_rate(norm, n, steps,
        d->config.encode_gain, &d->encode_rng, spikes,
        error);
    if (ok) {
        ok = frames_from_spikes(spikes, n, rows, cols,
            frames, steps, error);
    }
    free(spikes);
    return ok;
}

static int encode_sample(
    nir_driver_t *d, const uint8_t *pixels, size_t rows,
    size_t cols, size_t index, nir_tensor_t *frames,
    char **error
) {
    size_t n = rows * cols;
    const uint8_t *raw = pixels + index * n;
    float *norm = (float *)malloc(n * sizeof(float));
    if (norm == NULL) {
        set_error(error, "out of memory");
        return 0;
    }
    nir_data_normalize(raw, norm, n);
    int ok = 0;
    if (d->config.encode == NIR_DRIVER_ENCODE_REPEAT) {
        ok = encode_repeat_frames(d, norm, n, rows,
            cols, frames, error);
    } else {
        ok = encode_rate_frames(d, norm, n, rows, cols,
            frames, error);
    }
    free(norm);
    return ok;
}

static size_t argmax(const nir_tensor_t *t) {
    size_t best = 0;
    for (size_t i = 1; i < t->size; i++) {
        if (t->data[i] > t->data[best]) {
            best = i;
        }
    }
    return best;
}

static int accumulate_tape_grads(
    nir_driver_t *d, char **error
) {
    for (size_t i = 0; i < d->node_count; i++) {
        nir_driver_node_t *node = &d->nodes[i];
        if (!node->has_w) {
            continue;
        }
        const nir_tensor_t *gw = nir_tape_weight_grad(
            &d->tape, i);
        if (gw == NULL || gw->size != node->w_count) {
            set_error(error, "driver missing weight grad");
            return 0;
        }
        if (!nir_minibatch_accumulate(node->accum_w,
                gw->data, node->w_count)) {
            set_error(error, "driver grad not finite");
            return 0;
        }
        if (!node->has_b) {
            continue;
        }
        const nir_tensor_t *gb = nir_tape_bias_grad(
            &d->tape, i);
        if (gb == NULL || gb->size != node->b_count) {
            set_error(error, "driver missing bias grad");
            return 0;
        }
        if (!nir_minibatch_accumulate(node->accum_b,
                gb->data, node->b_count)) {
            set_error(error, "driver grad not finite");
            return 0;
        }
    }
    return 1;
}

static int forward_loss(
    nir_driver_t *d, nir_tensor_t *frames, size_t label,
    nir_tensor_t *readout, nir_tensor_t *grad,
    float *loss, char **error
) {
    memset(readout, 0, sizeof(*readout));
    if (!nir_tape_forward(&d->tape, frames, readout,
            error)) {
        return 0;
    }
    size_t shape[1] = {readout->size};
    *grad = nir_tensor_alloc(shape, 1);
    if (grad->data == NULL) {
        nir_tensor_free(readout);
        set_error(error, "out of memory");
        return 0;
    }
    if (!nir_cross_entropy(readout, label, loss, grad)) {
        nir_tensor_free(readout);
        nir_tensor_free(grad);
        set_error(error, "driver loss failed");
        return 0;
    }
    return 1;
}

static int train_one(
    nir_driver_t *d, const uint8_t *pixels,
    size_t sample, size_t rows, size_t cols,
    size_t label, float *loss, int *correct,
    char **error
) {
    size_t steps = d->config.num_steps;
    nir_tensor_t *frames = (nir_tensor_t *)calloc(
        steps, sizeof(nir_tensor_t));
    if (frames == NULL) {
        set_error(error, "out of memory");
        return 0;
    }
    int ok = 0;
    nir_tensor_t readout;
    nir_tensor_t grad;
    memset(&grad, 0, sizeof(grad));
    if (!encode_sample(d, pixels, rows, cols, sample,
            frames, error)) {
        goto done;
    }
    if (!forward_loss(d, frames, label, &readout, &grad,
            loss, error)) {
        goto done;
    }
    *correct = argmax(&readout) == label;
    if (!nir_tape_backward(&d->tape, &grad, error)) {
        nir_tensor_free(&readout);
        nir_tensor_free(&grad);
        goto done;
    }
    ok = accumulate_tape_grads(d, error);
    nir_tensor_free(&readout);
    nir_tensor_free(&grad);
done:
    free_frames(frames, steps);
    free(frames);
    return ok;
}

static void zero_accums(nir_driver_t *d) {
    for (size_t i = 0; i < d->node_count; i++) {
        nir_driver_node_t *n = &d->nodes[i];
        if (n->has_w) {
            nir_minibatch_zero(n->accum_w, n->w_count);
        }
        if (n->has_b) {
            nir_minibatch_zero(n->accum_b, n->b_count);
        }
    }
}

static int scale_accums(
    nir_driver_t *d, size_t batch, char **error
) {
    float scale = 1.0f / (float)batch;
    for (size_t i = 0; i < d->node_count; i++) {
        nir_driver_node_t *n = &d->nodes[i];
        if (n->has_w &&
            !nir_minibatch_scale(n->accum_w, n->w_count,
                scale)) {
            set_error(error, "driver scale failed");
            return 0;
        }
        if (n->has_b &&
            !nir_minibatch_scale(n->accum_b, n->b_count,
                scale)) {
            set_error(error, "driver scale failed");
            return 0;
        }
    }
    return 1;
}

static int step_node(
    nir_driver_t *d, size_t idx, char **error
) {
    nir_driver_node_t *n = &d->nodes[idx];
    if (!n->has_w) {
        return 1;
    }
    const char *name = d->graph->nodes[idx].name;
    const float *cw = NULL;
    const float *cb = NULL;
    size_t nw = 0;
    size_t nb = 0;
    if (!nir_graph_get_weights(d->graph, name, &cw, &nw,
            &cb, &nb)) {
        set_error(error, "driver weights missing");
        return 0;
    }
    float *w = (float *)cw;
    float *b = (float *)cb;
    int ok = 1;
    if (d->config.opt_kind == NIR_DRIVER_OPT_SGD) {
        ok = nir_sgd_step(&n->sgd_w, w, n->accum_w, nw);
        if (ok && n->has_b) {
            ok = nir_sgd_step(&n->sgd_b, b, n->accum_b,
                nb);
        }
    } else {
        ok = nir_adam_step(&n->adam_w, w, n->accum_w,
            nw);
        if (ok && n->has_b) {
            ok = nir_adam_step(&n->adam_b, b, n->accum_b,
                nb);
        }
    }
    if (!ok) {
        set_error(error, "driver optimizer step failed");
    }
    return ok;
}

static int current_lr(
    nir_driver_t *d, float *lr, char **error
) {
    const nir_driver_config_t *c = &d->config;
    int ok = 1;
    switch (c->sched) {
        case NIR_DRIVER_SCHED_NONE:
            *lr = c->lr;
            break;
        case NIR_DRIVER_SCHED_STEP:
            ok = nir_lr_step_decay(c->lr, c->sched_gamma,
                c->sched_step_size, d->step, lr);
            break;
        case NIR_DRIVER_SCHED_COSINE:
            ok = nir_lr_cosine(c->lr, c->sched_eta_min,
                c->sched_t_max, d->step, lr);
            break;
        default:
            ok = nir_lr_exponential(c->lr,
                c->sched_gamma, d->step, lr);
            break;
    }
    if (!ok) {
        set_error(error, "driver schedule failed");
    }
    return ok;
}

static void write_lr(nir_driver_t *d, float lr) {
    for (size_t i = 0; i < d->node_count; i++) {
        nir_driver_node_t *n = &d->nodes[i];
        if (n->sgd_w_live) {
            n->sgd_w.lr = lr;
        }
        if (n->sgd_b_live) {
            n->sgd_b.lr = lr;
        }
        if (n->adam_w_live) {
            n->adam_w.lr = lr;
        }
        if (n->adam_b_live) {
            n->adam_b.lr = lr;
        }
    }
}

static int schedule_and_step(
    nir_driver_t *d, char **error
) {
    float lr = 0.0f;
    if (!current_lr(d, &lr, error)) {
        return 0;
    }
    write_lr(d, lr);
    for (size_t i = 0; i < d->node_count; i++) {
        if (!step_node(d, i, error)) {
            return 0;
        }
    }
    d->step++;
    return 1;
}

static int epoch_args_ok(
    nir_driver_t *d, const uint8_t *px,
    const uint8_t *lb, size_t count, size_t rows,
    size_t cols, nir_driver_stats_t *st, char **error
) {
    if (d == NULL || !d->live || px == NULL ||
        lb == NULL || count == 0 || rows == 0 ||
        cols == 0 || st == NULL) {
        set_error(error, "driver epoch needs data");
        return 0;
    }
    return 1;
}

static int run_batch(
    nir_driver_t *d, const uint8_t *px,
    const uint8_t *lb, const size_t *order,
    size_t begin, size_t end, size_t rows, size_t cols,
    double *loss_sum, size_t *correct, char **error
) {
    zero_accums(d);
    for (size_t k = begin; k < end; k++) {
        size_t sample = order[k];
        float loss = 0.0f;
        int hit = 0;
        if (!train_one(d, px, sample, rows, cols,
                (size_t)lb[sample], &loss, &hit,
                error)) {
            return 0;
        }
        *loss_sum += (double)loss;
        if (hit) {
            (*correct)++;
        }
    }
    if (!scale_accums(d, end - begin, error)) {
        return 0;
    }
    return schedule_and_step(d, error);
}

int nir_driver_train_epoch(
    nir_driver_t *driver, const uint8_t *pixels,
    const uint8_t *labels, size_t count, size_t rows,
    size_t cols, nir_driver_stats_t *stats, char **error
) {
    if (!epoch_args_ok(driver, pixels, labels, count,
            rows, cols, stats, error)) {
        return 0;
    }
    size_t *order = (size_t *)malloc(
        count * sizeof(size_t));
    if (order == NULL) {
        set_error(error, "out of memory");
        return 0;
    }
    nir_sampler_arange(order, count);
    if (!nir_sampler_shuffle(&driver->shuffle_rng, order,
            count)) {
        free(order);
        set_error(error, "driver shuffle failed");
        return 0;
    }
    double loss_sum = 0.0;
    size_t correct = 0;
    size_t batch = driver->config.batch_size;
    int ok = 1;
    for (size_t s = 0; s < count; s += batch) {
        size_t e = s + batch;
        if (e > count) {
            e = count;
        }
        if (!run_batch(driver, pixels, labels, order, s,
                e, rows, cols, &loss_sum, &correct,
                error)) {
            ok = 0;
            break;
        }
    }
    free(order);
    if (!ok) {
        return 0;
    }
    stats->loss = (float)(loss_sum / (double)count);
    stats->accuracy = (float)correct / (float)count;
    return 1;
}

static int eval_one(
    nir_driver_t *d, const uint8_t *pixels,
    size_t sample, size_t rows, size_t cols,
    size_t label, float *loss, int *correct,
    char **error
) {
    size_t steps = d->config.num_steps;
    nir_tensor_t *frames = (nir_tensor_t *)calloc(
        steps, sizeof(nir_tensor_t));
    if (frames == NULL) {
        set_error(error, "out of memory");
        return 0;
    }
    int ok = 0;
    nir_tensor_t readout;
    nir_tensor_t grad;
    memset(&grad, 0, sizeof(grad));
    if (!encode_sample(d, pixels, rows, cols, sample,
            frames, error)) {
        goto done;
    }
    if (!forward_loss(d, frames, label, &readout, &grad,
            loss, error)) {
        goto done;
    }
    *correct = argmax(&readout) == label;
    ok = 1;
    nir_tensor_free(&readout);
    nir_tensor_free(&grad);
done:
    free_frames(frames, steps);
    free(frames);
    return ok;
}

int nir_driver_eval(
    nir_driver_t *driver, const uint8_t *pixels,
    const uint8_t *labels, size_t count, size_t rows,
    size_t cols, nir_driver_stats_t *stats, char **error
) {
    if (!epoch_args_ok(driver, pixels, labels, count,
            rows, cols, stats, error)) {
        return 0;
    }
    double loss_sum = 0.0;
    size_t correct = 0;
    for (size_t i = 0; i < count; i++) {
        float loss = 0.0f;
        int hit = 0;
        if (!eval_one(driver, pixels, i, rows, cols,
                (size_t)labels[i], &loss, &hit,
                error)) {
            return 0;
        }
        loss_sum += (double)loss;
        if (hit) {
            correct++;
        }
    }
    stats->loss = (float)(loss_sum / (double)count);
    stats->accuracy = (float)correct / (float)count;
    return 1;
}

static size_t count_trained(const nir_driver_t *d) {
    size_t n = 0;
    for (size_t i = 0; i < d->node_count; i++) {
        if (d->nodes[i].has_w) {
            n++;
        }
    }
    return n;
}

static void fill_opt(
    nir_driver_t *d, size_t idx, nir_checkpoint_opt_t *o
) {
    nir_driver_node_t *n = &d->nodes[idx];
    memset(o, 0, sizeof(*o));
    o->node = d->graph->nodes[idx].name;
    if (d->config.opt_kind == NIR_DRIVER_OPT_SGD) {
        o->kind = NIR_CHECKPOINT_OPT_SGD;
        o->sgd_weight = &n->sgd_w;
        o->sgd_bias = n->has_b ? &n->sgd_b : NULL;
    } else {
        o->kind = NIR_CHECKPOINT_OPT_ADAM;
        o->adam_weight = &n->adam_w;
        o->adam_bias = n->has_b ? &n->adam_b : NULL;
    }
}

static nir_checkpoint_opt_t *build_opts(
    nir_driver_t *d, size_t *count, char **error
) {
    size_t n = count_trained(d);
    *count = n;
    if (n == 0) {
        return NULL;
    }
    nir_checkpoint_opt_t *opts =
        (nir_checkpoint_opt_t *)calloc(n,
            sizeof(nir_checkpoint_opt_t));
    if (opts == NULL) {
        set_error(error, "out of memory");
        return NULL;
    }
    size_t k = 0;
    for (size_t i = 0; i < d->node_count; i++) {
        if (d->nodes[i].has_w) {
            fill_opt(d, i, &opts[k++]);
        }
    }
    return opts;
}

int nir_driver_save(
    nir_driver_t *driver, const char *path,
    const char *topology, char **error
) {
    if (driver == NULL || !driver->live ||
        path == NULL) {
        set_error(error, "driver save needs a path");
        return 0;
    }
    size_t count = 0;
    nir_checkpoint_opt_t *opts = build_opts(driver,
        &count, error);
    if (count > 0 && opts == NULL) {
        return 0;
    }
    nir_checkpoint_meta_t meta;
    memset(&meta, 0, sizeof(meta));
    meta.topology = (char *)topology;
    meta.seed = driver->config.seed;
    meta.step = driver->step;
    int ok = nir_checkpoint_save_with_opt(path,
        driver->graph, &meta, opts, count, error);
    free(opts);
    return ok;
}

int nir_driver_load(
    nir_driver_t *driver, const char *path, char **error
) {
    if (driver == NULL || !driver->live ||
        path == NULL) {
        set_error(error, "driver load needs a path");
        return 0;
    }
    size_t count = 0;
    nir_checkpoint_opt_t *opts = build_opts(driver,
        &count, error);
    if (count > 0 && opts == NULL) {
        return 0;
    }
    nir_checkpoint_meta_t meta;
    memset(&meta, 0, sizeof(meta));
    int ok = nir_checkpoint_load_with_opt(path,
        driver->graph, &meta, opts, count, error);
    if (ok) {
        driver->step = meta.step;
    }
    nir_checkpoint_meta_free(&meta);
    free(opts);
    return ok;
}
