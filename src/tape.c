#include "nir_c_runtime/tape.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The message is always data: copied verbatim, never interpreted
 * as a printf format string. */
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

/* Join a caller-supplied name into a message by concatenation, so the
 * name can never act as a printf conversion. */
static void set_error_arg(
    char **error, const char *prefix, const char *arg, const char *suffix
) {
    if (error == NULL || *error != NULL) {
        return;
    }
    if (arg == NULL) {
        arg = "";
    }
    size_t prefix_length = strlen(prefix);
    size_t arg_length = strlen(arg);
    size_t suffix_length = strlen(suffix);
    *error = (char *)malloc(prefix_length + arg_length + suffix_length + 1);
    if (*error != NULL) {
        memcpy(*error, prefix, prefix_length);
        memcpy(*error + prefix_length, arg, arg_length);
        memcpy(
            *error + prefix_length + arg_length, suffix, suffix_length + 1
        );
    }
}

static int checked_mul(size_t a, size_t b, size_t *out) {
    if (b != 0 && a > SIZE_MAX / b) {
        return 0;
    }
    *out = a * b;
    return 1;
}

/* A Delay source feeds its consumers from the previous step's
 * buffer, which only exists from step 1 on: every tape forward
 * starts from unprimed temporal state. */
static int source_contributes(
    const nir_graph_t *graph, size_t source, size_t step
) {
    return graph->nodes[source].kind != NIR_NODE_DELAY || step > 0;
}

static int tensor_valid(const nir_tensor_t *tensor) {
    if (tensor == NULL || tensor->data == NULL ||
        tensor->ndim > NIR_TENSOR_MAX_DIMS) {
        return 0;
    }
    size_t count = 1;
    for (size_t i = 0; i < tensor->ndim; i++) {
        if (tensor->shape[i] != 0 && count > SIZE_MAX / tensor->shape[i]) {
            return 0;
        }
        count *= tensor->shape[i];
    }
    return count == tensor->size && tensor->capacity >= tensor->size;
}

static int tensor_same_shape(
    const nir_tensor_t *a, const nir_tensor_t *b
) {
    if (a->ndim != b->ndim) {
        return 0;
    }
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return 0;
        }
    }
    return 1;
}

static void free_slots(nir_tensor_t *slots, size_t count) {
    if (slots == NULL) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        nir_tensor_free(&slots[i]);
    }
}

/* Retained-slot count. Safe: num_steps is only stored after the
 * checked multiplication in nir_tape_init succeeds. */
static size_t tape_slots(const nir_tape_t *tape) {
    if (tape->graph == NULL) {
        return 0;
    }
    return tape->num_steps * tape->graph->node_count;
}

static void clear_retained(nir_tape_t *tape) {
    size_t slots = tape_slots(tape);
    free_slots(tape->inputs, slots);
    free_slots(tape->values, slots);
    free_slots(tape->stored, slots);
    free_slots(tape->syn, slots);
    free_slots(tape->grads, slots);
    free_slots(tape->state_grads, slots);
    tape->forwarded = 0;
    tape->have_grads = 0;
}

/* Parameter-gradient element counts for one node: weight_count
 * is 0 for unweighted kinds, bias_count is 0 for unbiased
 * ones. Fails only on arithmetic overflow. */
static int grad_counts(
    const nir_node_t *node, size_t *weight_count, size_t *bias_count
) {
    size_t w = 0, b = 0;
    switch (node->kind) {
        case NIR_NODE_AFFINE:
            if (!checked_mul(
                    node->params.affine.out_features,
                    node->params.affine.in_features, &w)) {
                return 0;
            }
            if (node->params.affine.bias != NULL) {
                b = node->params.affine.out_features;
            }
            break;
        case NIR_NODE_LINEAR:
            if (!checked_mul(
                    node->params.linear.out_features,
                    node->params.linear.in_features, &w)) {
                return 0;
            }
            break;
        case NIR_NODE_CONV2D: {
            const nir_conv2d_params_t *p = &node->params.conv2d;
            w = p->out_channels;
            if (!checked_mul(w, p->in_channels, &w) ||
                !checked_mul(w, p->kernel_h, &w) ||
                !checked_mul(w, p->kernel_w, &w)) {
                return 0;
            }
            if (p->bias != NULL) {
                b = p->out_channels;
            }
            break;
        }
        case NIR_NODE_CONV1D: {
            const nir_conv1d_params_t *p = &node->params.conv1d;
            w = p->out_channels;
            if (!checked_mul(w, p->in_channels, &w) ||
                !checked_mul(w, p->kernel, &w)) {
                return 0;
            }
            if (p->bias != NULL) {
                b = p->out_channels;
            }
            break;
        }
        default:
            break;
    }
    *weight_count = w;
    *bias_count = b;
    return 1;
}

static int alloc_param_grads(nir_tape_t *tape, char **error) {
    size_t n = tape->graph->node_count;
    for (size_t i = 0; i < n; i++) {
        size_t w = 0, b = 0;
        if (!grad_counts(&tape->graph->nodes[i], &w, &b)) {
            set_error(error, "tape parameter block is too large");
            return 0;
        }
        if (w > 0) {
            size_t shape[1] = {w};
            tape->grad_weight[i] = nir_tensor_alloc(shape, 1);
            if (tape->grad_weight[i].data == NULL) {
                set_error(error, "out of memory");
                return 0;
            }
        }
        if (b > 0) {
            size_t shape[1] = {b};
            tape->grad_bias[i] = nir_tensor_alloc(shape, 1);
            if (tape->grad_bias[i].data == NULL) {
                set_error(error, "out of memory");
                return 0;
            }
        }
    }
    return 1;
}

int nir_tape_init(
    nir_tape_t *tape, const nir_graph_t *graph, size_t num_steps,
    nir_surrogate_kind_t surrogate, float slope, char **error
) {
    float probe;
    if (tape == NULL) {
        set_error(error, "invalid tape arguments");
        return 0;
    }
    memset(tape, 0, sizeof(*tape));
    if (graph == NULL || num_steps == 0 ||
        !nir_surrogate_grad(surrogate, slope, 0.0f, &probe)) {
        set_error(error, "invalid tape arguments");
        return 0;
    }
    /* Scheduler reuse: this validates the graph exactly as
     * inference would (including the Delay-breaking cycle
     * check) and computes the order the tape walks. */
    if (!nir_runtime_init(&tape->sched, graph, error)) {
        memset(tape, 0, sizeof(*tape));
        return 0;
    }
    size_t slots = 0;
    if (!checked_mul(num_steps, graph->node_count, &slots)) {
        nir_tape_free(tape);
        set_error(error, "tape run is too large");
        return 0;
    }
    tape->graph = graph;
    tape->num_steps = num_steps;
    tape->surrogate = surrogate;
    tape->slope = slope;
    size_t n = graph->node_count;
    tape->inputs = (nir_tensor_t *)calloc(
        slots == 0 ? 1 : slots, sizeof(nir_tensor_t)
    );
    tape->values = (nir_tensor_t *)calloc(
        slots == 0 ? 1 : slots, sizeof(nir_tensor_t)
    );
    tape->stored = (nir_tensor_t *)calloc(
        slots == 0 ? 1 : slots, sizeof(nir_tensor_t)
    );
    tape->syn = (nir_tensor_t *)calloc(
        slots == 0 ? 1 : slots, sizeof(nir_tensor_t)
    );
    tape->grads = (nir_tensor_t *)calloc(
        slots == 0 ? 1 : slots, sizeof(nir_tensor_t)
    );
    tape->state_grads = (nir_tensor_t *)calloc(
        slots == 0 ? 1 : slots, sizeof(nir_tensor_t)
    );
    tape->grad_weight = (nir_tensor_t *)calloc(
        n == 0 ? 1 : n, sizeof(nir_tensor_t)
    );
    tape->grad_bias = (nir_tensor_t *)calloc(
        n == 0 ? 1 : n, sizeof(nir_tensor_t)
    );
    tape->grad_scale = (float *)calloc(n == 0 ? 1 : n, sizeof(float));
    if (tape->inputs == NULL || tape->values == NULL ||
        tape->stored == NULL || tape->syn == NULL ||
        tape->grads == NULL || tape->state_grads == NULL ||
        tape->grad_weight == NULL || tape->grad_bias == NULL ||
        tape->grad_scale == NULL) {
        nir_tape_free(tape);
        set_error(error, "out of memory");
        return 0;
    }
    if (!alloc_param_grads(tape, error)) {
        nir_tape_free(tape);
        return 0;
    }
    return 1;
}

void nir_tape_free(nir_tape_t *tape) {
    if (tape == NULL) {
        return;
    }
    clear_retained(tape);
    size_t n = tape->graph != NULL ? tape->graph->node_count : 0;
    free_slots(tape->grad_weight, n);
    free_slots(tape->grad_bias, n);
    free(tape->inputs);
    free(tape->values);
    free(tape->stored);
    free(tape->syn);
    free(tape->grads);
    free(tape->state_grads);
    free(tape->grad_weight);
    free(tape->grad_bias);
    free(tape->grad_scale);
    nir_runtime_free(&tape->sched);
    memset(tape, 0, sizeof(*tape));
}

/* Sum the retained step-`step` values feeding `target`, mirroring
 * runtime.c's sum_incoming: an unprimed Delay source (any Delay
 * at step 0) contributes nothing, and a target with no
 * contributing source sums to a single-element zero. */
static int sum_incoming(
    nir_tape_t *tape, size_t step, size_t target, nir_tensor_t *out,
    char **error
) {
    const nir_graph_t *graph = tape->graph;
    size_t n = graph->node_count;
    int have_any = 0, found_edge = 0;
    memset(out, 0, sizeof(*out));
    for (size_t e = 0; e < graph->edge_count; e++) {
        if (graph->edges[e].target_index != target) {
            continue;
        }
        found_edge = 1;
        size_t src = graph->edges[e].source_index;
        if (!source_contributes(graph, src, step)) {
            continue;
        }
        const nir_tensor_t *value = &tape->values[step * n + src];
        if (!have_any) {
            *out = nir_tensor_clone(value);
            if (out->data == NULL) {
                set_error(error, "out of memory or invalid node tensor");
                return 0;
            }
            have_any = 1;
        } else if (!nir_tensor_add_inplace(out, value)) {
            nir_tensor_free(out);
            set_error(error, "shape mismatch summing node inputs");
            return 0;
        }
    }
    if (!found_edge) {
        set_error_arg(error, "node ", graph->nodes[target].name,
            " received no inputs");
        return 0;
    }
    if (!have_any) {
        size_t one[1] = {1};
        *out = nir_tensor_alloc(one, 1); /* zero-initialized scalar */
        if (out->data == NULL) {
            set_error(error, "out of memory");
            return 0;
        }
    }
    return 1;
}

static nir_tensor_t alloc_like(const nir_tensor_t *input) {
    if (input == NULL) {
        return (nir_tensor_t){0};
    }
    return nir_tensor_alloc(input->shape, input->ndim);
}

/* The previous step's retained membrane for a stateful node, or
 * NULL on the first step (treated as zeros, matching the
 * runtime's unprimed state). */
static const nir_tensor_t *prev_state(
    nir_tape_t *tape, size_t step, size_t index
) {
    if (step == 0) {
        return NULL;
    }
    size_t n = tape->graph->node_count;
    return &tape->stored[(step - 1) * n + index];
}

static int apply_li(
    nir_tape_t *tape, size_t step, size_t index,
    const nir_tensor_t *input, char **error
) {
    const nir_node_t *node = &tape->graph->nodes[index];
    size_t n = tape->graph->node_count;
    nir_tensor_t membrane = nir_op_li(
        &node->params.li, input, prev_state(tape, step, index)
    );
    if (membrane.data == NULL) {
        set_error(error, "node operation failed or out of memory");
        return 0;
    }
    nir_tensor_t next = nir_tensor_clone(&membrane);
    if (next.data == NULL) {
        nir_tensor_free(&membrane);
        set_error(error, "node operation failed or out of memory");
        return 0;
    }
    tape->values[step * n + index] = membrane;
    tape->stored[step * n + index] = next;
    return 1;
}

static int apply_spiking(
    nir_tape_t *tape, size_t step, size_t index,
    const nir_tensor_t *input, char **error
) {
    const nir_node_t *node = &tape->graph->nodes[index];
    size_t n = tape->graph->node_count;
    const nir_tensor_t *prev = prev_state(tape, step, index);
    nir_tensor_t spike = alloc_like(input);
    nir_tensor_t stored = alloc_like(input);
    int ok = spike.data != NULL && stored.data != NULL;
    if (ok && node->kind == NIR_NODE_LIF) {
        ok = nir_op_lif_into(
            &node->params.lif, input, prev, &spike, &stored
        );
    } else if (ok) {
        ok = nir_op_if_into(
            &node->params.if_, input, prev, &spike, &stored
        );
    }
    if (!ok) {
        nir_tensor_free(&spike);
        nir_tensor_free(&stored);
        set_error(error, "node operation failed or out of memory");
        return 0;
    }
    tape->values[step * n + index] = spike;
    tape->stored[step * n + index] = stored;
    return 1;
}

static int apply_cuba(
    nir_tape_t *tape, size_t step, size_t index,
    const nir_tensor_t *input, char **error
) {
    const nir_node_t *node = &tape->graph->nodes[index];
    size_t n = tape->graph->node_count;
    const nir_tensor_t *syn_prev = NULL;
    const nir_tensor_t *mem_prev = prev_state(tape, step, index);
    if (step > 0) {
        syn_prev = &tape->syn[(step - 1) * n + index];
    }
    nir_tensor_t spike = alloc_like(input);
    nir_tensor_t syn = alloc_like(input);
    nir_tensor_t stored = alloc_like(input);
    int ok = spike.data != NULL && syn.data != NULL &&
        stored.data != NULL && nir_op_cuba_into(
            &node->params.cuba, input, syn_prev, mem_prev,
            &spike, &syn, &stored
        );
    if (!ok) {
        nir_tensor_free(&spike);
        nir_tensor_free(&syn);
        nir_tensor_free(&stored);
        set_error(error, "node operation failed or out of memory");
        return 0;
    }
    tape->values[step * n + index] = spike;
    tape->syn[step * n + index] = syn;
    tape->stored[step * n + index] = stored;
    return 1;
}

/* Evaluate one non-Input, non-Delay node from its retained
 * summed input, retaining the value (and post-step state for
 * stateful kinds). The op sequence mirrors runtime.c's
 * apply_node exactly. */
static int apply_node(
    nir_tape_t *tape, size_t step, size_t index,
    const nir_tensor_t *input, char **error
) {
    const nir_node_t *node = &tape->graph->nodes[index];
    size_t n = tape->graph->node_count;
    nir_tensor_t *value = &tape->values[step * n + index];
    switch (node->kind) {
        case NIR_NODE_CONV2D:
            *value = nir_op_conv2d(&node->params.conv2d, input);
            break;
        case NIR_NODE_CONV1D:
            *value = nir_op_conv1d(&node->params.conv1d, input);
            break;
        case NIR_NODE_AFFINE:
            *value = nir_op_affine(&node->params.affine, input);
            break;
        case NIR_NODE_LINEAR:
            *value = nir_op_linear(&node->params.linear, input);
            break;
        case NIR_NODE_FLATTEN:
            *value = nir_op_flatten(input);
            break;
        case NIR_NODE_AVGPOOL2D:
            *value = nir_op_avgpool2d(&node->params.pool2d, input);
            break;
        case NIR_NODE_SUMPOOL2D:
            *value = nir_op_sumpool2d(&node->params.pool2d, input);
            break;
        case NIR_NODE_LI:
            return apply_li(tape, step, index, input, error);
        case NIR_NODE_LIF:
        case NIR_NODE_IF:
            return apply_spiking(tape, step, index, input, error);
        case NIR_NODE_CUBALIF:
            return apply_cuba(tape, step, index, input, error);
        case NIR_NODE_THRESHOLD:
            *value = nir_op_threshold(&node->params.threshold, input);
            break;
        case NIR_NODE_SCALE:
            *value = nir_op_scale(&node->params.scale, input);
            break;
        case NIR_NODE_OUTPUT:
            *value = nir_tensor_clone(input); /* pass-through */
            break;
        default: /* Input and Delay never reach here. */
            break;
    }
    if (value->data == NULL) {
        set_error(error, "node operation failed or out of memory");
        return 0;
    }
    return 1;
}

static int run_step(
    nir_tape_t *tape, const nir_tensor_t *frame, size_t step,
    char **error
) {
    const nir_graph_t *graph = tape->graph;
    size_t n = graph->node_count;
    tape->values[step * n + graph->input_index] =
        nir_tensor_clone(frame);
    if (tape->values[step * n + graph->input_index].data == NULL) {
        set_error(error, "invalid input tensor or out of memory");
        return 0;
    }
    /* A Delay's value is the previous step's retained summed
     * input -- the same buffer the runtime refreshes after each
     * step. At step 0 the slot stays empty (unprimed). */
    if (step > 0) {
        for (size_t i = 0; i < n; i++) {
            if (graph->nodes[i].kind != NIR_NODE_DELAY) {
                continue;
            }
            tape->values[step * n + i] = nir_tensor_clone(
                &tape->inputs[(step - 1) * n + i]
            );
            if (tape->values[step * n + i].data == NULL) {
                set_error(error, "invalid delay state or out of memory");
                return 0;
            }
        }
    }
    for (size_t o = 0; o < n; o++) {
        size_t index = tape->sched.order[o];
        nir_node_kind_t kind = graph->nodes[index].kind;
        if (kind == NIR_NODE_INPUT || kind == NIR_NODE_DELAY) {
            continue;
        }
        if (!sum_incoming(
                tape, step, index,
                &tape->inputs[step * n + index], error) ||
            !apply_node(
                tape, step, index,
                &tape->inputs[step * n + index], error)) {
            return 0;
        }
    }
    for (size_t i = 0; i < n; i++) {
        if (graph->nodes[i].kind != NIR_NODE_DELAY) {
            continue;
        }
        if (!sum_incoming(
                tape, step, i, &tape->inputs[step * n + i],
                error)) {
            return 0;
        }
    }
    return 1;
}

/* Zeroed gradient buffers shaped like the retained values and
 * states they differentiate. */
static int alloc_grad_buffers(nir_tape_t *tape, char **error) {
    size_t slots = tape_slots(tape);
    for (size_t s = 0; s < slots; s++) {
        if (tape->values[s].data != NULL) {
            tape->grads[s] = nir_tensor_alloc(
                tape->values[s].shape, tape->values[s].ndim
            );
            if (tape->grads[s].data == NULL) {
                set_error(error, "out of memory");
                return 0;
            }
        }
        if (tape->stored[s].data != NULL) {
            tape->state_grads[s] = nir_tensor_alloc(
                tape->stored[s].shape, tape->stored[s].ndim
            );
            if (tape->state_grads[s].data == NULL) {
                set_error(error, "out of memory");
                return 0;
            }
        }
    }
    return 1;
}

int nir_tape_forward(
    nir_tape_t *tape, const nir_tensor_t *frames,
    nir_tensor_t *readout_out, char **error
) {
    if (tape == NULL || tape->graph == NULL || frames == NULL ||
        readout_out == NULL) {
        set_error(error, "invalid tape arguments");
        return 0;
    }
    clear_retained(tape);
    const nir_graph_t *graph = tape->graph;
    size_t n = graph->node_count;
    for (size_t t = 0; t < tape->num_steps; t++) {
        if (!run_step(tape, &frames[t], t, error)) {
            clear_retained(tape);
            return 0;
        }
    }
    nir_tensor_t total = nir_tensor_clone(
        &tape->values[graph->output_index]
    );
    if (total.data == NULL) {
        clear_retained(tape);
        set_error(error, "out of memory or invalid output tensor");
        return 0;
    }
    for (size_t t = 1; t < tape->num_steps; t++) {
        if (!nir_tensor_add_inplace(
                &total, &tape->values[t * n + graph->output_index])) {
            nir_tensor_free(&total);
            clear_retained(tape);
            set_error(error, "runtime output shape changed");
            return 0;
        }
    }
    for (size_t i = 0; i < total.size; i++) {
        total.data[i] /= (float)tape->num_steps;
    }
    if (!alloc_grad_buffers(tape, error)) {
        nir_tensor_free(&total);
        clear_retained(tape);
        return 0;
    }
    *readout_out = total;
    tape->forwarded = 1;
    return 1;
}

/* Accumulate a consumer's input gradient into a source's value
 * gradient. Equal sizes add pointwise; a single-element source
 * (a Delay holding the zero scalar) folds the broadcast sum,
 * mirroring the forward broadcast add. Any other mismatch is
 * unreachable after a successful forward. */
static int accumulate_grad(nir_tensor_t *acc, const nir_tensor_t *g) {
    if (!tensor_valid(acc) || !tensor_valid(g)) {
        return 0;
    }
    if (acc->size == g->size) {
        for (size_t i = 0; i < acc->size; i++) {
            acc->data[i] += g->data[i];
        }
        return 1;
    }
    if (acc->size == 1) {
        double sum = 0.0;
        for (size_t i = 0; i < g->size; i++) {
            sum += (double)g->data[i];
        }
        acc->data[0] += (float)sum;
        return 1;
    }
    return 0;
}

/* Route `grad` (dL/d of `target`'s summed input) to every source
 * that contributed to that sum at step `src_step`. */
static int route_to_sources(
    nir_tape_t *tape, size_t target, size_t src_step,
    const nir_tensor_t *grad, char **error
) {
    const nir_graph_t *graph = tape->graph;
    size_t n = graph->node_count;
    for (size_t e = 0; e < graph->edge_count; e++) {
        if (graph->edges[e].target_index != target) {
            continue;
        }
        size_t src = graph->edges[e].source_index;
        if (!source_contributes(graph, src, src_step)) {
            continue;
        }
        if (!accumulate_grad(
                &tape->grads[src_step * n + src], grad)) {
            set_error(error, "gradient shape mismatch routing node inputs");
            return 0;
        }
    }
    return 1;
}

/* The layer backward ops overwrite their grad_input, so one
 * reusable scratch tensor serves every node; it only grows. */
static int scratch_for(nir_tensor_t *scratch, size_t need) {
    if (scratch->data != NULL && scratch->capacity >= need) {
        return 1;
    }
    nir_tensor_free(scratch);
    size_t shape[1] = {need};
    *scratch = nir_tensor_alloc(shape, 1);
    return scratch->data != NULL;
}

static const char *backward_error_for(nir_node_kind_t kind) {
    switch (kind) {
        case NIR_NODE_AFFINE:
            return "affine node backward pass failed";
        case NIR_NODE_LINEAR:
            return "linear node backward pass failed";
        case NIR_NODE_CONV2D:
            return "conv2d node backward pass failed";
        case NIR_NODE_CONV1D:
            return "conv1d node backward pass failed";
        case NIR_NODE_FLATTEN:
            return "flatten node backward pass failed";
        case NIR_NODE_AVGPOOL2D:
            return "avgpool2d node backward pass failed";
        case NIR_NODE_SUMPOOL2D:
            return "sumpool2d node backward pass failed";
        case NIR_NODE_THRESHOLD:
            return "threshold node backward pass failed";
        case NIR_NODE_SCALE:
            return "scale node backward pass failed";
        case NIR_NODE_LI:
            return "LI node backward pass failed";
        case NIR_NODE_LIF:
            return "LIF node backward pass failed";
        case NIR_NODE_IF:
            return "IF node backward pass failed";
        case NIR_NODE_CUBALIF:
            return "CubaLIF node backward pass failed";
        default:
            return "node backward pass failed";
    }
}

/* This step's routed stored-state gradient, for the spiking
 * ops' grad_stored tape wiring -- or NULL when the slot is
 * empty or shape-shifted (defensive; the ops ignore it under
 * the reset convention either way). */
static const nir_tensor_t *stored_grad_or_null(
    nir_tape_t *tape, size_t step, size_t index
) {
    size_t n = tape->graph->node_count;
    const nir_tensor_t *sg = &tape->state_grads[step * n + index];
    const nir_tensor_t *input = &tape->inputs[step * n + index];
    if (sg->data != NULL && tensor_same_shape(sg, input)) {
        return sg;
    }
    return NULL;
}

static int backward_spiking(
    nir_tape_t *tape, size_t step, size_t index,
    nir_tensor_t *scratch, char **error
) {
    const nir_node_t *node = &tape->graph->nodes[index];
    size_t n = tape->graph->node_count;
    const nir_tensor_t *input = &tape->inputs[step * n + index];
    const nir_tensor_t *gv = &tape->grads[step * n + index];
    const nir_tensor_t *v_prev =
        step > 0 ? &tape->stored[(step - 1) * n + index] : NULL;
    nir_tensor_t *gprev = step > 0 ?
        &tape->state_grads[(step - 1) * n + index] : NULL;
    const nir_tensor_t *gstored =
        stored_grad_or_null(tape, step, index);
    if (!scratch_for(scratch, input->size)) {
        set_error(error, "out of memory");
        return 0;
    }
    int ok = 0;
    if (node->kind == NIR_NODE_LIF) {
        ok = nir_op_lif_backward(
            &node->params.lif, input, v_prev, gv, gstored,
            tape->surrogate, tape->slope, scratch, gprev
        );
    } else if (node->kind == NIR_NODE_IF) {
        ok = nir_op_if_backward(
            &node->params.if_, input, v_prev, gv, gstored,
            tape->surrogate, tape->slope, scratch, gprev
        );
    } else {
        const nir_tensor_t *syn_prev = step > 0 ?
            &tape->syn[(step - 1) * n + index] : NULL;
        /* nir_op_cuba_backward has no gradient input for the
         * synaptic current, so cross-step synaptic credit is
         * truncated here (see tape.h). */
        ok = nir_op_cuba_backward(
            &node->params.cuba, input, syn_prev, v_prev, gv,
            gstored, tape->surrogate, tape->slope, scratch,
            NULL, gprev
        );
    }
    if (!ok) {
        set_error(error, backward_error_for(node->kind));
        return 0;
    }
    return route_to_sources(tape, index, step, scratch, error);
}

static int backward_li(
    nir_tape_t *tape, size_t step, size_t index,
    nir_tensor_t *scratch, char **error
) {
    const nir_node_t *node = &tape->graph->nodes[index];
    size_t n = tape->graph->node_count;
    nir_tensor_t *gv = &tape->grads[step * n + index];
    nir_tensor_t *sg = &tape->state_grads[step * n + index];
    nir_tensor_t *sprev = step > 0 ?
        &tape->state_grads[(step - 1) * n + index] : NULL;
    /* The membrane gradient is the consumer gradient plus the
     * next step's previous-state gradient; both are final by
     * now (reverse step order), so fold in place. */
    if (!nir_tensor_add_inplace(gv, sg) ||
        !scratch_for(scratch, gv->size) ||
        !nir_op_li_backward(&node->params.li, gv, scratch, sprev)) {
        set_error(error, backward_error_for(node->kind));
        return 0;
    }
    return route_to_sources(tape, index, step, scratch, error);
}

static int backward_affine(
    nir_tape_t *tape, size_t step, size_t index,
    nir_tensor_t *scratch, char **error
) {
    const nir_node_t *node = &tape->graph->nodes[index];
    size_t n = tape->graph->node_count;
    const nir_tensor_t *input = &tape->inputs[step * n + index];
    const nir_tensor_t *gv = &tape->grads[step * n + index];
    int ok = 0;
    if (!scratch_for(scratch, input->size)) {
        set_error(error, "out of memory");
        return 0;
    }
    if (node->kind == NIR_NODE_AFFINE) {
        nir_tensor_t *gb = node->params.affine.bias != NULL ?
            &tape->grad_bias[index] : NULL;
        ok = nir_op_affine_backward(
            &node->params.affine, input, gv,
            &tape->grad_weight[index], gb, scratch
        );
    } else {
        ok = nir_op_linear_backward(
            &node->params.linear, input, gv,
            &tape->grad_weight[index], scratch
        );
    }
    if (!ok) {
        set_error(error, backward_error_for(node->kind));
        return 0;
    }
    return route_to_sources(tape, index, step, scratch, error);
}

static int backward_conv(
    nir_tape_t *tape, size_t step, size_t index,
    nir_tensor_t *scratch, char **error
) {
    const nir_node_t *node = &tape->graph->nodes[index];
    size_t n = tape->graph->node_count;
    const nir_tensor_t *input = &tape->inputs[step * n + index];
    const nir_tensor_t *gv = &tape->grads[step * n + index];
    int ok = 0;
    if (!scratch_for(scratch, input->size)) {
        set_error(error, "out of memory");
        return 0;
    }
    if (node->kind == NIR_NODE_CONV2D) {
        nir_tensor_t *gb = node->params.conv2d.bias != NULL ?
            &tape->grad_bias[index] : NULL;
        ok = nir_op_conv2d_backward(
            &node->params.conv2d, input, gv,
            &tape->grad_weight[index], gb, scratch
        );
    } else {
        nir_tensor_t *gb = node->params.conv1d.bias != NULL ?
            &tape->grad_bias[index] : NULL;
        ok = nir_op_conv1d_backward(
            &node->params.conv1d, input, gv,
            &tape->grad_weight[index], gb, scratch
        );
    }
    if (!ok) {
        set_error(error, backward_error_for(node->kind));
        return 0;
    }
    return route_to_sources(tape, index, step, scratch, error);
}

static int backward_stateless(
    nir_tape_t *tape, size_t step, size_t index,
    nir_tensor_t *scratch, char **error
) {
    const nir_node_t *node = &tape->graph->nodes[index];
    size_t n = tape->graph->node_count;
    const nir_tensor_t *input = &tape->inputs[step * n + index];
    const nir_tensor_t *gv = &tape->grads[step * n + index];
    int ok = 0;
    if (!scratch_for(scratch, input->size)) {
        set_error(error, "out of memory");
        return 0;
    }
    switch (node->kind) {
        case NIR_NODE_FLATTEN:
            ok = nir_op_flatten_backward(input, gv, scratch);
            break;
        case NIR_NODE_AVGPOOL2D:
            ok = nir_op_avgpool2d_backward(
                &node->params.pool2d, input, gv, scratch
            );
            break;
        case NIR_NODE_SUMPOOL2D:
            ok = nir_op_sumpool2d_backward(
                &node->params.pool2d, input, gv, scratch
            );
            break;
        case NIR_NODE_THRESHOLD:
            ok = nir_op_threshold_backward(
                &node->params.threshold, input, gv,
                tape->surrogate, tape->slope, scratch
            );
            break;
        case NIR_NODE_SCALE:
            ok = nir_op_scale_backward(
                &node->params.scale, input, gv, scratch,
                &tape->grad_scale[index]
            );
            break;
        default:
            break;
    }
    if (!ok) {
        set_error(error, backward_error_for(node->kind));
        return 0;
    }
    return route_to_sources(tape, index, step, scratch, error);
}

static int backward_node(
    nir_tape_t *tape, size_t step, size_t index,
    nir_tensor_t *scratch, char **error
) {
    nir_node_kind_t kind = tape->graph->nodes[index].kind;
    switch (kind) {
        case NIR_NODE_INPUT:
            return 1;
        case NIR_NODE_DELAY:
            if (step == 0) {
                return 1;
            }
            /* The step-t value is the previous step's buffer:
             * route its gradient to the step-(t-1) sources. */
            return route_to_sources(
                tape, index, step - 1,
                &tape->grads[
                    step * tape->graph->node_count + index
                ],
                error
            );
        case NIR_NODE_OUTPUT:
            return route_to_sources(
                tape, index, step,
                &tape->grads[
                    step * tape->graph->node_count + index
                ],
                error
            );
        case NIR_NODE_AFFINE:
        case NIR_NODE_LINEAR:
            return backward_affine(tape, step, index, scratch, error);
        case NIR_NODE_CONV2D:
        case NIR_NODE_CONV1D:
            return backward_conv(tape, step, index, scratch, error);
        case NIR_NODE_FLATTEN:
        case NIR_NODE_AVGPOOL2D:
        case NIR_NODE_SUMPOOL2D:
        case NIR_NODE_THRESHOLD:
        case NIR_NODE_SCALE:
            return backward_stateless(
                tape, step, index, scratch, error
            );
        case NIR_NODE_LI:
            return backward_li(tape, step, index, scratch, error);
        case NIR_NODE_LIF:
        case NIR_NODE_IF:
        case NIR_NODE_CUBALIF:
            return backward_spiking(
                tape, step, index, scratch, error
            );
        default:
            set_error(error, "tape encountered an invalid node kind");
            return 0;
    }
}

static void zero_tensor_data(nir_tensor_t *tensor) {
    if (tensor->data != NULL) {
        memset(tensor->data, 0, tensor->size * sizeof(float));
    }
}

static void zero_backward_buffers(nir_tape_t *tape) {
    size_t slots = tape_slots(tape);
    for (size_t s = 0; s < slots; s++) {
        zero_tensor_data(&tape->grads[s]);
        zero_tensor_data(&tape->state_grads[s]);
    }
    for (size_t i = 0; i < tape->graph->node_count; i++) {
        zero_tensor_data(&tape->grad_weight[i]);
        zero_tensor_data(&tape->grad_bias[i]);
    }
    memset(
        tape->grad_scale, 0,
        tape->graph->node_count * sizeof(float)
    );
}

/* Seed one step's output-value gradient with grad/T. A
 * single-element step output (a step whose output summed only
 * unprimed-delay scalars) folds the broadcast sum, mirroring
 * the forward averaging. */
static int seed_output_step(
    nir_tensor_t *g, const nir_tensor_t *grad_readout, float inv,
    char **error
) {
    if (!tensor_valid(g)) {
        set_error(error, "readout gradient shape mismatch");
        return 0;
    }
    if (g->size == grad_readout->size) {
        for (size_t k = 0; k < g->size; k++) {
            g->data[k] += grad_readout->data[k] * inv;
        }
        return 1;
    }
    if (g->size == 1) {
        double sum = 0.0;
        for (size_t k = 0; k < grad_readout->size; k++) {
            sum += (double)grad_readout->data[k];
        }
        g->data[0] += (float)sum * inv;
        return 1;
    }
    set_error(error, "readout gradient shape mismatch");
    return 0;
}

int nir_tape_backward(
    nir_tape_t *tape, const nir_tensor_t *grad_readout,
    char **error
) {
    if (tape == NULL || tape->graph == NULL ||
        grad_readout == NULL) {
        set_error(error, "invalid tape arguments");
        return 0;
    }
    if (!tape->forwarded) {
        set_error(error, "tape has no retained forward pass");
        return 0;
    }
    const nir_graph_t *graph = tape->graph;
    size_t n = graph->node_count;
    if (!tensor_valid(grad_readout) || !tensor_same_shape(
            grad_readout, &tape->values[graph->output_index])) {
        set_error(error, "readout gradient shape mismatch");
        return 0;
    }
    zero_backward_buffers(tape);
    tape->have_grads = 0;
    float inv = 1.0f / (float)tape->num_steps;
    for (size_t t = 0; t < tape->num_steps; t++) {
        if (!seed_output_step(
                &tape->grads[t * n + graph->output_index],
                grad_readout, inv, error)) {
            return 0;
        }
    }
    nir_tensor_t scratch = {0};
    for (size_t ti = tape->num_steps; ti-- > 0;) {
        for (size_t oi = n; oi-- > 0;) {
            if (!backward_node(
                    tape, ti, tape->sched.order[oi], &scratch,
                    error)) {
                nir_tensor_free(&scratch);
                return 0;
            }
        }
    }
    nir_tensor_free(&scratch);
    tape->have_grads = 1;
    return 1;
}

static int is_weighted(nir_node_kind_t kind) {
    return kind == NIR_NODE_AFFINE || kind == NIR_NODE_LINEAR ||
        kind == NIR_NODE_CONV2D || kind == NIR_NODE_CONV1D;
}

static int node_has_bias(const nir_node_t *node) {
    switch (node->kind) {
        case NIR_NODE_AFFINE:
            return node->params.affine.bias != NULL;
        case NIR_NODE_CONV2D:
            return node->params.conv2d.bias != NULL;
        case NIR_NODE_CONV1D:
            return node->params.conv1d.bias != NULL;
        default:
            return 0;
    }
}

const nir_tensor_t *nir_tape_weight_grad(
    const nir_tape_t *tape, size_t node_index
) {
    if (tape == NULL || tape->graph == NULL || !tape->have_grads ||
        node_index >= tape->graph->node_count ||
        !is_weighted(tape->graph->nodes[node_index].kind)) {
        return NULL;
    }
    return &tape->grad_weight[node_index];
}

const nir_tensor_t *nir_tape_bias_grad(
    const nir_tape_t *tape, size_t node_index
) {
    if (tape == NULL || tape->graph == NULL || !tape->have_grads ||
        node_index >= tape->graph->node_count ||
        !node_has_bias(&tape->graph->nodes[node_index])) {
        return NULL;
    }
    return &tape->grad_bias[node_index];
}

int nir_tape_scale_grad(
    const nir_tape_t *tape, size_t node_index, float *out
) {
    if (tape == NULL || tape->graph == NULL || out == NULL ||
        !tape->have_grads ||
        node_index >= tape->graph->node_count ||
        tape->graph->nodes[node_index].kind != NIR_NODE_SCALE) {
        return 0;
    }
    *out = tape->grad_scale[node_index];
    return 1;
}

const nir_tensor_t *nir_tape_input_grad(
    const nir_tape_t *tape, size_t step
) {
    if (tape == NULL || tape->graph == NULL || !tape->have_grads ||
        step >= tape->num_steps) {
        return NULL;
    }
    return &tape->grads[
        step * tape->graph->node_count + tape->graph->input_index
    ];
}
