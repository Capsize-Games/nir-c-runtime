#include "nir_c_runtime/runtime.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/ops.h"

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

/* Topological order over "reduced" edges: an edge whose target is a
 * Delay node does not count, since a Delay's value for this step
 * comes from the previous step's buffer, not from evaluating a
 * source first. Mirrors reduced_edges() in spikeforge's
 * nir_bridge/ops_registry.py. */
static int compute_order(
    const nir_graph_t *graph, size_t *order, char **error
) {
    size_t n = graph->node_count;
    size_t *pending = (size_t *)calloc(n == 0 ? 1 : n, sizeof(size_t));
    size_t *queue = (size_t *)malloc((n == 0 ? 1 : n) * sizeof(size_t));
    if (pending == NULL || queue == NULL) {
        free(pending);
        free(queue);
        set_error(error, "out of memory");
        return 0;
    }
    for (size_t e = 0; e < graph->edge_count; e++) {
        size_t target = graph->edges[e].target_index;
        if (graph->nodes[target].kind != NIR_NODE_DELAY) {
            pending[target]++;
        }
    }
    size_t head = 0, tail = 0, filled = 0;
    for (size_t i = 0; i < n; i++) {
        if (pending[i] == 0) {
            queue[tail++] = i;
        }
    }
    while (head < tail) {
        size_t node = queue[head++];
        order[filled++] = node;
        for (size_t e = 0; e < graph->edge_count; e++) {
            if (graph->edges[e].source_index != node) {
                continue;
            }
            size_t child = graph->edges[e].target_index;
            if (graph->nodes[child].kind == NIR_NODE_DELAY) {
                continue;
            }
            if (--pending[child] == 0) {
                queue[tail++] = child;
            }
        }
    }
    free(pending);
    free(queue);
    if (filled != n) {
        set_error(error, "NIR graph cycle is not broken by a Delay node");
        return 0;
    }
    return 1;
}

int nir_runtime_init(
    nir_runtime_t *rt, const nir_graph_t *graph, char **error
) {
    if (rt == NULL) {
        set_error(error, "invalid runtime graph");
        return 0;
    }
    memset(rt, 0, sizeof(*rt));
    if (graph == NULL || graph->node_count == 0) {
        set_error(error, "invalid runtime graph");
        return 0;
    }
    size_t n = graph->node_count;
    if (n > SIZE_MAX / sizeof(size_t) ||
        n > SIZE_MAX / sizeof(nir_tensor_t) ||
        n > SIZE_MAX / sizeof(int)) {
        set_error(error, "runtime graph is too large");
        return 0;
    }
    if (graph->nodes == NULL ||
        (graph->edge_count != 0 && graph->edges == NULL) ||
        graph->input_index >= n || graph->output_index >= n) {
        set_error(error, "invalid runtime graph");
        return 0;
    }
    if (graph->nodes[graph->input_index].kind != NIR_NODE_INPUT ||
        graph->nodes[graph->output_index].kind != NIR_NODE_OUTPUT) {
        set_error(error, "runtime graph has invalid input or output node");
        return 0;
    }
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].kind < NIR_NODE_INPUT ||
            graph->nodes[i].kind > NIR_NODE_LINEAR) {
            set_error(error, "runtime graph contains an invalid node kind");
            return 0;
        }
    }
    for (size_t e = 0; e < graph->edge_count; e++) {
        if (graph->edges[e].source_index >= graph->node_count ||
            graph->edges[e].target_index >= graph->node_count) {
            set_error(error, "invalid runtime graph edge");
            return 0;
        }
    }
    rt->graph = graph;
    rt->order = (size_t *)malloc((n == 0 ? 1 : n) * sizeof(size_t));
    rt->li_state = (nir_tensor_t *)calloc(
        n == 0 ? 1 : n, sizeof(nir_tensor_t)
    );
    rt->li_primed = (int *)calloc(n == 0 ? 1 : n, sizeof(int));
    rt->syn_state = (nir_tensor_t *)calloc(
        n == 0 ? 1 : n, sizeof(nir_tensor_t)
    );
    rt->delay_buffer = (nir_tensor_t *)calloc(
        n == 0 ? 1 : n, sizeof(nir_tensor_t)
    );
    rt->delay_primed = (int *)calloc(n == 0 ? 1 : n, sizeof(int));
    if (rt->order == NULL || rt->li_state == NULL || rt->li_primed == NULL ||
        rt->syn_state == NULL || rt->delay_buffer == NULL ||
        rt->delay_primed == NULL) {
        nir_runtime_free(rt);
        set_error(error, "out of memory");
        return 0;
    }
    if (!compute_order(graph, rt->order, error)) {
        nir_runtime_free(rt);
        return 0;
    }
    return 1;
}

void nir_runtime_free(nir_runtime_t *rt) {
    if (rt == NULL) {
        return;
    }
    if (rt->graph != NULL) {
        for (size_t i = 0; i < rt->graph->node_count; i++) {
            if (rt->li_state != NULL) nir_tensor_free(&rt->li_state[i]);
            if (rt->syn_state != NULL) nir_tensor_free(&rt->syn_state[i]);
            if (rt->delay_buffer != NULL) {
                nir_tensor_free(&rt->delay_buffer[i]);
            }
        }
    }
    free(rt->order);
    free(rt->li_state);
    free(rt->li_primed);
    free(rt->syn_state);
    free(rt->delay_buffer);
    free(rt->delay_primed);
    memset(rt, 0, sizeof(*rt));
}

/* Sum the current values of every edge feeding into `target_index`.
 * A Delay source that has not produced a value yet (its very first
 * step) is skipped rather than read -- if every source is such a
 * delay (or there simply are none feeding a real value in yet), the
 * result is a single-element zero, matching the reference
 * interpreter's 0-d scalar-zero broadcast for an unprimed delay. */
static int sum_incoming(
    nir_runtime_t *rt,
    nir_tensor_t *values,
    size_t target_index,
    nir_tensor_t *out,
    char **error
) {
    const nir_graph_t *graph = rt->graph;
    int have_any = 0, found_edge = 0;
    for (size_t e = 0; e < graph->edge_count; e++) {
        if (graph->edges[e].target_index != target_index) {
            continue;
        }
        found_edge = 1;
        size_t src = graph->edges[e].source_index;
        if (graph->nodes[src].kind == NIR_NODE_DELAY &&
            !rt->delay_primed[src]) {
            continue;
        }
        if (!have_any) {
            *out = nir_tensor_clone(&values[src]);
            if (out->data == NULL) {
                set_error(error, "out of memory or invalid node tensor");
                return 0;
            }
            have_any = 1;
        } else if (!nir_tensor_add_inplace(out, &values[src])) {
            nir_tensor_free(out);
            set_error(error, "shape mismatch summing node inputs");
            return 0;
        }
    }
    if (!found_edge) {
        set_error_arg(error, "node ", graph->nodes[target_index].name,
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

static nir_tensor_t apply_node(
    const nir_node_t *node,
    const nir_tensor_t *input,
    nir_runtime_t *rt,
    size_t index
) {
    switch (node->kind) {
        case NIR_NODE_CONV2D:
            return nir_op_conv2d(&node->params.conv2d, input);
        case NIR_NODE_CONV1D:
            return nir_op_conv1d(&node->params.conv1d, input);
        case NIR_NODE_AFFINE:
            return nir_op_affine(&node->params.affine, input);
        case NIR_NODE_LINEAR:
            return nir_op_linear(&node->params.linear, input);
        case NIR_NODE_FLATTEN:
            return nir_op_flatten(input);
        case NIR_NODE_AVGPOOL2D:
            return nir_op_avgpool2d(&node->params.pool2d, input);
        case NIR_NODE_SUMPOOL2D:
            return nir_op_sumpool2d(&node->params.pool2d, input);
        case NIR_NODE_LI: {
            const nir_tensor_t *prev =
                rt->li_primed[index] ? &rt->li_state[index] : NULL;
            nir_tensor_t membrane = nir_op_li(
                &node->params.li, input, prev
            );
            if (membrane.data == NULL) {
                return membrane;
            }
            nir_tensor_t next_state = nir_tensor_clone(&membrane);
            if (next_state.data == NULL) {
                nir_tensor_free(&membrane);
                return (nir_tensor_t){0};
            }
            nir_tensor_free(&rt->li_state[index]);
            rt->li_state[index] = next_state;
            rt->li_primed[index] = 1;
            return membrane;
        }
        case NIR_NODE_LIF:
        case NIR_NODE_IF: {
            const nir_tensor_t *prev =
                rt->li_primed[index] ? &rt->li_state[index] : NULL;
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
                return (nir_tensor_t){0};
            }
            nir_tensor_free(&rt->li_state[index]);
            rt->li_state[index] = stored;
            rt->li_primed[index] = 1;
            return spike;
        }
        case NIR_NODE_CUBALIF: {
            const nir_tensor_t *syn_prev =
                rt->li_primed[index] ? &rt->syn_state[index] : NULL;
            const nir_tensor_t *mem_prev =
                rt->li_primed[index] ? &rt->li_state[index] : NULL;
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
                return (nir_tensor_t){0};
            }
            nir_tensor_free(&rt->syn_state[index]);
            nir_tensor_free(&rt->li_state[index]);
            rt->syn_state[index] = syn;
            rt->li_state[index] = stored;
            rt->li_primed[index] = 1;
            return spike;
        }
        case NIR_NODE_THRESHOLD:
            return nir_op_threshold(&node->params.threshold, input);
        case NIR_NODE_SCALE:
            return nir_op_scale(&node->params.scale, input);
        case NIR_NODE_OUTPUT:
        default:
            return nir_tensor_clone(input); /* Output is a pass-through */
    }
}

static void free_values(nir_tensor_t *values, size_t count) {
    for (size_t i = 0; i < count; i++) {
        nir_tensor_free(&values[i]);
    }
    free(values);
}

static int run_step(
    nir_runtime_t *rt, const nir_tensor_t *frame, nir_tensor_t *values,
    char **error
) {
    const nir_graph_t *graph = rt->graph;
    values[graph->input_index] = nir_tensor_clone(frame);
    if (values[graph->input_index].data == NULL) {
        set_error(error, "invalid input tensor or out of memory");
        return 0;
    }
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].kind == NIR_NODE_DELAY && rt->delay_primed[i]) {
            values[i] = nir_tensor_clone(&rt->delay_buffer[i]);
            if (values[i].data == NULL) {
                set_error(error, "invalid delay state or out of memory");
                return 0;
            }
        }
    }
    for (size_t o = 0; o < graph->node_count; o++) {
        size_t index = rt->order[o];
        nir_node_kind_t kind = graph->nodes[index].kind;
        if (kind == NIR_NODE_INPUT || kind == NIR_NODE_DELAY) {
            continue;
        }
        nir_tensor_t summed;
        if (!sum_incoming(rt, values, index, &summed, error)) {
            return 0;
        }
        values[index] = apply_node(&graph->nodes[index], &summed, rt, index);
        nir_tensor_free(&summed);
        if (values[index].data == NULL) {
            set_error(error, "node operation failed or out of memory");
            return 0;
        }
    }
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].kind != NIR_NODE_DELAY) {
            continue;
        }
        nir_tensor_t refreshed;
        if (!sum_incoming(rt, values, i, &refreshed, error)) {
            return 0;
        }
        nir_tensor_free(&rt->delay_buffer[i]);
        rt->delay_buffer[i] = refreshed;
        rt->delay_primed[i] = 1;
    }
    return 1;
}

static int graph_has_cuba(const nir_graph_t *graph) {
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].kind == NIR_NODE_CUBALIF) {
            return 1;
        }
    }
    return 0;
}

static int workspace_ready(
    const nir_runtime_t *rt, const nir_runtime_workspace_t *workspace,
    const nir_tensor_t *readout_out, char **error
) {
    size_t n = rt->graph->node_count;
    if (workspace == NULL || workspace->node_capacity < n ||
        workspace->values == NULL || workspace->li_state == NULL ||
        workspace->li_primed == NULL || workspace->delay_buffer == NULL ||
        workspace->delay_primed == NULL || workspace->sum == NULL ||
        workspace->total == NULL || readout_out == NULL ||
        workspace->sum->data == NULL || workspace->total->data == NULL ||
        readout_out->data == NULL ||
        (graph_has_cuba(rt->graph) && workspace->syn_state == NULL)) {
        set_error(error, "incomplete runtime workspace");
        return 0;
    }
    return 1;
}

static int set_zero_scalar(nir_tensor_t *tensor) {
    size_t shape[1] = {1};
    if (tensor->data == NULL || tensor->capacity < 1) {
        return 0;
    }
    tensor->ndim = 1;
    tensor->shape[0] = shape[0];
    tensor->shape[1] = 0;
    tensor->shape[2] = 0;
    tensor->size = 1;
    tensor->data[0] = 0.0f;
    return 1;
}

static int sum_incoming_workspace(
    nir_runtime_t *rt, nir_runtime_workspace_t *workspace,
    size_t target_index, char **error
) {
    const nir_graph_t *graph = rt->graph;
    nir_tensor_t *sum = workspace->sum;
    int have_any = 0, found_edge = 0;
    for (size_t e = 0; e < graph->edge_count; e++) {
        if (graph->edges[e].target_index != target_index) {
            continue;
        }
        found_edge = 1;
        size_t source = graph->edges[e].source_index;
        if (graph->nodes[source].kind == NIR_NODE_DELAY &&
            !workspace->delay_primed[source]) {
            continue;
        }
        if (!have_any) {
            if (!nir_tensor_copy_into(sum, &workspace->values[source])) {
                set_error(error, "workspace tensor is too small");
                return 0;
            }
            have_any = 1;
        } else if (!nir_tensor_add_inplace(
                       sum, &workspace->values[source])) {
            set_error(error, "shape mismatch summing node inputs");
            return 0;
        }
    }
    if (!found_edge) {
        set_error_arg(error, "node ", graph->nodes[target_index].name,
            " received no inputs");
        return 0;
    }
    if (!have_any && !set_zero_scalar(sum)) {
        set_error(error, "workspace tensor is too small");
        return 0;
    }
    return 1;
}

static int apply_node_workspace(
    const nir_node_t *node, const nir_tensor_t *input,
    nir_runtime_workspace_t *workspace, size_t index, char **error
) {
    nir_tensor_t *output = &workspace->values[index];
    int ok = 0;
    switch (node->kind) {
        case NIR_NODE_CONV2D:
            ok = nir_op_conv2d_into(&node->params.conv2d, input, output);
            break;
        case NIR_NODE_CONV1D:
            ok = nir_op_conv1d_into(&node->params.conv1d, input, output);
            break;
        case NIR_NODE_AFFINE:
            ok = nir_op_affine_into(&node->params.affine, input, output);
            break;
        case NIR_NODE_LINEAR:
            ok = nir_op_linear_into(&node->params.linear, input, output);
            break;
        case NIR_NODE_FLATTEN:
            ok = nir_op_flatten_into(input, output);
            break;
        case NIR_NODE_AVGPOOL2D:
            ok = nir_op_avgpool2d_into(&node->params.pool2d, input, output);
            break;
        case NIR_NODE_SUMPOOL2D:
            ok = nir_op_sumpool2d_into(&node->params.pool2d, input, output);
            break;
        case NIR_NODE_LI:
            ok = nir_op_li_into(
                &node->params.li, input,
                workspace->li_primed[index] ?
                    &workspace->li_state[index] : NULL,
                output
            );
            if (ok && !nir_tensor_copy_into(
                    &workspace->li_state[index], output)) {
                ok = 0;
            }
            if (ok) {
                workspace->li_primed[index] = 1;
            }
            break;
        case NIR_NODE_LIF:
        case NIR_NODE_IF: {
            const nir_tensor_t *prev = workspace->li_primed[index] ?
                &workspace->li_state[index] : NULL;
            /* The stored membrane lands directly in the state slot;
             * exact-aliasing the previous state is safe (elementwise)
             * and needs no copy, unlike LI above. */
            if (node->kind == NIR_NODE_LIF) {
                ok = nir_op_lif_into(
                    &node->params.lif, input, prev, output,
                    &workspace->li_state[index]
                );
            } else {
                ok = nir_op_if_into(
                    &node->params.if_, input, prev, output,
                    &workspace->li_state[index]
                );
            }
            if (ok) {
                workspace->li_primed[index] = 1;
            }
            break;
        }
        case NIR_NODE_CUBALIF: {
            const nir_tensor_t *syn_prev = workspace->li_primed[index] ?
                &workspace->syn_state[index] : NULL;
            const nir_tensor_t *mem_prev = workspace->li_primed[index] ?
                &workspace->li_state[index] : NULL;
            ok = nir_op_cuba_into(
                &node->params.cuba, input, syn_prev, mem_prev, output,
                &workspace->syn_state[index], &workspace->li_state[index]
            );
            if (ok) {
                workspace->li_primed[index] = 1;
            }
            break;
        }
        case NIR_NODE_THRESHOLD:
            ok = nir_op_threshold_into(
                &node->params.threshold, input, output
            );
            break;
        case NIR_NODE_SCALE:
            ok = nir_op_scale_into(&node->params.scale, input, output);
            break;
        case NIR_NODE_OUTPUT:
            ok = nir_tensor_copy_into(output, input);
            break;
        default:
            break;
    }
    if (!ok) {
        set_error(error, "workspace tensor is too small or invalid");
    }
    return ok;
}

static int run_step_workspace(
    nir_runtime_t *rt, const nir_tensor_t *frame,
    nir_runtime_workspace_t *workspace, char **error
) {
    const nir_graph_t *graph = rt->graph;
    if (!nir_tensor_copy_into(
            &workspace->values[graph->input_index], frame)) {
        set_error(error, "workspace input tensor is too small");
        return 0;
    }
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].kind == NIR_NODE_DELAY &&
            workspace->delay_primed[i] &&
            !nir_tensor_copy_into(
                &workspace->values[i], &workspace->delay_buffer[i])) {
            set_error(error, "workspace delay tensor is too small");
            return 0;
        }
    }
    for (size_t o = 0; o < graph->node_count; o++) {
        size_t index = rt->order[o];
        nir_node_kind_t kind = graph->nodes[index].kind;
        if (kind == NIR_NODE_INPUT || kind == NIR_NODE_DELAY) {
            continue;
        }
        if (!sum_incoming_workspace(rt, workspace, index, error) ||
            !apply_node_workspace(
                &graph->nodes[index], workspace->sum, workspace, index,
                error
            )) {
            return 0;
        }
    }
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].kind != NIR_NODE_DELAY) {
            continue;
        }
        if (!sum_incoming_workspace(rt, workspace, i, error) ||
            !nir_tensor_copy_into(
                &workspace->delay_buffer[i], workspace->sum
            )) {
            set_error(error, "workspace delay tensor is too small");
            return 0;
        }
        workspace->delay_primed[i] = 1;
    }
    return 1;
}

int nir_runtime_run(
    nir_runtime_t *rt,
    const nir_tensor_t *frames,
    size_t num_steps,
    nir_tensor_t *readout_out,
    char **error
) {
    if (rt == NULL || rt->graph == NULL || rt->graph->nodes == NULL ||
        rt->graph->node_count == 0 ||
        rt->graph->input_index >= rt->graph->node_count ||
        rt->graph->output_index >= rt->graph->node_count ||
        rt->order == NULL || rt->li_state == NULL || rt->li_primed == NULL ||
        rt->delay_buffer == NULL || rt->delay_primed == NULL || frames == NULL ||
        readout_out == NULL || num_steps == 0) {
        set_error(error, num_steps == 0 ? "no steps to run" :
                  "invalid runtime arguments");
        return 0;
    }
    const nir_graph_t *graph = rt->graph;
    nir_tensor_t total;
    memset(&total, 0, sizeof(total));
    int have_total = 0;

    for (size_t t = 0; t < num_steps; t++) {
        nir_tensor_t *values = (nir_tensor_t *)calloc(
            graph->node_count == 0 ? 1 : graph->node_count,
            sizeof(nir_tensor_t)
        );
        if (values == NULL) {
            nir_tensor_free(&total);
            set_error(error, "out of memory");
            return 0;
        }
        if (!run_step(rt, &frames[t], values, error)) {
            free_values(values, graph->node_count);
            nir_tensor_free(&total);
            return 0;
        }
        if (!have_total) {
            total = nir_tensor_clone(&values[graph->output_index]);
            if (total.data == NULL) {
                free_values(values, graph->node_count);
                set_error(error, "out of memory or invalid output tensor");
                return 0;
            }
            have_total = 1;
        } else {
            if (!nir_tensor_add_inplace(&total, &values[graph->output_index])) {
                free_values(values, graph->node_count);
                nir_tensor_free(&total);
                set_error(error, "runtime output shape changed");
                return 0;
            }
        }
        free_values(values, graph->node_count);
    }

    if (!have_total || num_steps == 0) {
        set_error(error, "no steps to run");
        nir_tensor_free(&total);
        return 0;
    }
    for (size_t i = 0; i < total.size; i++) {
        total.data[i] /= (float)num_steps;
    }
    *readout_out = total;
    return 1;
}

void nir_runtime_workspace_reset(nir_runtime_workspace_t *workspace) {
    if (workspace == NULL || workspace->li_primed == NULL ||
        workspace->delay_primed == NULL) {
        return;
    }
    for (size_t i = 0; i < workspace->node_capacity; i++) {
        workspace->li_primed[i] = 0;
        workspace->delay_primed[i] = 0;
    }
}

int nir_runtime_run_workspace(
    nir_runtime_t *rt,
    const nir_tensor_t *frames,
    size_t num_steps,
    nir_runtime_workspace_t *workspace,
    nir_tensor_t *readout_out,
    char **error
) {
    if (rt == NULL || rt->graph == NULL || rt->graph->nodes == NULL ||
        rt->graph->node_count == 0 ||
        rt->graph->input_index >= rt->graph->node_count ||
        rt->graph->output_index >= rt->graph->node_count ||
        rt->order == NULL || frames == NULL) {
        set_error(error, "invalid runtime arguments");
        return 0;
    }
    if (num_steps == 0) {
        set_error(error, "no steps to run");
        return 0;
    }
    if (!workspace_ready(rt, workspace, readout_out, error)) {
        return 0;
    }
    int have_total = 0;
    for (size_t t = 0; t < num_steps; t++) {
        if (!run_step_workspace(rt, &frames[t], workspace, error)) {
            return 0;
        }
        const nir_tensor_t *output =
            &workspace->values[rt->graph->output_index];
        if (!have_total) {
            if (!nir_tensor_copy_into(workspace->total, output)) {
                set_error(error, "workspace total tensor is too small");
                return 0;
            }
            have_total = 1;
        } else if (!nir_tensor_add_inplace(workspace->total, output)) {
            set_error(error, "workspace output shape changed");
            return 0;
        }
    }
    for (size_t i = 0; i < workspace->total->size; i++) {
        workspace->total->data[i] /= (float)num_steps;
    }
    if (!nir_tensor_copy_into(readout_out, workspace->total)) {
        set_error(error, "workspace readout tensor is too small");
        return 0;
    }
    return 1;
}
