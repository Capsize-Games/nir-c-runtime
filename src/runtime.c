#include "nir_c_runtime/runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/ops.h"

static void set_error(char **error, const char *fmt, const char *arg) {
    if (error == NULL || *error != NULL) {
        return;
    }
    size_t length = strlen(fmt) + (arg != NULL ? strlen(arg) : 0) + 8;
    *error = (char *)malloc(length);
    if (*error != NULL) {
        snprintf(*error, length, fmt, arg != NULL ? arg : "");
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
        set_error(error, "out of memory", NULL);
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
        set_error(
            error, "NIR graph cycle is not broken by a Delay node", NULL
        );
        return 0;
    }
    return 1;
}

int nir_runtime_init(
    nir_runtime_t *rt, const nir_graph_t *graph, char **error
) {
    memset(rt, 0, sizeof(*rt));
    rt->graph = graph;
    size_t n = graph->node_count;
    rt->order = (size_t *)malloc((n == 0 ? 1 : n) * sizeof(size_t));
    rt->li_state = (nir_tensor_t *)calloc(
        n == 0 ? 1 : n, sizeof(nir_tensor_t)
    );
    rt->li_primed = (int *)calloc(n == 0 ? 1 : n, sizeof(int));
    rt->delay_buffer = (nir_tensor_t *)calloc(
        n == 0 ? 1 : n, sizeof(nir_tensor_t)
    );
    rt->delay_primed = (int *)calloc(n == 0 ? 1 : n, sizeof(int));
    if (rt->order == NULL || rt->li_state == NULL || rt->li_primed == NULL ||
        rt->delay_buffer == NULL || rt->delay_primed == NULL) {
        nir_runtime_free(rt);
        set_error(error, "out of memory", NULL);
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
            if (rt->delay_buffer != NULL) {
                nir_tensor_free(&rt->delay_buffer[i]);
            }
        }
    }
    free(rt->order);
    free(rt->li_state);
    free(rt->li_primed);
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
            have_any = 1;
        } else if (!nir_tensor_add_inplace(out, &values[src])) {
            set_error(error, "shape mismatch summing node inputs", NULL);
            return 0;
        }
    }
    if (!found_edge) {
        set_error(
            error, "node %s received no inputs",
            graph->nodes[target_index].name
        );
        return 0;
    }
    if (!have_any) {
        size_t one[1] = {1};
        *out = nir_tensor_alloc(one, 1); /* zero-initialized scalar */
    }
    return 1;
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
        case NIR_NODE_AFFINE:
            return nir_op_affine(&node->params.affine, input);
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
            nir_tensor_free(&rt->li_state[index]);
            rt->li_state[index] = nir_tensor_clone(&membrane);
            rt->li_primed[index] = 1;
            return membrane;
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
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].kind == NIR_NODE_DELAY && rt->delay_primed[i]) {
            values[i] = nir_tensor_clone(&rt->delay_buffer[i]);
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

int nir_runtime_run(
    nir_runtime_t *rt,
    const nir_tensor_t *frames,
    size_t num_steps,
    nir_tensor_t *readout_out,
    char **error
) {
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
            set_error(error, "out of memory", NULL);
            return 0;
        }
        if (!run_step(rt, &frames[t], values, error)) {
            free_values(values, graph->node_count);
            nir_tensor_free(&total);
            return 0;
        }
        if (!have_total) {
            total = nir_tensor_clone(&values[graph->output_index]);
            have_total = 1;
        } else {
            nir_tensor_add_inplace(&total, &values[graph->output_index]);
        }
        free_values(values, graph->node_count);
    }

    if (!have_total || num_steps == 0) {
        set_error(error, "no steps to run", NULL);
        nir_tensor_free(&total);
        return 0;
    }
    for (size_t i = 0; i < total.size; i++) {
        total.data[i] /= (float)num_steps;
    }
    *readout_out = total;
    return 1;
}
