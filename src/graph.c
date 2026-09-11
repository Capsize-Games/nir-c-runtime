#include "nir_c_runtime/graph.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char **error, const char *fmt, const char *arg) {
    if (error == NULL || *error != NULL) {
        return; /* keep the first error, or the caller doesn't want one */
    }
    size_t length = strlen(fmt) + (arg != NULL ? strlen(arg) : 0) + 8;
    *error = (char *)malloc(length);
    if (*error != NULL) {
        snprintf(*error, length, fmt, arg != NULL ? arg : "");
    }
}

static char *dup_string(const char *text) {
    size_t length = strlen(text) + 1;
    char *copy = (char *)malloc(length);
    if (copy != NULL) {
        memcpy(copy, text, length);
    }
    return copy;
}

/* Flatten a 1D JSON number array into a newly malloc'd float buffer. */
static float *parse_floats_1d(const json_value_t *array, size_t *count) {
    size_t n = json_array_length(array);
    float *out = (float *)malloc((n == 0 ? 1 : n) * sizeof(float));
    if (out == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        out[i] = (float)json_as_number(json_array_get(array, i), 0.0);
    }
    *count = n;
    return out;
}

/* Flatten a 2D JSON number array ([rows][cols]) row-major. */
static float *parse_floats_2d(
    const json_value_t *array, size_t *rows, size_t *cols
) {
    size_t r = json_array_length(array);
    size_t c = r > 0 ? json_array_length(json_array_get(array, 0)) : 0;
    float *out = (float *)malloc((r * c == 0 ? 1 : r * c) * sizeof(float));
    if (out == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < r; i++) {
        const json_value_t *row = json_array_get(array, i);
        for (size_t j = 0; j < c; j++) {
            out[i * c + j] = (float)json_as_number(
                json_array_get(row, j), 0.0
            );
        }
    }
    *rows = r;
    *cols = c;
    return out;
}

/* Flatten a 4D JSON number array ([a][b][c][d]) row-major, the layout
 * a Conv2d weight tensor ([out_ch][in_ch][kh][kw]) needs. */
static float *parse_floats_4d(
    const json_value_t *array, size_t *d0, size_t *d1, size_t *d2, size_t *d3
) {
    size_t a = json_array_length(array);
    const json_value_t *l1 = a > 0 ? json_array_get(array, 0) : NULL;
    size_t b = json_array_length(l1);
    const json_value_t *l2 = b > 0 ? json_array_get(l1, 0) : NULL;
    size_t c = json_array_length(l2);
    const json_value_t *l3 = c > 0 ? json_array_get(l2, 0) : NULL;
    size_t d = json_array_length(l3);
    size_t total = a * b * c * d;
    float *out = (float *)malloc((total == 0 ? 1 : total) * sizeof(float));
    if (out == NULL) {
        return NULL;
    }
    size_t index = 0;
    for (size_t i = 0; i < a; i++) {
        const json_value_t *dim1 = json_array_get(array, i);
        for (size_t j = 0; j < b; j++) {
            const json_value_t *dim2 = json_array_get(dim1, j);
            for (size_t k = 0; k < c; k++) {
                const json_value_t *dim3 = json_array_get(dim2, k);
                for (size_t l = 0; l < d; l++) {
                    out[index++] = (float)json_as_number(
                        json_array_get(dim3, l), 0.0
                    );
                }
            }
        }
    }
    *d0 = a;
    *d1 = b;
    *d2 = c;
    *d3 = d;
    return out;
}

static int pair_at(const json_value_t *params, const char *key, int *a, int *b) {
    const json_value_t *pair = json_object_get(params, key);
    *a = (int)json_as_number(json_array_get(pair, 0), 0.0);
    *b = (int)json_as_number(json_array_get(pair, 1), 0.0);
    return pair != NULL;
}

static int load_conv2d(const json_value_t *params, nir_node_t *node) {
    nir_conv2d_params_t *p = &node->params.conv2d;
    const json_value_t *weight = json_object_get(params, "weight");
    p->weight = parse_floats_4d(
        weight, &p->out_channels, &p->in_channels, &p->kernel_h, &p->kernel_w
    );
    if (p->weight == NULL) {
        return 0;
    }
    const json_value_t *bias = json_object_get(params, "bias");
    if (bias != NULL) {
        size_t bias_count = 0;
        p->bias = parse_floats_1d(bias, &bias_count);
    } else {
        p->bias = NULL;
    }
    pair_at(params, "stride", &p->stride_h, &p->stride_w);
    pair_at(params, "padding", &p->padding_h, &p->padding_w);
    pair_at(params, "dilation", &p->dilation_h, &p->dilation_w);
    return 1;
}

static int load_affine(const json_value_t *params, nir_node_t *node) {
    nir_affine_params_t *p = &node->params.affine;
    const json_value_t *weight = json_object_get(params, "weight");
    p->weight = parse_floats_2d(weight, &p->out_features, &p->in_features);
    if (p->weight == NULL) {
        return 0;
    }
    size_t bias_count = 0;
    p->bias = parse_floats_1d(json_object_get(params, "bias"), &bias_count);
    return p->bias != NULL;
}

static void load_pool2d(const json_value_t *params, nir_node_t *node) {
    nir_pool2d_params_t *p = &node->params.pool2d;
    pair_at(params, "kernel_size", &p->kernel_h, &p->kernel_w);
    pair_at(params, "stride", &p->stride_h, &p->stride_w);
    pair_at(params, "padding", &p->padding_h, &p->padding_w);
}

static void load_li(const json_value_t *params, nir_node_t *node) {
    node->params.li.tau = (float)json_as_number(
        json_object_get(params, "tau"), 1.0
    );
    node->params.li.r = (float)json_as_number(
        json_object_get(params, "r"), 1.0
    );
    node->params.li.v_leak = (float)json_as_number(
        json_object_get(params, "v_leak"), 0.0
    );
}

typedef struct {
    const char *name;
    nir_node_kind_t kind;
} kind_entry_t;

static const kind_entry_t KIND_TABLE[] = {
    {"Input", NIR_NODE_INPUT},
    {"Output", NIR_NODE_OUTPUT},
    {"Conv2d", NIR_NODE_CONV2D},
    {"Affine", NIR_NODE_AFFINE},
    {"Flatten", NIR_NODE_FLATTEN},
    {"AvgPool2d", NIR_NODE_AVGPOOL2D},
    {"SumPool2d", NIR_NODE_SUMPOOL2D},
    {"LI", NIR_NODE_LI},
    {"Threshold", NIR_NODE_THRESHOLD},
    {"Scale", NIR_NODE_SCALE},
    {"Delay", NIR_NODE_DELAY},
};

static int resolve_kind(const char *text, nir_node_kind_t *kind) {
    size_t count = sizeof(KIND_TABLE) / sizeof(KIND_TABLE[0]);
    for (size_t i = 0; i < count; i++) {
        if (strcmp(KIND_TABLE[i].name, text) == 0) {
            *kind = KIND_TABLE[i].kind;
            return 1;
        }
    }
    return 0;
}

static int load_node(
    const json_value_t *node_json, nir_node_t *node, char **error
) {
    const char *name = json_as_string(json_object_get(node_json, "name"));
    const char *kind_text = json_as_string(
        json_object_get(node_json, "kind")
    );
    if (name == NULL || kind_text == NULL) {
        set_error(error, "node missing name or kind", NULL);
        return 0;
    }
    node->name = dup_string(name);
    if (!resolve_kind(kind_text, &node->kind)) {
        set_error(error, "unsupported node kind: %s", kind_text);
        return 0;
    }
    const json_value_t *params = json_object_get(node_json, "params");
    switch (node->kind) {
        case NIR_NODE_CONV2D: return load_conv2d(params, node);
        case NIR_NODE_AFFINE: return load_affine(params, node);
        case NIR_NODE_AVGPOOL2D:
        case NIR_NODE_SUMPOOL2D:
            load_pool2d(params, node);
            return 1;
        case NIR_NODE_LI:
            load_li(params, node);
            return 1;
        case NIR_NODE_THRESHOLD:
            node->params.threshold.threshold = (float)json_as_number(
                json_object_get(params, "threshold"), 1.0
            );
            return 1;
        case NIR_NODE_SCALE:
            node->params.scale.scale = (float)json_as_number(
                json_object_get(params, "scale"), 1.0
            );
            return 1;
        case NIR_NODE_INPUT:
        case NIR_NODE_OUTPUT:
        case NIR_NODE_FLATTEN:
        case NIR_NODE_DELAY:
            return 1;
        default:
            set_error(error, "unhandled node kind", NULL);
            return 0;
    }
}

static void free_node(nir_node_t *node) {
    free(node->name);
    if (node->kind == NIR_NODE_CONV2D) {
        free(node->params.conv2d.weight);
        free(node->params.conv2d.bias);
    } else if (node->kind == NIR_NODE_AFFINE) {
        free(node->params.affine.weight);
        free(node->params.affine.bias);
    }
    memset(node, 0, sizeof(*node));
}

size_t nir_graph_find(const nir_graph_t *graph, const char *name) {
    for (size_t i = 0; i < graph->node_count; i++) {
        if (strcmp(graph->nodes[i].name, name) == 0) {
            return i;
        }
    }
    return (size_t)-1;
}

static int load_edges(
    const json_value_t *edges_json, nir_graph_t *graph, char **error
) {
    size_t count = json_array_length(edges_json);
    graph->edges = (nir_edge_t *)malloc(
        (count == 0 ? 1 : count) * sizeof(nir_edge_t)
    );
    if (graph->edges == NULL) {
        return 0;
    }
    for (size_t i = 0; i < count; i++) {
        const json_value_t *edge = json_array_get(edges_json, i);
        const char *source = json_as_string(json_object_get(edge, "source"));
        const char *target = json_as_string(json_object_get(edge, "target"));
        size_t source_index = nir_graph_find(graph, source);
        size_t target_index = nir_graph_find(graph, target);
        if (source_index == (size_t)-1 || target_index == (size_t)-1) {
            set_error(error, "edge references an unknown node", NULL);
            return 0;
        }
        graph->edges[i].source_index = source_index;
        graph->edges[i].target_index = target_index;
    }
    graph->edge_count = count;
    return 1;
}

int nir_graph_load(
    const json_value_t *summary, nir_graph_t *out, char **error
) {
    memset(out, 0, sizeof(*out));
    const json_value_t *nodes_json = json_object_get(summary, "nodes");
    const json_value_t *edges_json = json_object_get(summary, "edges");
    if (nodes_json == NULL || edges_json == NULL) {
        set_error(error, "graph JSON missing \"nodes\" or \"edges\"", NULL);
        return 0;
    }
    size_t node_count = json_array_length(nodes_json);
    out->nodes = (nir_node_t *)calloc(
        node_count == 0 ? 1 : node_count, sizeof(nir_node_t)
    );
    if (out->nodes == NULL) {
        set_error(error, "out of memory", NULL);
        return 0;
    }
    out->node_count = node_count;
    for (size_t i = 0; i < node_count; i++) {
        if (!load_node(json_array_get(nodes_json, i), &out->nodes[i], error)) {
            nir_graph_free(out);
            return 0;
        }
        if (out->nodes[i].kind == NIR_NODE_INPUT) {
            out->input_index = i;
        } else if (out->nodes[i].kind == NIR_NODE_OUTPUT) {
            out->output_index = i;
        }
    }
    if (!load_edges(edges_json, out, error)) {
        nir_graph_free(out);
        return 0;
    }
    return 1;
}

void nir_graph_free(nir_graph_t *graph) {
    if (graph == NULL) {
        return;
    }
    for (size_t i = 0; i < graph->node_count; i++) {
        free_node(&graph->nodes[i]);
    }
    free(graph->nodes);
    free(graph->edges);
    memset(graph, 0, sizeof(*graph));
}
