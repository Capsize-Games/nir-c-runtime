/* A loaded NIR graph: nodes with typed parameters, plus edges.
 *
 * Scoped to exactly the node kinds spikeforge's conv_net topology
 * emits (see the reference interpreter this mirrors:
 * spikeforge/nir_bridge/{ops_neuron,ops_linear}.py) -- Input, Output,
 * Conv2d, Affine, Flatten, AvgPool2d, SumPool2d, LI, Threshold,
 * Scale, Delay. LIF/IF/CubaLIF/Conv1d/Linear are not implemented;
 * loading a graph that uses them fails with a named error rather than
 * silently misinterpreting them.
 */
#ifndef NIR_C_RUNTIME_GRAPH_H
#define NIR_C_RUNTIME_GRAPH_H

#include <stddef.h>

#include "nir_c_runtime/json.h"

typedef enum {
    NIR_NODE_INPUT,
    NIR_NODE_OUTPUT,
    NIR_NODE_CONV2D,
    NIR_NODE_AFFINE,
    NIR_NODE_FLATTEN,
    NIR_NODE_AVGPOOL2D,
    NIR_NODE_SUMPOOL2D,
    NIR_NODE_LI,
    NIR_NODE_THRESHOLD,
    NIR_NODE_SCALE,
    NIR_NODE_DELAY
} nir_node_kind_t;

typedef struct {
    size_t out_channels, in_channels, kernel_h, kernel_w;
    float *weight; /* [out_channels][in_channels][kernel_h][kernel_w] */
    float *bias;   /* [out_channels], NULL if the node has no bias */
    int stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w;
} nir_conv2d_params_t;

typedef struct {
    size_t out_features, in_features;
    float *weight; /* [out_features][in_features] */
    float *bias;   /* [out_features], NULL if the node has no bias */
} nir_affine_params_t;

typedef struct {
    int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
} nir_pool2d_params_t;

typedef struct {
    float tau, r, v_leak;
} nir_li_params_t;

typedef struct {
    float threshold;
} nir_threshold_params_t;

typedef struct {
    float scale;
} nir_scale_params_t;

typedef struct {
    char *name;
    nir_node_kind_t kind;
    union {
        nir_conv2d_params_t conv2d;
        nir_affine_params_t affine;
        nir_pool2d_params_t pool2d;
        nir_li_params_t li;
        nir_threshold_params_t threshold;
        nir_scale_params_t scale;
    } params;
} nir_node_t;

typedef struct {
    size_t source_index;
    size_t target_index;
} nir_edge_t;

typedef struct {
    nir_node_t *nodes;
    size_t node_count;
    nir_edge_t *edges;
    size_t edge_count;
    size_t input_index;
    size_t output_index;
} nir_graph_t;

/* Parse a graph_summary()-shaped JSON value (top-level {"nodes": [...],
 * "edges": [...]}, matching spikeforge.nir_bridge.exporter.graph_summary)
 * into `out`. Returns 0 and sets *error to a malloc'd message
 * (caller-owned) on failure, including an unsupported node kind. */
int nir_graph_load(
    const json_value_t *summary, nir_graph_t *out, char **error
);

/* Free every node's owned buffers, the node/edge arrays, and zero
 * `graph` out. Safe on an already-freed or zeroed graph. */
void nir_graph_free(nir_graph_t *graph);

/* Return the index of the node named `name`, or SIZE_MAX if absent. */
size_t nir_graph_find(const nir_graph_t *graph, const char *name);

#endif /* NIR_C_RUNTIME_GRAPH_H */
