#include "nir_c_runtime/stdp_graph.h"

#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/checkpoint.h"

static void set_error(char **error, const char *message) {
    if (error == NULL) return;
    *error = NULL;
    if (message == NULL) return;
    size_t n = strlen(message) + 1;
    char *copy = (char *)malloc(n);
    if (copy != NULL) memcpy(copy, message, n);
    *error = copy;
}

/* Out-major [o][i] -> pre-major [i][o]. */
static void to_premajor(
    const float *src, float *dst, size_t in, size_t out
) {
    for (size_t i = 0; i < in; i++) {
        for (size_t o = 0; o < out; o++) {
            dst[i * out + o] = src[o * in + i];
        }
    }
}

/* Pre-major [i][o] -> out-major [o][i]. */
static void to_outmajor(
    const float *src, float *dst, size_t in, size_t out
) {
    for (size_t o = 0; o < out; o++) {
        for (size_t i = 0; i < in; i++) {
            dst[o * in + i] = src[i * out + o];
        }
    }
}

int nir_stdp_graph_bind(
    nir_stdp_graph_t *b, nir_graph_t *graph, const char *node,
    float tau_pre, float tau_post, float a_plus, float a_minus,
    float w_min, float w_max, char **error
) {
    if (error != NULL) *error = NULL;
    if (b == NULL || graph == NULL || node == NULL) {
        set_error(error, "STDP bind needs a binding, graph, and node");
        return 0;
    }
    memset(b, 0, sizeof(*b));
    size_t index = nir_graph_find(graph, node);
    if (index == (size_t)-1) {
        set_error(error, "STDP bind names an unknown node");
        return 0;
    }
    const nir_node_t *found = &graph->nodes[index];
    size_t in = 0;
    size_t out = 0;
    if (found->kind == NIR_NODE_AFFINE) {
        in = found->params.affine.in_features;
        out = found->params.affine.out_features;
    } else if (found->kind == NIR_NODE_LINEAR) {
        in = found->params.linear.in_features;
        out = found->params.linear.out_features;
    } else {
        set_error(error, "STDP bind needs an Affine or Linear node");
        return 0;
    }
    if (!nir_stdp_init(
            &b->stdp, in, out, tau_pre, tau_post, a_plus, a_minus,
            w_min, w_max, error
        )) {
        memset(b, 0, sizeof(*b));
        return 0;
    }
    b->node = (char *)malloc(strlen(node) + 1);
    b->staging = (float *)calloc(in * out, sizeof(float));
    if (b->node == NULL || b->staging == NULL) {
        nir_stdp_graph_free(b);
        set_error(error, "out of memory");
        return 0;
    }
    strcpy(b->node, node);
    const float *weight = NULL;
    const float *bias = NULL;
    size_t wn = 0;
    size_t bn = 0;
    if (!nir_graph_get_weights(
            graph, node, &weight, &wn, &bias, &bn
        ) ||
        wn != in * out) {
        nir_stdp_graph_free(b);
        set_error(error, "STDP bind cannot read the node weights");
        return 0;
    }
    /* Staging doubles as the pre-major load buffer; set_weight
     * copies it into the STDP instance (and rejects non-finite
     * graph weights). */
    to_premajor(weight, b->staging, in, out);
    if (!nir_stdp_set_weight(&b->stdp, b->staging)) {
        nir_stdp_graph_free(b);
        set_error(error, "STDP bind found non-finite node weights");
        return 0;
    }
    b->graph = graph;
    b->in_features = in;
    b->out_features = out;
    return 1;
}

int nir_stdp_graph_step(
    nir_stdp_graph_t *b, const float *pre_spikes,
    const float *post_spikes
) {
    if (b == NULL || b->graph == NULL || b->node == NULL ||
        b->staging == NULL) {
        return 0;
    }
    if (!nir_stdp_step(&b->stdp, pre_spikes, post_spikes)) {
        return 0;
    }
    to_outmajor(
        b->stdp.weight, b->staging, b->in_features, b->out_features
    );
    return nir_graph_set_weights(
        b->graph, b->node, b->staging, b->in_features * b->out_features,
        NULL, 0, NULL
    );
}

void nir_stdp_graph_free(nir_stdp_graph_t *b) {
    if (b == NULL) return;
    nir_stdp_free(&b->stdp);
    free(b->node);
    free(b->staging);
    memset(b, 0, sizeof(*b));
}
