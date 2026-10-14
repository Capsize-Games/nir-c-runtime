/* Graph-bound STDP: apply the train.h local-learning rule to a
 * loaded graph's weights (#48).
 *
 * `nir_stdp_t` keeps its own weight matrix in [in][out] pre-major
 * layout, while graph Affine/Linear nodes store [out][in]
 * out-major weights -- so manual wiring through
 * `nir_graph_get_weights`/`nir_graph_set_weights` alone would
 * silently transpose the matrix. This adapter is the documented,
 * tested path instead: it binds to one named node, loads that
 * node's current weights (transposed) into an owned STDP instance,
 * and writes every step's update back through
 * `nir_graph_set_weights` (biases are left unchanged).
 *
 * Only Affine and Linear nodes bind. Conv2d/Conv1d carry 3D/4D
 * kernels with no agreed flattening into the STDP rule's 2D
 * pre/post matrix, so they fail with a named error until that
 * mapping is decided.
 *
 * Intended on-chip local-learning flow: run the graph with
 * `nir_runtime_run`, record each step's input spikes to the bound
 * node (pre) and output spikes from the node or its downstream
 * neuron (post), pass those vectors to `nir_stdp_graph_step`, and
 * persist the trained weights with `nir_checkpoint_save`. The
 * binding owns the STDP-side copy: bind snapshots the graph's
 * weights, and each step overwrites the graph's weights with the
 * update. The graph must outlive the binding.
 */
#ifndef NIR_C_RUNTIME_STDP_GRAPH_H
#define NIR_C_RUNTIME_STDP_GRAPH_H

#include <stddef.h>

#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/train.h"

typedef struct {
    nir_stdp_t stdp;
    nir_graph_t *graph; /* borrowed; must outlive the binding */
    char *node;         /* bound node name, malloc'd */
    float *staging;     /* [out][in] write-back buffer */
    size_t in_features;
    size_t out_features;
} nir_stdp_graph_t;

/* Bind `node` (Affine or Linear) in `graph`, snapshotting its
 * current weights. Hyperparameters follow `nir_stdp_init` (taus
 * finite and > 0, w_min <= w_max); the dimensions come from the
 * node. Returns 0 and sets *error on unknown/unweighted/
 * non-2D nodes and on invalid hyperparameters. */
int nir_stdp_graph_bind(
    nir_stdp_graph_t *b, nir_graph_t *graph, const char *node,
    float tau_pre, float tau_post, float a_plus, float a_minus,
    float w_min, float w_max, char **error
);

/* One STDP step from this step's pre/post spike vectors (lengths
 * in_features/out_features), written back to the graph through
 * `nir_graph_set_weights`. Returns 0 on NULL or non-finite
 * spikes. */
int nir_stdp_graph_step(
    nir_stdp_graph_t *b, const float *pre_spikes,
    const float *post_spikes
);

/* Free the STDP state, node name, and staging buffer, and zero
 * `b` out. Safe on an already-freed or zeroed binding. */
void nir_stdp_graph_free(nir_stdp_graph_t *b);

#endif /* NIR_C_RUNTIME_STDP_GRAPH_H */
