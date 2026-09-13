/* Executes a loaded nir_graph_t over T timesteps.
 *
 * Mirrors spikeforge's reference interpreter's discretization exactly
 * (spikeforge/nir_bridge/interpreter.py): a Delay node's value for
 * step t is whatever its sources summed to at step t-1 (zero/absent
 * before its first step, contributing nothing to a sum rather than a
 * broadcast zero -- see tensor.h); every other node is evaluated once
 * per step in an order where an edge into a Delay node does not
 * constrain scheduling, since the Delay already holds a value before
 * the step begins.
 */
#ifndef NIR_C_RUNTIME_RUNTIME_H
#define NIR_C_RUNTIME_RUNTIME_H

#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/tensor.h"

typedef struct {
    const nir_graph_t *graph;
    size_t *order; /* topological order, node indices, length node_count */
    nir_tensor_t *li_state;    /* previous membrane, valid for LI nodes */
    int *li_primed;
    nir_tensor_t *delay_buffer; /* buffered value, valid for Delay nodes */
    int *delay_primed;
} nir_runtime_t;

/* Caller-owned buffers for allocation-free graph execution. Every tensor
 * entry must be bound with `nir_tensor_bind` before the first run; entries
 * need capacity for the largest tensor that node can produce. The arrays and
 * flag buffers must each contain at least `runtime->graph->node_count`
 * entries. `sum` and `total` are scratch/output tensors. */
typedef struct {
    nir_tensor_t *values;
    nir_tensor_t *li_state;
    int *li_primed;
    nir_tensor_t *delay_buffer;
    int *delay_primed;
    nir_tensor_t *sum;
    nir_tensor_t *total;
    size_t node_capacity;
} nir_runtime_workspace_t;

/* Compute the graph's execution order and allocate per-node state.
 * `graph` must outlive `rt`. Returns 0 and sets *error on a graph the
 * runtime can't schedule (a cycle no Delay node breaks). */
int nir_runtime_init(nir_runtime_t *rt, const nir_graph_t *graph, char **error);

/* Free all state owned by `rt` (not the graph itself). */
void nir_runtime_free(nir_runtime_t *rt);

/* Run `frames[0..num_steps)` (each shaped exactly as the graph's Input
 * node's first consumer expects) through the graph and return the
 * time-averaged Output-node value in `readout_out` (caller frees it
 * with nir_tensor_free). Returns 0 and sets *error on failure. */
int nir_runtime_run(
    nir_runtime_t *rt,
    const nir_tensor_t *frames,
    size_t num_steps,
    nir_tensor_t *readout_out,
    char **error
);

/* Run using only caller-owned workspace storage after setup. `readout_out`
 * must be bound before the call. No allocation or deallocation occurs on the
 * successful path, and runtime state is retained across calls. */
int nir_runtime_run_workspace(
    nir_runtime_t *rt,
    const nir_tensor_t *frames,
    size_t num_steps,
    nir_runtime_workspace_t *workspace,
    nir_tensor_t *readout_out,
    char **error
);

/* Clear temporal flags in a workspace without releasing its buffers. */
void nir_runtime_workspace_reset(nir_runtime_workspace_t *workspace);

#endif /* NIR_C_RUNTIME_RUNTIME_H */
