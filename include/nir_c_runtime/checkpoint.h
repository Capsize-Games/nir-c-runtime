/* Torch-free checkpoint save/load plus the reverse bridge (#29, #30),
 * with optimizer-state resume (#43).
 *
 * Checkpoints are version-stamped JSON (mirroring the
 * `spikeforge-nir-graph` v1 envelope discipline), currently version 2:
 *
 *   {"format": "spikeforge-c-checkpoint", "version": 2,
 *    "topology": "conv_net", "seed": 7, "step": 100,
 *    "nodes": [{"name": "fc", "kind": "Affine",
 *               "shape": [4, 8], "weight": [...], "bias": [...]}, ...],
 *    "optimizers": {"fc": {"kind": "adam", "step": 10,
 *                          "m": [...], "v": [...], "step_bias": 10,
 *                          "m_bias": [...], "v_bias": [...]}, ...}}
 *
 * Weights are flat row-major float arrays with an explicit shape;
 * floats print with %.9g, which round-trips float32 exactly. Unknown
 * format versions fail with a named error, as do missing nodes,
 * kind/shape mismatches, and non-finite weights -- never a silent
 * reinterpretation.
 *
 * The `optimizers` section (new in v2) is optional and keyed by node
 * name. Each entry carries one node's optimizer state: `kind` is
 * "sgd" (with `velocity` plus optional `velocity_bias`) or "adam"
 * (with `step`/`m`/`v` plus optional `step_bias`/`m_bias`/`v_bias`
 * for the bias). Bias arrays are omitted when the node has no bias
 * or the bias is untrained. Hyperparameters (lr, momentum, betas,
 * eps, weight decay) are NOT persisted: re-init optimizers with the
 * same hyperparameters before load for a bit-identical resume.
 *
 * Version 1 files (no `optimizers` section) still load: weights
 * restore normally and caller optimizers keep their current state
 * (fresh when newly init). The plain dumps/save/loads/load entry
 * points persist weights only, ignoring any `optimizers` section on
 * load; the with_opt variants persist both. Every optimizer entry
 * in the file must match a caller entry, else load fails rather
 * than silently restarting that node's optimizer; caller entries
 * absent from the file keep their state.
 *
 * The JSON writer here is new code (the existing json module stays
 * parse-only, owned by another worker). The weight setter mutates
 * the graph's public param buffers in place, which is also the
 * mutation path a future C optimizer would use.
 *
 * `nir_graph_export_summary` is the C half of the reverse bridge:
 * it re-emits a loaded graph as `graph_summary`-shaped JSON with
 * current (possibly C-trained) weights, which `nir_graph_load`
 * accepts and `tools/patch_nir_weights.py` can merge into a
 * Python-side summary.
 */
#ifndef NIR_C_RUNTIME_CHECKPOINT_H
#define NIR_C_RUNTIME_CHECKPOINT_H

#include <stddef.h>
#include <stdint.h>

#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/train.h"

#define NIR_CHECKPOINT_FORMAT "spikeforge-c-checkpoint"
#define NIR_CHECKPOINT_VERSION 2
#define NIR_CHECKPOINT_VERSION_1 1

/* Growing JSON text builder. */
typedef struct nir_json_writer nir_json_writer_t;

nir_json_writer_t *nir_json_writer_new(void);
void nir_json_writer_free(nir_json_writer_t *writer);

/* The finished document; valid until the next call or free. NULL
 * after any failed call. */
const char *nir_json_writer_data(const nir_json_writer_t *writer);
size_t nir_json_writer_length(const nir_json_writer_t *writer);

int nir_json_begin_object(nir_json_writer_t *writer);
int nir_json_begin_array(nir_json_writer_t *writer);
/* Close the innermost container. Fails when unbalanced. */
int nir_json_end(nir_json_writer_t *writer);
int nir_json_key(nir_json_writer_t *writer, const char *key);
int nir_json_string(nir_json_writer_t *writer, const char *value);
int nir_json_float(nir_json_writer_t *writer, float value);
int nir_json_double(nir_json_writer_t *writer, double value);
int nir_json_int(nir_json_writer_t *writer, long long value);
int nir_json_bool(nir_json_writer_t *writer, int value);
int nir_json_null(nir_json_writer_t *writer);

typedef struct {
    char *topology; /* malloc'd, NULL when the file omits it */
    uint64_t seed;
    unsigned long step;
} nir_checkpoint_meta_t;

void nir_checkpoint_meta_free(nir_checkpoint_meta_t *meta);

/* Serialize `graph`'s weights plus `meta` (NULL for weights only)
 * to a malloc'd JSON document. Weights only: any optimizer state
 * is omitted (see the with_opt variants). */
char *nir_checkpoint_dumps(
    const nir_graph_t *graph, const nir_checkpoint_meta_t *meta,
    char **error
);

/* Write the same document to `path`. */
int nir_checkpoint_save(
    const char *path, const nir_graph_t *graph,
    const nir_checkpoint_meta_t *meta, char **error
);

/* Load weights from a checkpoint document into an already-loaded
 * graph: every weighted node must be present with matching kind
 * and shape. Fills `meta` (NULL to skip; caller frees with
 * `nir_checkpoint_meta_free`). Weights only: any `optimizers`
 * section in the document is ignored. */
int nir_checkpoint_loads(
    const char *text, nir_graph_t *graph, nir_checkpoint_meta_t *meta,
    char **error
);

/* Load weights from the checkpoint file at `path`. */
int nir_checkpoint_load(
    const char *path, nir_graph_t *graph, nir_checkpoint_meta_t *meta,
    char **error
);

/* One node's optimizer state for checkpoint save/load.
 *
 * `node` borrows the graph node name. `kind` selects which
 * pointers are live: SGD uses `sgd_weight` (required) and
 * `sgd_bias` (NULL when the node has no bias or the bias is
 * untrained); Adam uses `adam_weight`/`adam_bias` the same way.
 * The inactive kind's pointers must be NULL. Weight/bias
 * optimizers are separate `train.h` instances stepped per buffer;
 * Adam weight/bias steps may differ and persist separately. */
typedef enum {
    NIR_CHECKPOINT_OPT_SGD,
    NIR_CHECKPOINT_OPT_ADAM
} nir_checkpoint_opt_kind_t;

typedef struct {
    const char *node;
    nir_checkpoint_opt_kind_t kind;
    nir_sgd_t *sgd_weight;
    nir_sgd_t *sgd_bias;
    nir_adam_t *adam_weight;
    nir_adam_t *adam_bias;
} nir_checkpoint_opt_t;

/* Serialize weights plus `opt_count` optimizer states. NULL/0
 * writes weights only (same document as `nir_checkpoint_dumps`).
 * Fails on unknown/unweighted/duplicate nodes, buffer counts that
 * do not match the graph, mixed-kind entries, and non-finite
 * state. */
char *nir_checkpoint_dumps_with_opt(
    const nir_graph_t *graph, const nir_checkpoint_meta_t *meta,
    const nir_checkpoint_opt_t *opts, size_t opt_count,
    char **error
);

/* Write the same document to `path`. */
int nir_checkpoint_save_with_opt(
    const char *path, const nir_graph_t *graph,
    const nir_checkpoint_meta_t *meta,
    const nir_checkpoint_opt_t *opts, size_t opt_count,
    char **error
);

/* Load weights plus optimizer state: every `optimizers` entry in
 * the document must match a caller entry (by node name and kind),
 * else load fails instead of silently restarting that optimizer.
 * Caller entries absent from the document (including all of them
 * for a v1 file) keep their current state. NULL/0 restores weights
 * only, ignoring any `optimizers` section. */
int nir_checkpoint_loads_with_opt(
    const char *text, nir_graph_t *graph, nir_checkpoint_meta_t *meta,
    nir_checkpoint_opt_t *opts, size_t opt_count,
    char **error
);

/* Load weights plus optimizer state from the file at `path`. */
int nir_checkpoint_load_with_opt(
    const char *path, nir_graph_t *graph, nir_checkpoint_meta_t *meta,
    nir_checkpoint_opt_t *opts, size_t opt_count,
    char **error
);

/* Point `*weight`/`*bias` at node `name`'s buffers with element
 * counts (`*bias` NULL, `*bias_count` 0 when the node has no bias).
 * Returns 0 for unknown nodes and nodes without weights. */
int nir_graph_get_weights(
    const nir_graph_t *graph, const char *name, const float **weight,
    size_t *weight_count, const float **bias, size_t *bias_count
);

/* Copy `weight_count`/`bias_count` floats into node `name`'s
 * buffers. Weight counts must match exactly; pass `bias` NULL to
 * leave the bias unchanged (nodes without a bias reject bias
 * data). Returns 0 and sets *error on any mismatch. */
int nir_graph_set_weights(
    nir_graph_t *graph, const char *name, const float *weight,
    size_t weight_count, const float *bias, size_t bias_count,
    char **error
);

/* Re-emit `graph` as `graph_summary`-shaped JSON (nodes with kind
 * params, edges with delayed flags) carrying the current weights.
 * Returns a malloc'd document, or NULL with *error set. */
char *nir_graph_export_summary(
    const nir_graph_t *graph, char **error
);

#endif /* NIR_C_RUNTIME_CHECKPOINT_H */
