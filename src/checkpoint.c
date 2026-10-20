#include "nir_c_runtime/checkpoint.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/json.h"
#include "nir_c_runtime/train.h"

struct nir_json_writer {
    char *text;
    size_t length;
    size_t capacity;
    char *stack; /* '{' / '[' per open container */
    size_t depth;
    size_t stack_capacity;
    int need_comma;
    int failed;
};

nir_json_writer_t *nir_json_writer_new(void) {
    nir_json_writer_t *writer =
        (nir_json_writer_t *)calloc(1, sizeof(*writer));
    return writer;
}

void nir_json_writer_free(nir_json_writer_t *writer) {
    if (writer == NULL) {
        return;
    }
    free(writer->text);
    free(writer->stack);
    free(writer);
}

const char *nir_json_writer_data(const nir_json_writer_t *writer) {
    if (writer == NULL || writer->failed || writer->text == NULL) {
        return NULL;
    }
    return writer->text;
}

size_t nir_json_writer_length(const nir_json_writer_t *writer) {
    if (writer == NULL || writer->failed) {
        return 0;
    }
    return writer->length;
}

static int writer_reserve(nir_json_writer_t *w, size_t extra) {
    if (w->failed) {
        return 0;
    }
    if (w->length + extra + 1 <= w->capacity) {
        return 1;
    }
    size_t capacity = w->capacity == 0 ? 256 : w->capacity;
    while (w->length + extra + 1 > capacity) {
        capacity *= 2;
    }
    char *text = (char *)realloc(w->text, capacity);
    if (text == NULL) {
        w->failed = 1;
        return 0;
    }
    w->text = text;
    w->capacity = capacity;
    return 1;
}

static int writer_put(nir_json_writer_t *w, const char *s, size_t n) {
    if (!writer_reserve(w, n)) {
        return 0;
    }
    memcpy(w->text + w->length, s, n);
    w->length += n;
    w->text[w->length] = '\0';
    return 1;
}

static int writer_byte(nir_json_writer_t *w, char c) {
    return writer_put(w, &c, 1);
}

/* Emit a separator (or the object colon handling lives in key). */
static int writer_value(nir_json_writer_t *w) {
    if (w->failed) {
        return 0;
    }
    if (w->need_comma && !writer_byte(w, ',')) {
        return 0;
    }
    w->need_comma = 1;
    return 1;
}

static int writer_push(nir_json_writer_t *w, char kind, char open) {
    if (!writer_value(w) || !writer_byte(w, open)) {
        return 0;
    }
    if (w->depth == w->stack_capacity) {
        size_t capacity = w->stack_capacity == 0 ? 8 : w->stack_capacity * 2;
        char *stack = (char *)realloc(w->stack, capacity);
        if (stack == NULL) {
            w->failed = 1;
            return 0;
        }
        w->stack = stack;
        w->stack_capacity = capacity;
    }
    w->stack[w->depth++] = kind;
    w->need_comma = 0;
    return 1;
}

int nir_json_begin_object(nir_json_writer_t *writer) {
    return writer != NULL && writer_push(writer, '{', '{');
}

int nir_json_begin_array(nir_json_writer_t *writer) {
    return writer != NULL && writer_push(writer, '[', '[');
}

int nir_json_end(nir_json_writer_t *writer) {
    if (writer == NULL || writer->failed || writer->depth == 0) {
        if (writer != NULL) {
            writer->failed = 1;
        }
        return 0;
    }
    char kind = writer->stack[--writer->depth];
    writer->need_comma = 1;
    return writer_byte(writer, kind == '{' ? '}' : ']');
}

int nir_json_key(nir_json_writer_t *writer, const char *key) {
    if (writer == NULL || key == NULL || writer->failed ||
        writer->depth == 0 || writer->stack[writer->depth - 1] != '{') {
        if (writer != NULL) {
            writer->failed = 1;
        }
        return 0;
    }
    if (writer->need_comma && !writer_byte(writer, ',')) {
        return 0;
    }
    writer->need_comma = 0;
    if (!nir_json_string(writer, key)) {
        return 0;
    }
    writer->need_comma = 0;
    return writer_byte(writer, ':');
}

int nir_json_string(nir_json_writer_t *writer, const char *value) {
    static const char *escapes[32] = {
        "\\u0000", "\\u0001", "\\u0002", "\\u0003",
        "\\u0004", "\\u0005", "\\u0006", "\\u0007",
        "\\b", "\\t", "\\n", "\\u000b",
        "\\f", "\\r", "\\u000e", "\\u000f",
        "\\u0010", "\\u0011", "\\u0012", "\\u0013",
        "\\u0014", "\\u0015", "\\u0016", "\\u0017",
        "\\u0018", "\\u0019", "\\u001a", "\\u001b",
        "\\u001c", "\\u001d", "\\u001e", "\\u001f",
    };
    if (writer == NULL || value == NULL || !writer_value(writer) ||
        !writer_byte(writer, '"')) {
        return 0;
    }
    for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
        if (*p == '"' || *p == '\\') {
            if (!writer_byte(writer, '\\') || !writer_byte(writer, *p)) {
                return 0;
            }
        } else if (*p < 32) {
            const char *esc = escapes[*p];
            if (!writer_put(writer, esc, strlen(esc))) {
                return 0;
            }
        } else if (!writer_byte(writer, *p)) {
            return 0;
        }
    }
    writer->need_comma = 1;
    return writer_byte(writer, '"');
}

static int writer_number(nir_json_writer_t *w, const char *text) {
    return writer_value(w) && writer_put(w, text, strlen(text));
}

int nir_json_float(nir_json_writer_t *writer, float value) {
    char text[32];
    if (writer == NULL) {
        return 0;
    }
    if (!isfinite(value)) {
        return nir_json_null(writer);
    }
    snprintf(text, sizeof(text), "%.9g", value);
    return writer_number(writer, text);
}

int nir_json_double(nir_json_writer_t *writer, double value) {
    char text[32];
    if (writer == NULL) {
        return 0;
    }
    if (!isfinite(value)) {
        return nir_json_null(writer);
    }
    snprintf(text, sizeof(text), "%.17g", value);
    return writer_number(writer, text);
}

int nir_json_int(nir_json_writer_t *writer, long long value) {
    char text[32];
    if (writer == NULL) {
        return 0;
    }
    snprintf(text, sizeof(text), "%lld", value);
    return writer_number(writer, text);
}

int nir_json_bool(nir_json_writer_t *writer, int value) {
    if (writer == NULL) {
        return 0;
    }
    return writer_number(writer, value ? "true" : "false");
}

int nir_json_null(nir_json_writer_t *writer) {
    if (writer == NULL) {
        return 0;
    }
    return writer_number(writer, "null");
}

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

void nir_checkpoint_meta_free(nir_checkpoint_meta_t *meta) {
    if (meta == NULL) {
        return;
    }
    free(meta->topology);
    memset(meta, 0, sizeof(*meta));
}

/* Row-major element counts per weighted kind; 0 weight means the
 * kind carries no weights. */
static int weight_shapes(
    const nir_node_t *node, const float **weight, size_t *wn,
    const float **bias, size_t *bn
) {
    switch (node->kind) {
        case NIR_NODE_CONV2D: {
            const nir_conv2d_params_t *p = &node->params.conv2d;
            *weight = p->weight;
            *wn = p->out_channels * p->in_channels * p->kernel_h *
                p->kernel_w;
            *bias = p->bias;
            *bn = p->bias != NULL ? p->out_channels : 0;
            return 1;
        }
        case NIR_NODE_AFFINE: {
            const nir_affine_params_t *p = &node->params.affine;
            *weight = p->weight;
            *wn = p->out_features * p->in_features;
            *bias = p->bias;
            *bn = p->bias != NULL ? p->out_features : 0;
            return 1;
        }
        case NIR_NODE_CONV1D: {
            const nir_conv1d_params_t *p = &node->params.conv1d;
            *weight = p->weight;
            *wn = p->out_channels * p->in_channels * p->kernel;
            *bias = p->bias;
            *bn = p->bias != NULL ? p->out_channels : 0;
            return 1;
        }
        case NIR_NODE_LINEAR: {
            const nir_linear_params_t *p = &node->params.linear;
            *weight = p->weight;
            *wn = p->out_features * p->in_features;
            *bias = NULL;
            *bn = 0;
            return 1;
        }
        default:
            return 0;
    }
}

int nir_graph_get_weights(
    const nir_graph_t *graph, const char *name, const float **weight,
    size_t *weight_count, const float **bias, size_t *bias_count
) {
    if (graph == NULL || name == NULL || weight == NULL ||
        weight_count == NULL || bias == NULL || bias_count == NULL) {
        return 0;
    }
    size_t index = nir_graph_find(graph, name);
    if (index == (size_t)-1) {
        return 0;
    }
    return weight_shapes(
        &graph->nodes[index], weight, weight_count, bias, bias_count
    );
}

int nir_graph_set_weights(
    nir_graph_t *graph, const char *name, const float *weight,
    size_t weight_count, const float *bias, size_t bias_count,
    char **error
) {
    if (graph == NULL || name == NULL) {
        set_error(error, "weight update needs a graph and node name");
        return 0;
    }
    size_t index = nir_graph_find(graph, name);
    if (index == (size_t)-1) {
        set_error(error, "weight update names an unknown node");
        return 0;
    }
    const float *dst_w = NULL;
    const float *dst_b = NULL;
    size_t want_w = 0;
    size_t want_b = 0;
    if (!weight_shapes(
            &graph->nodes[index], &dst_w, &want_w, &dst_b, &want_b
        )) {
        set_error(error, "weight update targets an unweighted node");
        return 0;
    }
    if (weight == NULL || weight_count != want_w) {
        set_error(error, "weight update has the wrong weight count");
        return 0;
    }
    if (bias == NULL) {
        if (bias_count != 0) {
            set_error(error, "weight update has an inconsistent bias");
            return 0;
        }
    } else if (dst_b == NULL || bias_count != want_b) {
        set_error(error, "weight update has the wrong bias count");
        return 0;
    }
    memcpy((float *)dst_w, weight, want_w * sizeof(float));
    if (bias != NULL) {
        memcpy((float *)dst_b, bias, want_b * sizeof(float));
    }
    return 1;
}

static const char *kind_name(nir_node_kind_t kind) {
    switch (kind) {
        case NIR_NODE_INPUT: return "Input";
        case NIR_NODE_OUTPUT: return "Output";
        case NIR_NODE_CONV2D: return "Conv2d";
        case NIR_NODE_AFFINE: return "Affine";
        case NIR_NODE_FLATTEN: return "Flatten";
        case NIR_NODE_AVGPOOL2D: return "AvgPool2d";
        case NIR_NODE_SUMPOOL2D: return "SumPool2d";
        case NIR_NODE_LI: return "LI";
        case NIR_NODE_THRESHOLD: return "Threshold";
        case NIR_NODE_SCALE: return "Scale";
        case NIR_NODE_DELAY: return "Delay";
        case NIR_NODE_LIF: return "LIF";
        case NIR_NODE_IF: return "IF";
        case NIR_NODE_CUBALIF: return "CubaLIF";
        case NIR_NODE_CONV1D: return "Conv1d";
        case NIR_NODE_LINEAR: return "Linear";
        case NIR_NODE_RELU: return "ReLU";
        default: return NULL;
    }
}

static int write_floats(
    nir_json_writer_t *w, const float *values, size_t n
) {
    if (!nir_json_begin_array(w)) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(values[i]) || !nir_json_float(w, values[i])) {
            return 0;
        }
    }
    return nir_json_end(w);
}

/* Explicit shape vector per weighted kind, matching the loader's
 * dimension order. */
static int write_shape(nir_json_writer_t *w, const nir_node_t *node) {
    if (!nir_json_key(w, "shape") || !nir_json_begin_array(w)) {
        return 0;
    }
    long long dims[4];
    size_t ndim = 0;
    switch (node->kind) {
        case NIR_NODE_CONV2D: {
            const nir_conv2d_params_t *p = &node->params.conv2d;
            dims[0] = (long long)p->out_channels;
            dims[1] = (long long)p->in_channels;
            dims[2] = (long long)p->kernel_h;
            dims[3] = (long long)p->kernel_w;
            ndim = 4;
            break;
        }
        case NIR_NODE_AFFINE: {
            const nir_affine_params_t *p = &node->params.affine;
            dims[0] = (long long)p->out_features;
            dims[1] = (long long)p->in_features;
            ndim = 2;
            break;
        }
        case NIR_NODE_CONV1D: {
            const nir_conv1d_params_t *p = &node->params.conv1d;
            dims[0] = (long long)p->out_channels;
            dims[1] = (long long)p->in_channels;
            dims[2] = (long long)p->kernel;
            ndim = 3;
            break;
        }
        case NIR_NODE_LINEAR: {
            const nir_linear_params_t *p = &node->params.linear;
            dims[0] = (long long)p->out_features;
            dims[1] = (long long)p->in_features;
            ndim = 2;
            break;
        }
        default:
            return 0;
    }
    for (size_t i = 0; i < ndim; i++) {
        if (!nir_json_int(w, dims[i])) {
            return 0;
        }
    }
    return nir_json_end(w);
}

static int write_node(nir_json_writer_t *w, const nir_node_t *node) {
    const float *weight = NULL;
    const float *bias = NULL;
    size_t wn = 0;
    size_t bn = 0;
    if (!weight_shapes(node, &weight, &wn, &bias, &bn)) {
        return 0;
    }
    return nir_json_begin_object(w) && nir_json_key(w, "name") &&
        nir_json_string(w, node->name) && nir_json_key(w, "kind") &&
        nir_json_string(w, kind_name(node->kind)) &&
        write_shape(w, node) && nir_json_key(w, "weight") &&
        write_floats(w, weight, wn) && nir_json_key(w, "bias") &&
        (bias != NULL ? write_floats(w, bias, bn)
                      : nir_json_null(w)) &&
        nir_json_end(w);
}

/* Validate caller optimizer entries against the graph: known
 * weighted nodes, no duplicates, live-kind weight set with
 * matching buffer counts, inactive-kind pointers NULL. */
static int validate_opt_entries(
    const nir_graph_t *graph, const nir_checkpoint_opt_t *opts,
    size_t opt_count, char **error
) {
    if (opt_count == 0) {
        return 1;
    }
    if (opts == NULL) {
        set_error(error, "checkpoint optimizer entry needs a node");
        return 0;
    }
    for (size_t i = 0; i < opt_count; i++) {
        const nir_checkpoint_opt_t *o = &opts[i];
        if (o->node == NULL) {
            set_error(error, "checkpoint optimizer entry needs a node");
            return 0;
        }
        if (o->kind != NIR_CHECKPOINT_OPT_SGD &&
            o->kind != NIR_CHECKPOINT_OPT_ADAM) {
            set_error(error, "checkpoint optimizer entry is invalid");
            return 0;
        }
        if (o->kind == NIR_CHECKPOINT_OPT_SGD) {
            if (o->sgd_weight == NULL) {
                set_error(
                    error, "checkpoint optimizer entry needs state"
                );
                return 0;
            }
            if (o->adam_weight != NULL || o->adam_bias != NULL) {
                set_error(
                    error, "checkpoint optimizer entry mixes kinds"
                );
                return 0;
            }
        } else {
            if (o->adam_weight == NULL) {
                set_error(
                    error, "checkpoint optimizer entry needs state"
                );
                return 0;
            }
            if (o->sgd_weight != NULL || o->sgd_bias != NULL) {
                set_error(
                    error, "checkpoint optimizer entry mixes kinds"
                );
                return 0;
            }
        }
        size_t index = nir_graph_find(graph, o->node);
        if (index == (size_t)-1) {
            set_error(
                error, "checkpoint optimizer names an unknown node"
            );
            return 0;
        }
        const float *w = NULL;
        const float *b = NULL;
        size_t wn = 0;
        size_t bn = 0;
        if (!weight_shapes(&graph->nodes[index], &w, &wn, &b, &bn)) {
            set_error(
                error, "checkpoint optimizer targets an unweighted node"
            );
            return 0;
        }
        if (o->kind == NIR_CHECKPOINT_OPT_SGD) {
            if (o->sgd_weight->velocity == NULL ||
                o->sgd_weight->count != wn) {
                set_error(
                    error,
                    "checkpoint optimizer state does not match graph"
                );
                return 0;
            }
            if (o->sgd_bias != NULL &&
                (b == NULL || o->sgd_bias->velocity == NULL ||
                    o->sgd_bias->count != bn)) {
                set_error(
                    error,
                    "checkpoint optimizer state does not match graph"
                );
                return 0;
            }
        } else {
            if (o->adam_weight->m == NULL || o->adam_weight->v == NULL ||
                o->adam_weight->count != wn) {
                set_error(
                    error,
                    "checkpoint optimizer state does not match graph"
                );
                return 0;
            }
            if (o->adam_bias != NULL &&
                (b == NULL || o->adam_bias->m == NULL ||
                    o->adam_bias->v == NULL ||
                    o->adam_bias->count != bn)) {
                set_error(
                    error,
                    "checkpoint optimizer state does not match graph"
                );
                return 0;
            }
        }
        for (size_t j = 0; j < i; j++) {
            if (strcmp(opts[j].node, o->node) == 0) {
                set_error(
                    error, "checkpoint optimizer entry is duplicated"
                );
                return 0;
            }
        }
    }
    return 1;
}

static int write_sgd_entry(
    nir_json_writer_t *w, const nir_checkpoint_opt_t *o
) {
    return nir_json_key(w, "kind") && nir_json_string(w, "sgd") &&
        nir_json_key(w, "velocity") &&
        write_floats(
            w, o->sgd_weight->velocity, o->sgd_weight->count
        ) &&
        (o->sgd_bias == NULL ||
            (nir_json_key(w, "velocity_bias") &&
                write_floats(
                    w, o->sgd_bias->velocity, o->sgd_bias->count
                )));
}

static int write_adam_entry(
    nir_json_writer_t *w, const nir_checkpoint_opt_t *o
) {
    int ok = nir_json_key(w, "kind") && nir_json_string(w, "adam") &&
        nir_json_key(w, "step") &&
        nir_json_int(w, (long long)o->adam_weight->step) &&
        nir_json_key(w, "m") &&
        write_floats(w, o->adam_weight->m, o->adam_weight->count) &&
        nir_json_key(w, "v") &&
        write_floats(w, o->adam_weight->v, o->adam_weight->count);
    if (ok && o->adam_bias != NULL) {
        ok = nir_json_key(w, "step_bias") &&
            nir_json_int(w, (long long)o->adam_bias->step) &&
            nir_json_key(w, "m_bias") &&
            write_floats(
                w, o->adam_bias->m, o->adam_bias->count
            ) &&
            nir_json_key(w, "v_bias") &&
            write_floats(
                w, o->adam_bias->v, o->adam_bias->count
            );
    }
    return ok;
}

static int write_optimizers(
    nir_json_writer_t *w, const nir_checkpoint_opt_t *opts,
    size_t opt_count
) {
    if (!nir_json_key(w, "optimizers") || !nir_json_begin_object(w)) {
        return 0;
    }
    for (size_t i = 0; i < opt_count; i++) {
        const nir_checkpoint_opt_t *o = &opts[i];
        int ok = nir_json_key(w, o->node) &&
            nir_json_begin_object(w);
        if (ok) {
            ok = o->kind == NIR_CHECKPOINT_OPT_SGD
                ? write_sgd_entry(w, o)
                : write_adam_entry(w, o);
        }
        ok = ok && nir_json_end(w);
        if (!ok) {
            return 0;
        }
    }
    return nir_json_end(w);
}

static char *dumps_impl(
    const nir_graph_t *graph, const nir_checkpoint_meta_t *meta,
    const nir_checkpoint_opt_t *opts, size_t opt_count,
    char **error
) {
    if (graph == NULL) {
        set_error(error, "checkpoint save needs a graph");
        return NULL;
    }
    nir_json_writer_t *w = nir_json_writer_new();
    int ok = w != NULL && nir_json_begin_object(w) &&
        nir_json_key(w, "format") &&
        nir_json_string(w, NIR_CHECKPOINT_FORMAT) &&
        nir_json_key(w, "version") &&
        nir_json_int(w, NIR_CHECKPOINT_VERSION);
    if (ok && meta != NULL) {
        if (meta->topology != NULL) {
            ok = nir_json_key(w, "topology") &&
                nir_json_string(w, meta->topology);
        }
        ok = ok && nir_json_key(w, "seed") &&
            nir_json_double(w, (double)meta->seed) &&
            nir_json_key(w, "step") &&
            nir_json_int(w, (long long)meta->step);
    }
    ok = ok && nir_json_key(w, "nodes") && nir_json_begin_array(w);
    for (size_t i = 0; ok && i < graph->node_count; i++) {
        const nir_node_t *node = &graph->nodes[i];
        const float *weight = NULL;
        const float *bias = NULL;
        size_t wn = 0;
        size_t bn = 0;
        if (weight_shapes(node, &weight, &wn, &bias, &bn)) {
            ok = write_node(w, node);
        }
    }
    ok = ok && nir_json_end(w);
    int opt_failed = 0;
    if (ok && opts != NULL && opt_count > 0) {
        ok = write_optimizers(w, opts, opt_count);
        opt_failed = !ok;
    }
    ok = ok && nir_json_end(w);
    char *text = NULL;
    if (ok && nir_json_writer_data(w) != NULL) {
        size_t length = nir_json_writer_length(w) + 1;
        text = (char *)malloc(length);
        if (text != NULL) {
            memcpy(text, nir_json_writer_data(w), length);
        }
    }
    nir_json_writer_free(w);
    if (text == NULL) {
        set_error(
            error, opt_failed
                ? "checkpoint optimizer state must be finite"
                : "checkpoint weights must be finite"
        );
    }
    return text;
}

char *nir_checkpoint_dumps(
    const nir_graph_t *graph, const nir_checkpoint_meta_t *meta,
    char **error
) {
    return dumps_impl(graph, meta, NULL, 0, error);
}

char *nir_checkpoint_dumps_with_opt(
    const nir_graph_t *graph, const nir_checkpoint_meta_t *meta,
    const nir_checkpoint_opt_t *opts, size_t opt_count,
    char **error
) {
    if (graph != NULL &&
        !validate_opt_entries(graph, opts, opt_count, error)) {
        return NULL;
    }
    return dumps_impl(graph, meta, opts, opt_count, error);
}

static int save_text(const char *path, char *text, char **error) {
    FILE *f = fopen(path, "wb");
    int ok = 0;
    if (f != NULL) {
        size_t length = strlen(text);
        ok = fwrite(text, 1, length, f) == length;
        ok = fclose(f) == 0 && ok;
    }
    free(text);
    if (!ok) {
        set_error(error, "cannot write checkpoint file");
    }
    return ok;
}

int nir_checkpoint_save(
    const char *path, const nir_graph_t *graph,
    const nir_checkpoint_meta_t *meta, char **error
) {
    if (path == NULL) {
        set_error(error, "checkpoint save needs a path");
        return 0;
    }
    char *text = nir_checkpoint_dumps(graph, meta, error);
    if (text == NULL) {
        return 0;
    }
    return save_text(path, text, error);
}

int nir_checkpoint_save_with_opt(
    const char *path, const nir_graph_t *graph,
    const nir_checkpoint_meta_t *meta,
    const nir_checkpoint_opt_t *opts, size_t opt_count,
    char **error
) {
    if (path == NULL) {
        set_error(error, "checkpoint save needs a path");
        return 0;
    }
    char *text = nir_checkpoint_dumps_with_opt(
        graph, meta, opts, opt_count, error
    );
    if (text == NULL) {
        return 0;
    }
    return save_text(path, text, error);
}

/* Read a flat numeric array into `out[0..n)`; every element must be
 * a finite number. */
static int read_floats(
    const json_value_t *array, float *out, size_t n
) {
    if (array == NULL || array->type != JSON_ARRAY ||
        json_array_length(array) != n) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        const json_value_t *item = json_array_get(array, i);
        if (item == NULL || item->type != JSON_NUMBER ||
            !isfinite(item->as.number)) {
            return 0;
        }
        out[i] = (float)item->as.number;
    }
    return 1;
}

static int shape_matches(const nir_node_t *node, const json_value_t *shape) {
    long long want[4];
    size_t ndim = 0;
    switch (node->kind) {
        case NIR_NODE_CONV2D: {
            const nir_conv2d_params_t *p = &node->params.conv2d;
            want[0] = (long long)p->out_channels;
            want[1] = (long long)p->in_channels;
            want[2] = (long long)p->kernel_h;
            want[3] = (long long)p->kernel_w;
            ndim = 4;
            break;
        }
        case NIR_NODE_AFFINE: {
            const nir_affine_params_t *p = &node->params.affine;
            want[0] = (long long)p->out_features;
            want[1] = (long long)p->in_features;
            ndim = 2;
            break;
        }
        case NIR_NODE_CONV1D: {
            const nir_conv1d_params_t *p = &node->params.conv1d;
            want[0] = (long long)p->out_channels;
            want[1] = (long long)p->in_channels;
            want[2] = (long long)p->kernel;
            ndim = 3;
            break;
        }
        case NIR_NODE_LINEAR: {
            const nir_linear_params_t *p = &node->params.linear;
            want[0] = (long long)p->out_features;
            want[1] = (long long)p->in_features;
            ndim = 2;
            break;
        }
        default:
            return 0;
    }
    if (shape == NULL || shape->type != JSON_ARRAY ||
        json_array_length(shape) != ndim) {
        return 0;
    }
    for (size_t i = 0; i < ndim; i++) {
        const json_value_t *dim = json_array_get(shape, i);
        if (dim == NULL || dim->type != JSON_NUMBER ||
            dim->as.number != (double)want[i]) {
            return 0;
        }
    }
    return 1;
}

static int load_node_weights(
    nir_graph_t *graph, const json_value_t *entry, char **error
) {
    const char *name = json_as_string(json_object_get(entry, "name"));
    const char *kind = json_as_string(json_object_get(entry, "kind"));
    if (name == NULL || kind == NULL) {
        set_error(error, "checkpoint node is missing name or kind");
        return 0;
    }
    size_t index = nir_graph_find(graph, name);
    if (index == (size_t)-1) {
        set_error(error, "checkpoint names an unknown node");
        return 0;
    }
    nir_node_t *node = &graph->nodes[index];
    const char *want_kind = kind_name(node->kind);
    if (want_kind == NULL || strcmp(kind, want_kind) != 0) {
        set_error(error, "checkpoint node kind does not match graph");
        return 0;
    }
    const float *dst_w = NULL;
    const float *dst_b = NULL;
    size_t wn = 0;
    size_t bn = 0;
    if (!weight_shapes(node, &dst_w, &wn, &dst_b, &bn) ||
        !shape_matches(node, json_object_get(entry, "shape"))) {
        set_error(error, "checkpoint node shape does not match graph");
        return 0;
    }
    if (!read_floats(json_object_get(entry, "weight"), (float *)dst_w, wn)) {
        set_error(error, "checkpoint node weight is not a flat array");
        return 0;
    }
    const json_value_t *bias = json_object_get(entry, "bias");
    if (bias == NULL || bias->type == JSON_NULL) {
        if (dst_b != NULL) {
            set_error(error, "checkpoint drops a node bias");
            return 0;
        }
        return 1;
    }
    if (dst_b == NULL || !read_floats(bias, (float *)dst_b, bn)) {
        set_error(error, "checkpoint node bias does not match graph");
        return 0;
    }
    return 1;
}

static char *read_whole_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    char *text = (char *)malloc((size_t)size + 1);
    if (text == NULL || (size > 0 &&
            fread(text, 1, (size_t)size, f) != (size_t)size)) {
        free(text);
        fclose(f);
        return NULL;
    }
    text[size] = '\0';
    fclose(f);
    return text;
}

/* True when every element of an already length-checked array is a
 * finite JSON number. */
static int floats_finite(const json_value_t *array) {
    size_t n = json_array_length(array);
    for (size_t i = 0; i < n; i++) {
        const json_value_t *item = json_array_get(array, i);
        if (item == NULL || item->type != JSON_NUMBER ||
            !isfinite(item->as.number)) {
            return 0;
        }
    }
    return 1;
}

/* Validate an Adam step counter: a non-negative integer within the
 * exactly representable double range (2^53-1 covers any realistic
 * training run). */
static int read_opt_step(const json_value_t *v, unsigned long *out) {
    if (v == NULL || v->type != JSON_NUMBER) {
        return 0;
    }
    double d = v->as.number;
    if (!isfinite(d) || d < 0.0 || d != floor(d) ||
        d > 9007199254740991.0) {
        return 0;
    }
    *out = (unsigned long)d;
    return 1;
}

static nir_checkpoint_opt_t *find_opt(
    nir_checkpoint_opt_t *opts, size_t opt_count, const char *node
) {
    for (size_t i = 0; i < opt_count; i++) {
        if (opts[i].node != NULL && strcmp(opts[i].node, node) == 0) {
            return &opts[i];
        }
    }
    return NULL;
}

/* Check one required optimizer buffer: present array, exact
 * length, all finite. Length mismatches report a shape error;
 * missing/non-array/non-finite report a flat-array error. */
static int check_opt_buffer(
    const json_value_t *entry, const char *key, size_t want,
    char **error
) {
    const json_value_t *arr = json_object_get(entry, key);
    if (arr == NULL || arr->type != JSON_ARRAY) {
        set_error(
            error, "checkpoint optimizer state is not a flat array"
        );
        return 0;
    }
    if (json_array_length(arr) != want) {
        set_error(
            error, "checkpoint optimizer shape does not match graph"
        );
        return 0;
    }
    if (!floats_finite(arr)) {
        set_error(
            error, "checkpoint optimizer state is not a flat array"
        );
        return 0;
    }
    return 1;
}

/* Validate one file optimizer entry and restore it into its
 * matching caller entry (already located). Validates every field
 * before writing any, so a failing entry leaves that node's
 * optimizer untouched. */
static int load_one_opt(
    const char *name, const json_value_t *entry,
    nir_checkpoint_opt_t *dst, size_t wn, size_t bn, int has_bias,
    char **error
) {
    (void)name;
    if (entry == NULL || entry->type != JSON_OBJECT) {
        set_error(error, "checkpoint optimizer entry is invalid");
        return 0;
    }
    const char *kind = json_as_string(json_object_get(entry, "kind"));
    int is_sgd = kind != NULL && strcmp(kind, "sgd") == 0;
    int is_adam = kind != NULL && strcmp(kind, "adam") == 0;
    if (!is_sgd && !is_adam) {
        set_error(error, "checkpoint optimizer has an unknown kind");
        return 0;
    }
    if ((is_sgd && dst->kind != NIR_CHECKPOINT_OPT_SGD) ||
        (is_adam && dst->kind != NIR_CHECKPOINT_OPT_ADAM)) {
        set_error(
            error, "checkpoint optimizer kind does not match entry"
        );
        return 0;
    }
    if (is_sgd) {
        if (!check_opt_buffer(entry, "velocity", wn, error)) {
            return 0;
        }
        const json_value_t *vb = json_object_get(entry, "velocity_bias");
        int want_bias = vb != NULL && vb->type != JSON_NULL;
        if (want_bias) {
            if (!has_bias) {
                set_error(
                    error,
                    "checkpoint optimizer shape does not match graph"
                );
                return 0;
            }
            if (!check_opt_buffer(
                    entry, "velocity_bias", bn, error
                )) {
                return 0;
            }
        }
        if (!read_floats(
                json_object_get(entry, "velocity"),
                dst->sgd_weight->velocity, wn
            )) {
            set_error(
                error, "checkpoint optimizer state is not a flat array"
            );
            return 0;
        }
        if (want_bias && dst->sgd_bias != NULL &&
            !read_floats(vb, dst->sgd_bias->velocity, bn)) {
            set_error(
                error, "checkpoint optimizer state is not a flat array"
            );
            return 0;
        }
        return 1;
    }
    unsigned long step = 0;
    if (!read_opt_step(json_object_get(entry, "step"), &step)) {
        set_error(error, "checkpoint optimizer step is invalid");
        return 0;
    }
    if (!check_opt_buffer(entry, "m", wn, error) ||
        !check_opt_buffer(entry, "v", wn, error)) {
        return 0;
    }
    const json_value_t *mb = json_object_get(entry, "m_bias");
    const json_value_t *vb = json_object_get(entry, "v_bias");
    int has_mb = mb != NULL && mb->type != JSON_NULL;
    int has_vb = vb != NULL && vb->type != JSON_NULL;
    if (has_mb != has_vb) {
        set_error(
            error, "checkpoint optimizer bias state is incomplete"
        );
        return 0;
    }
    unsigned long step_bias = step;
    if (has_mb) {
        if (!has_bias) {
            set_error(
                error,
                "checkpoint optimizer shape does not match graph"
            );
            return 0;
        }
        const json_value_t *sb = json_object_get(entry, "step_bias");
        if (sb != NULL && sb->type != JSON_NULL &&
            !read_opt_step(sb, &step_bias)) {
            set_error(error, "checkpoint optimizer step is invalid");
            return 0;
        }
        if (!check_opt_buffer(entry, "m_bias", bn, error) ||
            !check_opt_buffer(entry, "v_bias", bn, error)) {
            return 0;
        }
    }
    if (!read_floats(
            json_object_get(entry, "m"), dst->adam_weight->m, wn
        ) ||
        !read_floats(
            json_object_get(entry, "v"), dst->adam_weight->v, wn
        )) {
        set_error(
            error, "checkpoint optimizer state is not a flat array"
        );
        return 0;
    }
    dst->adam_weight->step = step;
    if (has_mb && dst->adam_bias != NULL) {
        if (!read_floats(mb, dst->adam_bias->m, bn) ||
            !read_floats(vb, dst->adam_bias->v, bn)) {
            set_error(
                error, "checkpoint optimizer state is not a flat array"
            );
            return 0;
        }
        dst->adam_bias->step = step_bias;
    }
    return 1;
}

static int load_optimizers(
    const json_value_t *root, nir_graph_t *graph,
    nir_checkpoint_opt_t *opts, size_t opt_count, char **error
) {
    const json_value_t *opt_root = json_object_get(root, "optimizers");
    if (opt_root == NULL || opt_root->type == JSON_NULL) {
        return 1;
    }
    if (opt_root->type != JSON_OBJECT) {
        set_error(
            error, "checkpoint optimizers section is not an object"
        );
        return 0;
    }
    for (size_t i = 0; i < opt_root->as.object.count; i++) {
        const char *name = opt_root->as.object.keys[i];
        const json_value_t *entry = opt_root->as.object.values[i];
        size_t index = nir_graph_find(graph, name);
        if (index == (size_t)-1) {
            set_error(
                error, "checkpoint optimizer names an unknown node"
            );
            return 0;
        }
        const float *w = NULL;
        const float *b = NULL;
        size_t wn = 0;
        size_t bn = 0;
        if (!weight_shapes(&graph->nodes[index], &w, &wn, &b, &bn)) {
            set_error(
                error, "checkpoint optimizer targets an unweighted node"
            );
            return 0;
        }
        nir_checkpoint_opt_t *dst = find_opt(opts, opt_count, name);
        if (dst == NULL) {
            set_error(
                error, "checkpoint optimizer has no matching entry"
            );
            return 0;
        }
        if (!load_one_opt(
                name, entry, dst, wn, bn, b != NULL, error
            )) {
            return 0;
        }
    }
    return 1;
}

static int loads_impl(
    const char *text, nir_graph_t *graph, nir_checkpoint_meta_t *meta,
    nir_checkpoint_opt_t *opts, size_t opt_count, char **error
) {
    if (text == NULL || graph == NULL) {
        set_error(error, "checkpoint load needs text and a graph");
        return 0;
    }
    char *parse_error = NULL;
    json_value_t *root = json_parse(text, &parse_error);
    if (root == NULL) {
        free(parse_error);
        set_error(error, "checkpoint is not valid JSON");
        return 0;
    }
    const char *format = json_as_string(json_object_get(root, "format"));
    const json_value_t *version = json_object_get(root, "version");
    int ok = format != NULL && strcmp(format, NIR_CHECKPOINT_FORMAT) == 0;
    if (!ok) {
        set_error(error, "checkpoint has an unknown format marker");
    } else if (version == NULL || version->type != JSON_NUMBER ||
        (version->as.number != 1.0 && version->as.number != 2.0)) {
        set_error(error, "checkpoint has an unsupported version");
        ok = 0;
    }
    const json_value_t *nodes = json_object_get(root, "nodes");
    if (ok && (nodes == NULL || nodes->type != JSON_ARRAY)) {
        set_error(error, "checkpoint has no node array");
        ok = 0;
    }
    unsigned char *seen = NULL;
    if (ok) {
        seen = (unsigned char *)calloc(
            graph->node_count == 0 ? 1 : graph->node_count, 1
        );
        ok = seen != NULL;
    }
    for (size_t i = 0; ok && i < json_array_length(nodes); i++) {
        const json_value_t *entry = json_array_get(nodes, i);
        const char *name = json_as_string(json_object_get(entry, "name"));
        ok = load_node_weights(graph, entry, error);
        if (ok) {
            seen[nir_graph_find(graph, name)] = 1;
        }
    }
    /* Every weighted node must be covered: a partial resume would
     * silently mix stale and fresh weights. */
    for (size_t i = 0; ok && i < graph->node_count; i++) {
        const float *w = NULL;
        const float *b = NULL;
        size_t wn = 0;
        size_t bn = 0;
        if (weight_shapes(&graph->nodes[i], &w, &wn, &b, &bn) &&
            !seen[i]) {
            set_error(error, "checkpoint lacks weights for a node");
            ok = 0;
        }
    }
    if (ok && opts != NULL && opt_count > 0) {
        ok = load_optimizers(root, graph, opts, opt_count, error);
    }
    if (ok && meta != NULL) {
        memset(meta, 0, sizeof(*meta));
        const char *topology = json_as_string(
            json_object_get(root, "topology")
        );
        if (topology != NULL) {
            meta->topology = (char *)malloc(strlen(topology) + 1);
            if (meta->topology == NULL) {
                set_error(error, "out of memory");
                ok = 0;
            } else {
                strcpy(meta->topology, topology);
            }
        }
        meta->seed = (uint64_t)json_as_number(
            json_object_get(root, "seed"), 0.0
        );
        meta->step = (unsigned long)json_as_number(
            json_object_get(root, "step"), 0.0
        );
    }
    free(seen);
    json_free(root);
    return ok;
}

int nir_checkpoint_loads(
    const char *text, nir_graph_t *graph, nir_checkpoint_meta_t *meta,
    char **error
) {
    return loads_impl(text, graph, meta, NULL, 0, error);
}

int nir_checkpoint_loads_with_opt(
    const char *text, nir_graph_t *graph, nir_checkpoint_meta_t *meta,
    nir_checkpoint_opt_t *opts, size_t opt_count,
    char **error
) {
    if (graph != NULL && opt_count > 0 &&
        !validate_opt_entries(graph, opts, opt_count, error)) {
        return 0;
    }
    /* NULL/0 restores weights only, ignoring any file optimizers. */
    if (opts == NULL || opt_count == 0) {
        return loads_impl(text, graph, meta, NULL, 0, error);
    }
    return loads_impl(text, graph, meta, opts, opt_count, error);
}

int nir_checkpoint_load(
    const char *path, nir_graph_t *graph, nir_checkpoint_meta_t *meta,
    char **error
) {
    if (path == NULL) {
        set_error(error, "checkpoint load needs a path");
        return 0;
    }
    char *text = read_whole_file(path);
    if (text == NULL) {
        set_error(error, "cannot read checkpoint file");
        return 0;
    }
    int ok = nir_checkpoint_loads(text, graph, meta, error);
    free(text);
    return ok;
}

int nir_checkpoint_load_with_opt(
    const char *path, nir_graph_t *graph, nir_checkpoint_meta_t *meta,
    nir_checkpoint_opt_t *opts, size_t opt_count,
    char **error
) {
    if (path == NULL) {
        set_error(error, "checkpoint load needs a path");
        return 0;
    }
    char *text = read_whole_file(path);
    if (text == NULL) {
        set_error(error, "cannot read checkpoint file");
        return 0;
    }
    int ok = nir_checkpoint_loads_with_opt(
        text, graph, meta, opts, opt_count, error
    );
    free(text);
    return ok;
}

/* Emit `weight` (flat row-major, `wn` elements) as a nested array
 * with per-level sizes `dims[0..ndim)`. */
static int write_nested(
    nir_json_writer_t *w, const float *weight, const size_t *dims,
    size_t ndim
) {
    if (ndim == 1) {
        if (!nir_json_begin_array(w)) {
            return 0;
        }
        for (size_t i = 0; i < dims[0]; i++) {
            if (!isfinite(weight[i]) || !nir_json_float(w, weight[i])) {
                return 0;
            }
        }
        return nir_json_end(w);
    }
    if (!nir_json_begin_array(w)) {
        return 0;
    }
    size_t stride = 1;
    for (size_t i = 1; i < ndim; i++) {
        stride *= dims[i];
    }
    for (size_t i = 0; i < dims[0]; i++) {
        if (!write_nested(w, weight + i * stride, dims + 1, ndim - 1)) {
            return 0;
        }
    }
    return nir_json_end(w);
}

static int write_pair(nir_json_writer_t *w, const char *key, int a, int b) {
    return nir_json_key(w, key) && nir_json_begin_array(w) &&
        nir_json_int(w, a) && nir_json_int(w, b) && nir_json_end(w);
}

static int write_scalar(
    nir_json_writer_t *w, const char *key, float value
) {
    return nir_json_key(w, key) && nir_json_float(w, value);
}

static int export_weighted(nir_json_writer_t *w, const nir_node_t *node) {
    const float *weight = NULL;
    const float *bias = NULL;
    size_t wn = 0;
    size_t bn = 0;
    size_t dims[4];
    size_t ndim = 0;
    if (!weight_shapes(node, &weight, &wn, &bias, &bn)) {
        return 0;
    }
    switch (node->kind) {
        case NIR_NODE_CONV2D: {
            const nir_conv2d_params_t *p = &node->params.conv2d;
            dims[0] = p->out_channels;
            dims[1] = p->in_channels;
            dims[2] = (size_t)p->kernel_h;
            dims[3] = (size_t)p->kernel_w;
            ndim = 4;
            break;
        }
        case NIR_NODE_AFFINE: {
            const nir_affine_params_t *p = &node->params.affine;
            dims[0] = p->out_features;
            dims[1] = p->in_features;
            ndim = 2;
            break;
        }
        case NIR_NODE_CONV1D: {
            const nir_conv1d_params_t *p = &node->params.conv1d;
            dims[0] = p->out_channels;
            dims[1] = p->in_channels;
            dims[2] = p->kernel;
            ndim = 3;
            break;
        }
        case NIR_NODE_LINEAR: {
            const nir_linear_params_t *p = &node->params.linear;
            dims[0] = p->out_features;
            dims[1] = p->in_features;
            ndim = 2;
            break;
        }
        default:
            return 0;
    }
    int ok = nir_json_key(w, "weight") &&
        write_nested(w, weight, dims, ndim) && nir_json_key(w, "bias");
    if (!ok) {
        return 0;
    }
    if (node->kind == NIR_NODE_LINEAR) {
        ok = nir_json_null(w);
    } else if (bias != NULL) {
        ok = write_floats(w, bias, bn);
    } else {
        ok = nir_json_null(w);
    }
    return ok;
}

static int export_conv2d(nir_json_writer_t *w, const nir_node_t *node) {
    const nir_conv2d_params_t *p = &node->params.conv2d;
    return export_weighted(w, node) &&
        write_pair(w, "stride", p->stride_h, p->stride_w) &&
        write_pair(w, "padding", p->padding_h, p->padding_w) &&
        write_pair(w, "dilation", p->dilation_h, p->dilation_w);
}

static int export_conv1d(nir_json_writer_t *w, const nir_node_t *node) {
    const nir_conv1d_params_t *p = &node->params.conv1d;
    return export_weighted(w, node) &&
        nir_json_key(w, "stride") && nir_json_int(w, p->stride) &&
        nir_json_key(w, "padding") && nir_json_int(w, p->padding) &&
        nir_json_key(w, "dilation") && nir_json_int(w, p->dilation);
}

static int export_pool2d(nir_json_writer_t *w, const nir_node_t *node) {
    const nir_pool2d_params_t *p = &node->params.pool2d;
    return write_pair(w, "kernel_size", p->kernel_h, p->kernel_w) &&
        write_pair(w, "stride", p->stride_h, p->stride_w) &&
        write_pair(w, "padding", p->padding_h, p->padding_w);
}

static int export_lif(nir_json_writer_t *w, const nir_node_t *node) {
    const nir_lif_params_t *p = &node->params.lif;
    return write_scalar(w, "tau", p->tau) &&
        write_scalar(w, "r", p->r) &&
        write_scalar(w, "v_leak", p->v_leak) &&
        write_scalar(w, "v_threshold", p->v_threshold) &&
        write_scalar(w, "v_reset", p->v_reset);
}

static int export_if(nir_json_writer_t *w, const nir_node_t *node) {
    const nir_if_params_t *p = &node->params.if_;
    return write_scalar(w, "r", p->r) &&
        write_scalar(w, "v_threshold", p->v_threshold) &&
        write_scalar(w, "v_reset", p->v_reset);
}

static int export_cuba(nir_json_writer_t *w, const nir_node_t *node) {
    const nir_cuba_params_t *p = &node->params.cuba;
    return write_scalar(w, "tau_syn", p->tau_syn) &&
        write_scalar(w, "tau_mem", p->tau_mem) &&
        write_scalar(w, "r", p->r) &&
        write_scalar(w, "v_leak", p->v_leak) &&
        write_scalar(w, "v_threshold", p->v_threshold) &&
        write_scalar(w, "v_reset", p->v_reset) &&
        write_scalar(w, "w_in", p->w_in);
}

static int export_params(nir_json_writer_t *w, const nir_node_t *node) {
    if (!nir_json_key(w, "params") || !nir_json_begin_object(w)) {
        return 0;
    }
    int ok = 1;
    switch (node->kind) {
        case NIR_NODE_CONV2D:
            ok = export_conv2d(w, node);
            break;
        case NIR_NODE_CONV1D:
            ok = export_conv1d(w, node);
            break;
        case NIR_NODE_AFFINE:
        case NIR_NODE_LINEAR:
            ok = export_weighted(w, node);
            break;
        case NIR_NODE_AVGPOOL2D:
        case NIR_NODE_SUMPOOL2D:
            ok = export_pool2d(w, node);
            break;
        case NIR_NODE_LI: {
            const nir_li_params_t *p = &node->params.li;
            ok = write_scalar(w, "tau", p->tau) &&
                write_scalar(w, "r", p->r) &&
                write_scalar(w, "v_leak", p->v_leak);
            break;
        }
        case NIR_NODE_LIF:
            ok = export_lif(w, node);
            break;
        case NIR_NODE_IF:
            ok = export_if(w, node);
            break;
        case NIR_NODE_CUBALIF:
            ok = export_cuba(w, node);
            break;
        case NIR_NODE_THRESHOLD:
            ok = write_scalar(
                w, "threshold", node->params.threshold.threshold
            );
            break;
        case NIR_NODE_SCALE:
            ok = write_scalar(w, "scale", node->params.scale.scale);
            break;
        case NIR_NODE_RELU:
            break;
        default:
            break; /* Input/Output/Flatten/Delay carry no params */
    }
    return ok && nir_json_end(w);
}

char *nir_graph_export_summary(
    const nir_graph_t *graph, char **error
) {
    if (graph == NULL) {
        set_error(error, "summary export needs a graph");
        return NULL;
    }
    nir_json_writer_t *w = nir_json_writer_new();
    int ok = w != NULL && nir_json_begin_object(w) &&
        nir_json_key(w, "nodes") && nir_json_begin_array(w);
    for (size_t i = 0; ok && i < graph->node_count; i++) {
        const nir_node_t *node = &graph->nodes[i];
        const char *kind = kind_name(node->kind);
        ok = kind != NULL && nir_json_begin_object(w) &&
            nir_json_key(w, "name") &&
            nir_json_string(w, node->name) && nir_json_key(w, "kind") &&
            nir_json_string(w, kind) && export_params(w, node) &&
            nir_json_end(w);
    }
    ok = ok && nir_json_end(w) && nir_json_key(w, "edges") &&
        nir_json_begin_array(w);
    for (size_t i = 0; ok && i < graph->edge_count; i++) {
        size_t src = graph->edges[i].source_index;
        size_t dst = graph->edges[i].target_index;
        if (src >= graph->node_count || dst >= graph->node_count) {
            ok = 0;
            break;
        }
        ok = nir_json_begin_object(w) && nir_json_key(w, "source") &&
            nir_json_string(w, graph->nodes[src].name) &&
            nir_json_key(w, "target") &&
            nir_json_string(w, graph->nodes[dst].name) &&
            nir_json_key(w, "delayed") &&
            nir_json_bool(
                w, graph->nodes[dst].kind == NIR_NODE_DELAY
            ) &&
            nir_json_end(w);
    }
    ok = ok && nir_json_end(w) && nir_json_end(w);
    char *text = NULL;
    if (ok && nir_json_writer_data(w) != NULL) {
        size_t length = nir_json_writer_length(w) + 1;
        text = (char *)malloc(length);
        if (text != NULL) {
            memcpy(text, nir_json_writer_data(w), length);
        }
    }
    nir_json_writer_free(w);
    if (text == NULL) {
        set_error(error, "summary export failed to serialize");
    }
    return text;
}
