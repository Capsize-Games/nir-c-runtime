#include "nir_c_runtime/graph.h"

#include <stdio.h>
#include <limits.h>
#include <math.h>
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

typedef struct {
    const json_value_t *data;
    double scale;
    double zero_point;
    double quant_min;
    double quant_max;
    int quantized;
} weight_view_t;

static int weight_view(const json_value_t *value, weight_view_t *view) {
    view->data = value;
    view->scale = 1.0;
    view->zero_point = 0.0;
    view->quant_min = -INFINITY;
    view->quant_max = INFINITY;
    view->quantized = 0;
    if (value == NULL || value->type == JSON_ARRAY) {
        return value != NULL;
    }
    if (value->type != JSON_OBJECT) {
        return 0;
    }
    const json_value_t *data = json_object_get(value, "data");
    const json_value_t *dtype = json_object_get(value, "dtype");
    const json_value_t *scale = json_object_get(value, "scale");
    const json_value_t *zero = json_object_get(value, "zero_point");
    const char *dtype_name = json_as_string(dtype);
    if (data == NULL || data->type != JSON_ARRAY || dtype_name == NULL ||
        (strcmp(dtype_name, "int8") != 0 &&
         strcmp(dtype_name, "uint8") != 0) ||
        scale == NULL || scale->type != JSON_NUMBER ||
        zero == NULL || zero->type != JSON_NUMBER) {
        return 0;
    }
    view->data = data;
    view->scale = json_as_number(scale, 0.0);
    view->zero_point = json_as_number(zero, 0.0);
    view->quantized = 1;
    view->quant_min = strcmp(dtype_name, "int8") == 0 ? -128.0 : 0.0;
    view->quant_max = strcmp(dtype_name, "int8") == 0 ? 127.0 : 255.0;
    return isfinite(view->scale) && view->scale > 0.0 &&
           isfinite(view->zero_point) && trunc(view->zero_point) == view->zero_point;
}

static float decode_weight(
    const weight_view_t *view, const json_value_t *value
) {
    return (float)((json_as_number(value, 0.0) - view->zero_point) *
                   view->scale);
}

static int checked_product(size_t *product, size_t factor) {
    if (factor != 0 && *product > (size_t)-1 / factor) return 0;
    *product *= factor;
    return 1;
}

static int valid_number(const json_value_t *value) {
    return value != NULL && value->type == JSON_NUMBER &&
           isfinite(value->as.number) && isfinite((float)value->as.number);
}

static int decode_input_valid(const weight_view_t *view,
                              const json_value_t *value) {
    return valid_number(value) &&
           (!view->quantized ||
            (trunc(value->as.number) == value->as.number &&
             value->as.number >= view->quant_min &&
             value->as.number <= view->quant_max));
}

/* Flatten a raw or quantized 1D JSON array into a float buffer. */
static float *parse_floats_1d(const json_value_t *array, size_t *count) {
    weight_view_t view;
    if (!weight_view(array, &view)) {
        return NULL;
    }
    size_t n = json_array_length(view.data);
    if (view.data->type != JSON_ARRAY || n > (size_t)-1 / sizeof(float))
        return NULL;
    float *out = (float *)malloc((n == 0 ? 1 : n) * sizeof(float));
    if (out == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < n; i++) {
        const json_value_t *item = json_array_get(view.data, i);
        if (!decode_input_valid(&view, item)) { free(out); return NULL; }
        out[i] = decode_weight(&view, item);
        if (!isfinite(out[i])) { free(out); return NULL; }
    }
    *count = n;
    return out;
}

/* Flatten a raw or quantized 2D JSON array row-major. */
static float *parse_floats_2d(
    const json_value_t *array, size_t *rows, size_t *cols
) {
    weight_view_t view;
    if (!weight_view(array, &view)) {
        return NULL;
    }
    size_t r = json_array_length(view.data);
    if (view.data->type != JSON_ARRAY) return NULL;
    size_t c = r > 0 ? json_array_length(json_array_get(view.data, 0)) : 0;
    size_t total = r;
    if (!checked_product(&total, c) || total > (size_t)-1 / sizeof(float))
        return NULL;
    float *out = (float *)malloc((total == 0 ? 1 : total) * sizeof(float));
    if (out == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < r; i++) {
        const json_value_t *row = json_array_get(view.data, i);
        if (row == NULL || row->type != JSON_ARRAY ||
            json_array_length(row) != c) { free(out); return NULL; }
        for (size_t j = 0; j < c; j++) {
            const json_value_t *item = json_array_get(row, j);
            if (!decode_input_valid(&view, item)) { free(out); return NULL; }
            out[i * c + j] = decode_weight(&view, item);
            if (!isfinite(out[i * c + j])) { free(out); return NULL; }
        }
    }
    *rows = r;
    *cols = c;
    return out;
}

/* Flatten a raw or quantized 3D array row-major for Conv1d weights. */
static float *parse_floats_3d(
    const json_value_t *array, size_t *d0, size_t *d1, size_t *d2
) {
    weight_view_t view;
    if (!weight_view(array, &view)) {
        return NULL;
    }
    size_t a = json_array_length(view.data);
    if (view.data->type != JSON_ARRAY) return NULL;
    const json_value_t *l1 = a > 0 ? json_array_get(view.data, 0) : NULL;
    size_t b = json_array_length(l1);
    const json_value_t *l2 = b > 0 ? json_array_get(l1, 0) : NULL;
    size_t c = json_array_length(l2);
    size_t total = a;
    if (!checked_product(&total, b) || !checked_product(&total, c) ||
        total > (size_t)-1 / sizeof(float))
        return NULL;
    float *out = (float *)malloc((total == 0 ? 1 : total) * sizeof(float));
    if (out == NULL) {
        return NULL;
    }
    size_t index = 0;
    for (size_t i = 0; i < a; i++) {
        const json_value_t *dim1 = json_array_get(view.data, i);
        if (dim1 == NULL || dim1->type != JSON_ARRAY ||
            json_array_length(dim1) != b) { free(out); return NULL; }
        for (size_t j = 0; j < b; j++) {
            const json_value_t *dim2 = json_array_get(dim1, j);
            if (dim2 == NULL || dim2->type != JSON_ARRAY ||
                json_array_length(dim2) != c) { free(out); return NULL; }
            for (size_t k = 0; k < c; k++) {
                const json_value_t *item = json_array_get(dim2, k);
                if (!decode_input_valid(&view, item)) { free(out); return NULL; }
                out[index++] = decode_weight(&view, item);
                if (!isfinite(out[index - 1])) { free(out); return NULL; }
            }
        }
    }
    *d0 = a;
    *d1 = b;
    *d2 = c;
    return out;
}

/* Flatten a raw or quantized 4D array row-major for Conv2d weights. */
static float *parse_floats_4d(
    const json_value_t *array, size_t *d0, size_t *d1, size_t *d2, size_t *d3
) {
    weight_view_t view;
    if (!weight_view(array, &view)) {
        return NULL;
    }
    size_t a = json_array_length(view.data);
    if (view.data->type != JSON_ARRAY) return NULL;
    const json_value_t *l1 = a > 0 ? json_array_get(view.data, 0) : NULL;
    size_t b = json_array_length(l1);
    const json_value_t *l2 = b > 0 ? json_array_get(l1, 0) : NULL;
    size_t c = json_array_length(l2);
    const json_value_t *l3 = c > 0 ? json_array_get(l2, 0) : NULL;
    size_t d = json_array_length(l3);
    size_t total = a;
    if (!checked_product(&total, b) || !checked_product(&total, c) ||
        !checked_product(&total, d) || total > (size_t)-1 / sizeof(float))
        return NULL;
    float *out = (float *)malloc((total == 0 ? 1 : total) * sizeof(float));
    if (out == NULL) {
        return NULL;
    }
    size_t index = 0;
    for (size_t i = 0; i < a; i++) {
        const json_value_t *dim1 = json_array_get(view.data, i);
        if (dim1 == NULL || dim1->type != JSON_ARRAY ||
            json_array_length(dim1) != b) { free(out); return NULL; }
        for (size_t j = 0; j < b; j++) {
            const json_value_t *dim2 = json_array_get(dim1, j);
            if (dim2 == NULL || dim2->type != JSON_ARRAY ||
                json_array_length(dim2) != c) { free(out); return NULL; }
            for (size_t k = 0; k < c; k++) {
                const json_value_t *dim3 = json_array_get(dim2, k);
                if (dim3 == NULL || dim3->type != JSON_ARRAY ||
                    json_array_length(dim3) != d) { free(out); return NULL; }
                for (size_t l = 0; l < d; l++) {
                    const json_value_t *item = json_array_get(dim3, l);
                    if (!decode_input_valid(&view, item)) { free(out); return NULL; }
                    out[index++] = decode_weight(
                        &view, item
                    );
                    if (!isfinite(out[index - 1])) { free(out); return NULL; }
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
    const json_value_t *first = json_array_get(pair, 0);
    const json_value_t *second = json_array_get(pair, 1);
    if (pair == NULL || pair->type != JSON_ARRAY ||
        json_array_length(pair) != 2 || !valid_number(first) ||
        !valid_number(second) || trunc(first->as.number) != first->as.number ||
        trunc(second->as.number) != second->as.number ||
        first->as.number < 0 || second->as.number < 0 ||
        first->as.number > INT_MAX || second->as.number > INT_MAX) return 0;
    *a = (int)first->as.number;
    *b = (int)second->as.number;
    return 1;
}

static int load_conv2d(
    const json_value_t *params, nir_node_t *node, char **error
) {
    nir_conv2d_params_t *p = &node->params.conv2d;
    const json_value_t *weight = json_object_get(params, "weight");
    p->weight = parse_floats_4d(
        weight, &p->out_channels, &p->in_channels, &p->kernel_h, &p->kernel_w
    );
    if (p->weight == NULL || p->out_channels == 0 || p->in_channels == 0 ||
        p->kernel_h == 0 || p->kernel_w == 0) {
        set_error(error, "Conv2d weight must be a nonempty rectangular 4D numeric array", NULL);
        return 0;
    }
    const json_value_t *bias = json_object_get(params, "bias");
    if (bias != NULL && bias->type != JSON_NULL) {
        size_t bias_count = 0;
        p->bias = parse_floats_1d(bias, &bias_count);
        if (p->bias == NULL || bias_count != p->out_channels) {
            set_error(error,
                "Conv2d bias must contain one value per output channel",
                NULL);
            return 0;
        }
    } else {
        p->bias = NULL;
    }
    if (!pair_at(params, "stride", &p->stride_h, &p->stride_w) ||
        p->stride_h == 0 || p->stride_w == 0 ||
        !pair_at(params, "padding", &p->padding_h, &p->padding_w) ||
        !pair_at(params, "dilation", &p->dilation_h, &p->dilation_w) ||
        p->dilation_h == 0 || p->dilation_w == 0) {
        set_error(error, "Conv2d stride/dilation must be positive pairs and padding a nonnegative pair", NULL);
        return 0;
    }
    return 1;
}

static int int_at(const json_value_t *params, const char *key, int *out) {
    const json_value_t *value = json_object_get(params, key);
    if (!valid_number(value) || trunc(value->as.number) != value->as.number ||
        value->as.number < 0 || value->as.number > INT_MAX) return 0;
    *out = (int)value->as.number;
    return 1;
}

static int load_conv1d(
    const json_value_t *params, nir_node_t *node, char **error
) {
    nir_conv1d_params_t *p = &node->params.conv1d;
    const json_value_t *weight = json_object_get(params, "weight");
    p->weight = parse_floats_3d(
        weight, &p->out_channels, &p->in_channels, &p->kernel
    );
    if (p->weight == NULL || p->out_channels == 0 || p->in_channels == 0 ||
        p->kernel == 0) {
        set_error(error, "Conv1d weight must be a nonempty rectangular 3D numeric array", NULL);
        return 0;
    }
    const json_value_t *bias = json_object_get(params, "bias");
    if (bias != NULL && bias->type != JSON_NULL) {
        size_t bias_count = 0;
        p->bias = parse_floats_1d(bias, &bias_count);
        if (p->bias == NULL || bias_count != p->out_channels) {
            set_error(error,
                "Conv1d bias must contain one value per output channel",
                NULL);
            return 0;
        }
    } else {
        p->bias = NULL;
    }
    const json_value_t *padding = json_object_get(params, "padding");
    if (padding != NULL && padding->type == JSON_STRING) {
        set_error(error, "Conv1d padding must be an integer; \"same\" and \"valid\" padding are not supported", NULL);
        return 0;
    }
    int groups = 1;
    const json_value_t *groups_json = json_object_get(params, "groups");
    if (groups_json != NULL && !int_at(params, "groups", &groups)) {
        set_error(error, "Conv1d groups must be a nonnegative integer", NULL);
        return 0;
    }
    if (groups != 1) {
        set_error(error, "Conv1d groups must be 1; grouped convolution is not supported", NULL);
        return 0;
    }
    if (!int_at(params, "stride", &p->stride) || p->stride == 0 ||
        !int_at(params, "padding", &p->padding) ||
        !int_at(params, "dilation", &p->dilation) || p->dilation == 0) {
        set_error(error, "Conv1d stride/dilation must be positive integers and padding a nonnegative integer", NULL);
        return 0;
    }
    return 1;
}

static int load_linear(
    const json_value_t *params, nir_node_t *node, char **error
) {
    nir_linear_params_t *p = &node->params.linear;
    const json_value_t *weight = json_object_get(params, "weight");
    p->weight = parse_floats_2d(weight, &p->out_features, &p->in_features);
    if (p->weight == NULL || p->out_features == 0 || p->in_features == 0) {
        set_error(error, "Linear weight must be a nonempty rectangular 2D numeric array", NULL);
        return 0;
    }
    return 1;
}

static int load_affine(
    const json_value_t *params, nir_node_t *node, char **error
) {
    nir_affine_params_t *p = &node->params.affine;
    const json_value_t *weight = json_object_get(params, "weight");
    p->weight = parse_floats_2d(weight, &p->out_features, &p->in_features);
    if (p->weight == NULL || p->out_features == 0 || p->in_features == 0) {
        set_error(error, "Affine weight must be a nonempty rectangular 2D numeric array", NULL);
        return 0;
    }
    size_t bias_count = 0;
    p->bias = parse_floats_1d(json_object_get(params, "bias"), &bias_count);
    if (p->bias == NULL || bias_count != p->out_features) {
        set_error(error,
            "Affine bias must contain one value per output feature", NULL);
        return 0;
    }
    return 1;
}

static int load_pool2d(const json_value_t *params, nir_node_t *node) {
    nir_pool2d_params_t *p = &node->params.pool2d;
    return pair_at(params, "kernel_size", &p->kernel_h, &p->kernel_w) &&
        p->kernel_h > 0 && p->kernel_w > 0 &&
        pair_at(params, "stride", &p->stride_h, &p->stride_w) &&
        p->stride_h > 0 && p->stride_w > 0 &&
        pair_at(params, "padding", &p->padding_h, &p->padding_w);
}

static int parameter_number(const json_value_t *params, const char *key,
                            float fallback, float *out) {
    const json_value_t *value = json_object_get(params, key);
    if (value == NULL) { *out = fallback; return 1; }
    if (!valid_number(value)) return 0;
    *out = (float)value->as.number;
    return isfinite(*out);
}

static int load_li(const json_value_t *params, nir_node_t *node) {
    return parameter_number(params, "tau", 1.0f, &node->params.li.tau) &&
        node->params.li.tau > 0.0f &&
        parameter_number(params, "r", 1.0f, &node->params.li.r) &&
        parameter_number(params, "v_leak", 0.0f, &node->params.li.v_leak);
}

static int load_lif(const json_value_t *params, nir_node_t *node) {
    nir_lif_params_t *p = &node->params.lif;
    return parameter_number(params, "tau", 1.0f, &p->tau) &&
        p->tau > 0.0f &&
        parameter_number(params, "r", 1.0f, &p->r) &&
        parameter_number(params, "v_leak", 0.0f, &p->v_leak) &&
        parameter_number(params, "v_threshold", 1.0f, &p->v_threshold) &&
        parameter_number(params, "v_reset", 0.0f, &p->v_reset);
}

static int load_if(const json_value_t *params, nir_node_t *node) {
    nir_if_params_t *p = &node->params.if_;
    return parameter_number(params, "r", 1.0f, &p->r) &&
        parameter_number(params, "v_threshold", 1.0f, &p->v_threshold) &&
        parameter_number(params, "v_reset", 0.0f, &p->v_reset);
}

static int load_cuba(const json_value_t *params, nir_node_t *node) {
    nir_cuba_params_t *p = &node->params.cuba;
    return parameter_number(params, "tau_syn", 1.0f, &p->tau_syn) &&
        p->tau_syn > 0.0f &&
        parameter_number(params, "tau_mem", 1.0f, &p->tau_mem) &&
        p->tau_mem > 0.0f &&
        parameter_number(params, "r", 1.0f, &p->r) &&
        parameter_number(params, "v_leak", 0.0f, &p->v_leak) &&
        parameter_number(params, "v_threshold", 1.0f, &p->v_threshold) &&
        parameter_number(params, "v_reset", 0.0f, &p->v_reset) &&
        parameter_number(params, "w_in", 1.0f, &p->w_in);
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
    {"LIF", NIR_NODE_LIF},
    {"IF", NIR_NODE_IF},
    {"CubaLIF", NIR_NODE_CUBALIF},
    {"Conv1d", NIR_NODE_CONV1D},
    {"Linear", NIR_NODE_LINEAR},
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
    if (name == NULL || name[0] == '\0' || kind_text == NULL) {
        set_error(error, "node missing name or kind", NULL);
        return 0;
    }
    node->name = dup_string(name);
    if (node->name == NULL) {
        set_error(error, "out of memory", NULL);
        return 0;
    }
    if (!resolve_kind(kind_text, &node->kind)) {
        set_error(error, "unsupported node kind: %s", kind_text);
        return 0;
    }
    const json_value_t *params = json_object_get(node_json, "params");
    switch (node->kind) {
        case NIR_NODE_CONV2D: return load_conv2d(params, node, error);
        case NIR_NODE_CONV1D: return load_conv1d(params, node, error);
        case NIR_NODE_AFFINE: return load_affine(params, node, error);
        case NIR_NODE_LINEAR: return load_linear(params, node, error);
        case NIR_NODE_AVGPOOL2D:
        case NIR_NODE_SUMPOOL2D:
            if (!load_pool2d(params, node)) {
                set_error(error, "Pool2d kernel and stride must be positive pairs and padding a nonnegative pair", NULL);
                return 0;
            }
            return 1;
        case NIR_NODE_LI:
            if (!load_li(params, node)) {
                set_error(error, "LI tau must be positive and parameters finite numbers", NULL);
                return 0;
            }
            return 1;
        case NIR_NODE_LIF:
            if (!load_lif(params, node)) {
                set_error(error, "LIF tau must be positive and parameters finite numbers", NULL);
                return 0;
            }
            return 1;
        case NIR_NODE_IF:
            if (!load_if(params, node)) {
                set_error(error, "IF parameters must be finite numbers", NULL);
                return 0;
            }
            return 1;
        case NIR_NODE_CUBALIF:
            if (!load_cuba(params, node)) {
                set_error(error, "CubaLIF time constants must be positive and parameters finite numbers", NULL);
                return 0;
            }
            return 1;
        case NIR_NODE_THRESHOLD:
            if (!parameter_number(params, "threshold", 1.0f,
                                  &node->params.threshold.threshold)) {
                set_error(error, "Threshold parameter must be a finite number", NULL);
                return 0;
            }
            return 1;
        case NIR_NODE_SCALE:
            if (!parameter_number(params, "scale", 1.0f,
                                  &node->params.scale.scale)) {
                set_error(error, "Scale parameter must be a finite number", NULL);
                return 0;
            }
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
    } else if (node->kind == NIR_NODE_CONV1D) {
        free(node->params.conv1d.weight);
        free(node->params.conv1d.bias);
    } else if (node->kind == NIR_NODE_AFFINE) {
        free(node->params.affine.weight);
        free(node->params.affine.bias);
    } else if (node->kind == NIR_NODE_LINEAR) {
        free(node->params.linear.weight);
    }
    memset(node, 0, sizeof(*node));
}

size_t nir_graph_find(const nir_graph_t *graph, const char *name) {
    if (graph == NULL || name == NULL || graph->nodes == NULL)
        return (size_t)-1;
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
    if (count > (size_t)-1 / sizeof(nir_edge_t)) {
        set_error(error, "graph has too many edges", NULL);
        return 0;
    }
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
        if (source == NULL || target == NULL) {
            set_error(error, "edge source and target must be strings", NULL);
            return 0;
        }
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
    if (error != NULL) *error = NULL;
    if (out == NULL) {
        set_error(error, "graph output must not be null", NULL);
        return 0;
    }
    memset(out, 0, sizeof(*out));
    const json_value_t *nodes_json = json_object_get(summary, "nodes");
    const json_value_t *edges_json = json_object_get(summary, "edges");
    if (nodes_json == NULL || nodes_json->type != JSON_ARRAY ||
        edges_json == NULL || edges_json->type != JSON_ARRAY) {
        set_error(error, "graph JSON missing \"nodes\" or \"edges\"", NULL);
        return 0;
    }
    size_t node_count = json_array_length(nodes_json);
    if (node_count == 0) {
        set_error(error, "graph must contain at least one node", NULL);
        return 0;
    }
    if (node_count > (size_t)-1 / sizeof(nir_node_t)) {
        set_error(error, "graph has too many nodes", NULL);
        return 0;
    }
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
        for (size_t previous = 0; previous < i; previous++) {
            if (strcmp(out->nodes[previous].name, out->nodes[i].name) == 0) {
                set_error(error, "duplicate node name: %s", out->nodes[i].name);
                nir_graph_free(out);
                return 0;
            }
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
