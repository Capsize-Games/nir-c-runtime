#include <string.h>

#include "nir_c_runtime/graph.h"
#include "test_util.h"

static int load_with_edge(
    const char *edge, nir_graph_t *graph, char **error
) {
    char text[1024];
    snprintf(text, sizeof(text),
        "{\"nodes\":[{\"name\":\"input\",\"kind\":\"Input\"},"
        "{\"name\":\"output\",\"kind\":\"Output\"}],\"edges\":[%s]}", edge);
    json_value_t *json = json_parse(text, error);
    CHECK(json != NULL && *error == NULL);
    int ok = nir_graph_load(json, graph, error);
    json_free(json);
    return ok;
}

static void test_invalid_endpoints_are_rejected(void) {
    const char *invalid[] = {
        "{\"source\":null,\"target\":\"output\"}",
        "{\"source\":\"input\",\"target\":null}",
        "{\"target\":\"output\"}", "{\"source\":\"input\"}",
        "{\"source\":42,\"target\":\"output\"}",
        "{\"source\":\"input\",\"target\":false}",
        "{\"source\":[],\"target\":\"output\"}",
        "{\"source\":\"input\",\"target\":{}}", "null", "[]"
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        nir_graph_t graph;
        char *error = NULL;
        CHECK(!load_with_edge(invalid[i], &graph, &error));
        CHECK(error != NULL && strstr(error, "strings") != NULL);
        CHECK(graph.nodes == NULL && graph.node_count == 0);
        CHECK(graph.edges == NULL && graph.edge_count == 0);
        nir_graph_free(&graph);
        free(error);
    }
}

static void test_unknown_string_endpoint_is_rejected(void) {
    nir_graph_t graph;
    char *error = NULL;
    CHECK(!load_with_edge(
        "{\"source\":\"missing\",\"target\":\"output\"}", &graph, &error));
    CHECK(error != NULL && strstr(error, "unknown node") != NULL);
    nir_graph_free(&graph);
    free(error);
}

static void test_valid_edge_loads(void) {
    nir_graph_t graph;
    char *error = NULL;
    CHECK(load_with_edge(
        "{\"source\":\"input\",\"target\":\"output\"}", &graph, &error));
    CHECK(error == NULL && graph.edge_count == 1);
    CHECK(graph.edges[0].source_index == 0);
    CHECK(graph.edges[0].target_index == 1);
    nir_graph_free(&graph);
}

static void test_malformed_graph_parameters_are_rejected(void) {
    const char *invalid[] = {
        "{\"nodes\":[{\"name\":\"c\",\"kind\":\"Conv2d\",\"params\":{\"weight\":[[[[1,2]]],[[[3]]]],\"stride\":[1,1],\"padding\":[0,0],\"dilation\":[1,1]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"c\",\"kind\":\"Conv2d\",\"params\":{\"weight\":[[[[1]]]],\"stride\":[1.5,1],\"padding\":[0,0],\"dilation\":[1,1]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"p\",\"kind\":\"AvgPool2d\",\"params\":{\"kernel_size\":[2,2],\"stride\":[0,1],\"padding\":[0,0]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"f\",\"kind\":\"Affine\",\"params\":{\"weight\":[[1,2],[3]],\"bias\":[0,0]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"f\",\"kind\":\"Affine\",\"params\":{\"weight\":[[1,\"bad\"]],\"bias\":[0]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"LI\",\"params\":{\"tau\":\"slow\"}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"LI\",\"params\":{\"tau\":0}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"Threshold\",\"params\":{\"threshold\":1e999}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"Scale\",\"params\":{\"scale\":[]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"Scale\",\"params\":{\"scale\":1}},{\"name\":\"n\",\"kind\":\"Scale\",\"params\":{}}],\"edges\":[]}",
        "{\"nodes\":[],\"edges\":[]}",
        "{\"nodes\":{},\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"\",\"kind\":\"Input\",\"params\":{}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"f\",\"kind\":\"Affine\",\"params\":{\"weight\":{\"data\":[[128]],\"dtype\":\"int8\",\"scale\":1,\"zero_point\":0},\"bias\":[0]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"f\",\"kind\":\"Affine\",\"params\":{\"weight\":{\"data\":[[1.5]],\"dtype\":\"uint8\",\"scale\":1,\"zero_point\":0},\"bias\":[0]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"LIF\",\"params\":{\"tau\":0}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"LIF\",\"params\":{\"tau\":\"slow\"}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"IF\",\"params\":{\"r\":[]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"CubaLIF\",\"params\":{\"tau_syn\":0,\"tau_mem\":1}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"CubaLIF\",\"params\":{\"tau_mem\":-1}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"c\",\"kind\":\"Conv1d\",\"params\":{\"weight\":[[[1,2]],[[3]]],\"stride\":1,\"padding\":0,\"dilation\":1}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"c\",\"kind\":\"Conv1d\",\"params\":{\"weight\":[[[1]]],\"stride\":1,\"padding\":\"same\",\"dilation\":1}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"c\",\"kind\":\"Conv1d\",\"params\":{\"weight\":[[[1]]],\"stride\":1,\"padding\":0,\"dilation\":1,\"groups\":2}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"c\",\"kind\":\"Conv1d\",\"params\":{\"weight\":[[[1]]],\"stride\":0,\"padding\":0,\"dilation\":1}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"f\",\"kind\":\"Linear\",\"params\":{\"weight\":[]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"f\",\"kind\":\"Linear\",\"params\":{\"weight\":[[1],[2,3]]}}],\"edges\":[]}"
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++) {
        char *error = NULL;
        json_value_t *json = json_parse(invalid[i], &error);
        CHECK(json != NULL && error == NULL);
        nir_graph_t graph;
        CHECK(!nir_graph_load(json, &graph, &error));
        CHECK(error != NULL);
        CHECK(graph.nodes == NULL && graph.node_count == 0);
        nir_graph_free(&graph);
        json_free(json);
        free(error);
    }
}

static void test_unsupported_kind_names_kind(void) {
    const char *text =
        "{\"nodes\":[{\"name\":\"m\",\"kind\":\"Mystery\",\"params\":{}}],"
        "\"edges\":[]}";
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(json != NULL && error == NULL);
    nir_graph_t graph;
    CHECK(!nir_graph_load(json, &graph, &error));
    CHECK(error != NULL && strstr(error, "unsupported node kind") != NULL);
    CHECK(strstr(error, "Mystery") != NULL);
    CHECK(graph.nodes == NULL && graph.node_count == 0);
    nir_graph_free(&graph);
    json_free(json);
    free(error);
}

static void test_new_kinds_load_with_defaults(void) {
    const char *text =
        "{\"nodes\":["
        "{\"name\":\"input\",\"kind\":\"Input\"},"
        "{\"name\":\"a\",\"kind\":\"LIF\",\"params\":{}},"
        "{\"name\":\"b\",\"kind\":\"IF\",\"params\":{}},"
        "{\"name\":\"c\",\"kind\":\"CubaLIF\",\"params\":{}},"
        "{\"name\":\"d\",\"kind\":\"Linear\",\"params\":{\"weight\":[[1]]}},"
        "{\"name\":\"e\",\"kind\":\"Conv1d\",\"params\":{\"weight\":[[[1]]],"
        "\"stride\":1,\"padding\":0,\"dilation\":1}},"
        "{\"name\":\"output\",\"kind\":\"Output\"}],\"edges\":[]}";
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(json != NULL && error == NULL);
    nir_graph_t graph;
    CHECK(nir_graph_load(json, &graph, &error));
    CHECK(error == NULL && graph.node_count == 7);
    CHECK(graph.nodes[1].kind == NIR_NODE_LIF);
    CHECK(graph.nodes[2].kind == NIR_NODE_IF);
    CHECK(graph.nodes[3].kind == NIR_NODE_CUBALIF);
    CHECK(graph.nodes[4].kind == NIR_NODE_LINEAR);
    CHECK(graph.nodes[5].kind == NIR_NODE_CONV1D);
    CHECK(graph.nodes[5].params.conv1d.bias == NULL);
    CHECK(graph.input_index == 0 && graph.output_index == 6);
    nir_graph_free(&graph);
    json_free(json);
}

static void test_conv2d_groups(void) {
    const char *prefix =
        "{\"nodes\":[{\"name\":\"input\",\"kind\":\"Input\"},"
        "{\"name\":\"c\",\"kind\":\"Conv2d\",\"params\":{"
        "\"weight\":[[[[1]]]],\"stride\":[1,1],\"padding\":[0,0],"
        "\"dilation\":[1,1]";
    const char *suffix =
        "}},{\"name\":\"output\",\"kind\":\"Output\"}],\"edges\":[]}";
    const char *rejected[][2] = {
        {",\"groups\":2", "Conv2d groups must be 1"},
        {",\"groups\":0", "Conv2d groups must be 1"},
        {",\"groups\":1.5", "Conv2d groups must be a nonnegative integer"},
        {",\"groups\":\"two\"", "Conv2d groups must be a nonnegative integer"},
    };
    for (size_t i = 0; i < sizeof(rejected) / sizeof(rejected[0]); i++) {
        char text[512];
        snprintf(text, sizeof(text), "%s%s%s",
            prefix, rejected[i][0], suffix);
        char *error = NULL;
        json_value_t *json = json_parse(text, &error);
        CHECK(json != NULL && error == NULL);
        nir_graph_t graph;
        CHECK(!nir_graph_load(json, &graph, &error));
        CHECK(error != NULL && strstr(error, rejected[i][1]) != NULL);
        CHECK(graph.nodes == NULL && graph.node_count == 0);
        nir_graph_free(&graph);
        json_free(json);
        free(error);
    }
    const char *accepted[] = {"", ",\"groups\":1"};
    for (size_t i = 0; i < sizeof(accepted) / sizeof(accepted[0]); i++) {
        char text[512];
        snprintf(text, sizeof(text), "%s%s%s",
            prefix, accepted[i], suffix);
        char *error = NULL;
        json_value_t *json = json_parse(text, &error);
        CHECK(json != NULL && error == NULL);
        nir_graph_t graph;
        CHECK(nir_graph_load(json, &graph, &error));
        CHECK(error == NULL && graph.node_count == 3);
        CHECK(graph.input_index == 0 && graph.output_index == 2);
        nir_graph_free(&graph);
        json_free(json);
    }
}

static void test_input_output_endpoints_are_required(void) {
    const char *scale =
        "{\"name\":\"s\",\"kind\":\"Scale\",\"params\":{\"scale\":2}}";
    const char *input = "{\"name\":\"input\",\"kind\":\"Input\"}";
    const char *input2 = "{\"name\":\"input2\",\"kind\":\"Input\"}";
    const char *output = "{\"name\":\"output\",\"kind\":\"Output\"}";
    const char *output2 = "{\"name\":\"output2\",\"kind\":\"Output\"}";
    const char *shapes[][4] = {
        {scale, output, NULL, "no Input node"},
        {input, scale, NULL, "no Output node"},
        {NULL, NULL, NULL, "at least one node"},
        {input, input2, output, "multiple Input nodes"},
        {input, output, output2, "multiple Output nodes"},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        char text[1024];
        size_t at = 0;
        at += (size_t)snprintf(text + at, sizeof(text) - at, "{\"nodes\":[");
        for (size_t n = 0; shapes[i][n] != NULL && n < 3; n++) {
            at += (size_t)snprintf(text + at, sizeof(text) - at, "%s%s",
                n > 0 ? "," : "", shapes[i][n]);
        }
        snprintf(text + at, sizeof(text) - at, "],\"edges\":[]}");
        char *error = NULL;
        json_value_t *json = json_parse(text, &error);
        CHECK(json != NULL && error == NULL);
        nir_graph_t graph;
        CHECK(!nir_graph_load(json, &graph, &error));
        CHECK(error != NULL && strstr(error, shapes[i][3]) != NULL);
        CHECK(graph.nodes == NULL && graph.node_count == 0);
        CHECK(graph.edges == NULL && graph.edge_count == 0);
        nir_graph_free(&graph);
        json_free(json);
        free(error);
    }
}

int main(void) {
    test_invalid_endpoints_are_rejected();
    test_unknown_string_endpoint_is_rejected();
    test_valid_edge_loads();
    test_malformed_graph_parameters_are_rejected();
    test_unsupported_kind_names_kind();
    test_new_kinds_load_with_defaults();
    test_conv2d_groups();
    test_input_output_endpoints_are_required();
    return 0;
}
