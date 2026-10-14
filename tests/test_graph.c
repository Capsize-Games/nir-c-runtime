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
        "{\"name\":\"a\",\"kind\":\"LIF\",\"params\":{}},"
        "{\"name\":\"b\",\"kind\":\"IF\",\"params\":{}},"
        "{\"name\":\"c\",\"kind\":\"CubaLIF\",\"params\":{}},"
        "{\"name\":\"d\",\"kind\":\"Linear\",\"params\":{\"weight\":[[1]]}},"
        "{\"name\":\"e\",\"kind\":\"Conv1d\",\"params\":{\"weight\":[[[1]]],"
        "\"stride\":1,\"padding\":0,\"dilation\":1}}],\"edges\":[]}";
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(json != NULL && error == NULL);
    nir_graph_t graph;
    CHECK(nir_graph_load(json, &graph, &error));
    CHECK(error == NULL && graph.node_count == 5);
    CHECK(graph.nodes[0].kind == NIR_NODE_LIF);
    CHECK(graph.nodes[1].kind == NIR_NODE_IF);
    CHECK(graph.nodes[2].kind == NIR_NODE_CUBALIF);
    CHECK(graph.nodes[3].kind == NIR_NODE_LINEAR);
    CHECK(graph.nodes[4].kind == NIR_NODE_CONV1D);
    CHECK(graph.nodes[4].params.conv1d.bias == NULL);
    nir_graph_free(&graph);
    json_free(json);
}

int main(void) {
    test_invalid_endpoints_are_rejected();
    test_unknown_string_endpoint_is_rejected();
    test_valid_edge_loads();
    test_malformed_graph_parameters_are_rejected();
    test_unsupported_kind_names_kind();
    test_new_kinds_load_with_defaults();
    return 0;
}
