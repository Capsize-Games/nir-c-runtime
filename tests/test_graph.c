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
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"LIF\",\"params\":{\"tau\":0}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"LIF\",\"params\":{\"v_threshold\":\"high\"}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"LIF\",\"params\":{\"v_reset\":1e999}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"LIF\",\"params\":{\"reset_mode\":\"unknown\"}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"LIF\",\"params\":{\"reset_mode\":2}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"Threshold\",\"params\":{\"threshold\":1e999}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"Scale\",\"params\":{\"scale\":[]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"n\",\"kind\":\"Scale\",\"params\":{\"scale\":1}},{\"name\":\"n\",\"kind\":\"Scale\",\"params\":{}}],\"edges\":[]}",
        "{\"nodes\":[],\"edges\":[]}",
        "{\"nodes\":{},\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"\",\"kind\":\"Input\",\"params\":{}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"f\",\"kind\":\"Affine\",\"params\":{\"weight\":{\"data\":[[128]],\"dtype\":\"int8\",\"scale\":1,\"zero_point\":0},\"bias\":[0]}}],\"edges\":[]}",
        "{\"nodes\":[{\"name\":\"f\",\"kind\":\"Affine\",\"params\":{\"weight\":{\"data\":[[1.5]],\"dtype\":\"uint8\",\"scale\":1,\"zero_point\":0},\"bias\":[0]}}],\"edges\":[]}"
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

static void test_lif_defaults_and_explicit_reset_mode(void) {
    const char *text =
        "{\"nodes\":["
        "{\"name\":\"input\",\"kind\":\"Input\",\"params\":{}},"
        "{\"name\":\"lif\",\"kind\":\"LIF\",\"params\":{}},"
        "{\"name\":\"output\",\"kind\":\"Output\",\"params\":{}}],"
        "\"edges\":[{\"source\":\"input\",\"target\":\"lif\"},"
        "{\"source\":\"lif\",\"target\":\"output\"}]}";
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(json != NULL && error == NULL);
    nir_graph_t graph;
    CHECK(nir_graph_load(json, &graph, &error));
    const nir_lif_params_t *p = &graph.nodes[1].params.lif;
    CHECK_NEAR(p->tau, 1.0f, 1e-9);
    CHECK_NEAR(p->r, 1.0f, 1e-9);
    CHECK_NEAR(p->v_leak, 0.0f, 1e-9);
    CHECK_NEAR(p->v_threshold, 1.0f, 1e-9);
    CHECK_NEAR(p->v_reset, 0.0f, 1e-9);
    CHECK(p->reset_mode == NIR_LIF_RESET_HARD);
    nir_graph_free(&graph);
    json_free(json);

    const char *subtract_text =
        "{\"nodes\":[{\"name\":\"lif\",\"kind\":\"LIF\","
        "\"params\":{\"reset_mode\":\"subtractive\"}}],"
        "\"edges\":[]}";
    json = json_parse(subtract_text, &error);
    CHECK(json != NULL && error == NULL);
    CHECK(nir_graph_load(json, &graph, &error));
    CHECK(graph.nodes[0].params.lif.reset_mode ==
          NIR_LIF_RESET_SUBTRACTIVE);
    nir_graph_free(&graph);
    json_free(json);
}

int main(void) {
    test_invalid_endpoints_are_rejected();
    test_unknown_string_endpoint_is_rejected();
    test_valid_edge_loads();
    test_malformed_graph_parameters_are_rejected();
    test_lif_defaults_and_explicit_reset_mode();
    return 0;
}
