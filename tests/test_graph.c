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

int main(void) {
    test_invalid_endpoints_are_rejected();
    test_unknown_string_endpoint_is_rejected();
    test_valid_edge_loads();
    return 0;
}
