#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "test_util.h"

static void test_quantized_weights_are_dequantized(void) {
    const char *text =
        "{\"nodes\":["
        "{\"name\":\"input\",\"kind\":\"Input\",\"params\":{}},"
        "{\"name\":\"fc\",\"kind\":\"Affine\",\"params\":{"
        "\"weight\":{\"data\":[[-2,2]],\"dtype\":\"int8\","
        "\"scale\":0.5,\"zero_point\":0},"
        "\"bias\":{\"data\":[130],\"dtype\":\"uint8\","
        "\"scale\":0.5,\"zero_point\":128}}},"
        "{\"name\":\"output\",\"kind\":\"Output\",\"params\":{}}],"
        "\"edges\":[{\"source\":\"input\",\"target\":\"fc\"},"
        "{\"source\":\"fc\",\"target\":\"output\"}]}";
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(error == NULL);
    nir_graph_t graph;
    CHECK(nir_graph_load(json, &graph, &error));
    CHECK_NEAR(graph.nodes[1].params.affine.weight[0], -1.0, 1e-6);
    CHECK_NEAR(graph.nodes[1].params.affine.weight[1], 1.0, 1e-6);
    CHECK_NEAR(graph.nodes[1].params.affine.bias[0], 1.0, 1e-6);
    nir_graph_free(&graph);
    json_free(json);
}

int main(void) {
    test_quantized_weights_are_dequantized();
    return 0;
}
