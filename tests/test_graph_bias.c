#include <string.h>

#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/ops.h"
#include "test_util.h"

typedef struct {
    const char *name;
    const char *kind;
    const char *bias;
    int accepted;
} bias_case_t;

static const bias_case_t CASES[] = {
    {"affine-short", "Affine", ",\"bias\":[0]", 0},
    {"affine-long", "Affine", ",\"bias\":[0,0,0]", 0},
    {"affine-empty", "Affine", ",\"bias\":[]", 0},
    {"affine-invalid", "Affine", ",\"bias\":\"invalid\"", 0},
    {"affine-exact", "Affine", ",\"bias\":[0.5,-0.5]", 1},
    {"conv-short", "Conv2d", ",\"bias\":[0]", 0},
    {"conv-long", "Conv2d", ",\"bias\":[0,0,0]", 0},
    {"conv-empty", "Conv2d", ",\"bias\":[]", 0},
    {"conv-invalid", "Conv2d", ",\"bias\":\"invalid\"", 0},
    {"conv-exact", "Conv2d", ",\"bias\":[0.5,-0.5]", 1},
    {"conv-omitted", "Conv2d", "", 1},
    {"conv-null", "Conv2d", ",\"bias\":null", 1},
    {"conv1d-short", "Conv1d", ",\"bias\":[0]", 0},
    {"conv1d-long", "Conv1d", ",\"bias\":[0,0,0]", 0},
    {"conv1d-empty", "Conv1d", ",\"bias\":[]", 0},
    {"conv1d-invalid", "Conv1d", ",\"bias\":\"invalid\"", 0},
    {"conv1d-exact", "Conv1d", ",\"bias\":[0.5,-0.5]", 1},
    {"conv1d-omitted", "Conv1d", "", 1},
    {"conv1d-null", "Conv1d", ",\"bias\":null", 1},
};

static void check_accepted_output(const nir_node_t *node) {
    size_t shape[] = {1, 1, 1};
    size_t ndim = 3;
    if (node->kind == NIR_NODE_AFFINE) {
        ndim = 1;
    } else if (node->kind == NIR_NODE_CONV1D) {
        ndim = 2;
    }
    nir_tensor_t input = nir_tensor_alloc(shape, ndim);
    CHECK(input.data != NULL);
    input.data[0] = 3;
    nir_tensor_t output;
    const float *bias;
    if (node->kind == NIR_NODE_AFFINE) {
        output = nir_op_affine(&node->params.affine, &input);
        bias = node->params.affine.bias;
    } else if (node->kind == NIR_NODE_CONV1D) {
        output = nir_op_conv1d(&node->params.conv1d, &input);
        bias = node->params.conv1d.bias;
    } else {
        output = nir_op_conv2d(&node->params.conv2d, &input);
        bias = node->params.conv2d.bias;
    }
    CHECK(output.data != NULL && output.size == 2);
    CHECK_NEAR(output.data[0], 3 + (bias ? 0.5 : 0), 1e-6);
    CHECK_NEAR(output.data[1], 6 - (bias ? 0.5 : 0), 1e-6);
    nir_tensor_free(&output);
    nir_tensor_free(&input);
}

static void check_case(const bias_case_t *test) {
    char text[1024];
    const char *weight = "[[[[1]]],[[[2]]]]";
    const char *geometry =
        "\"stride\":[1,1],\"padding\":[0,0],\"dilation\":[1,1]";
    if (strcmp(test->kind, "Affine") == 0) {
        weight = "[[1],[2]]";
    } else if (strcmp(test->kind, "Conv1d") == 0) {
        weight = "[[[1]],[[2]]]";
        geometry = "\"stride\":1,\"padding\":0,\"dilation\":1,\"groups\":1";
    }
    snprintf(text, sizeof(text),
        "{\"nodes\":[{\"name\":\"input\",\"kind\":\"Input\"},"
        "{\"name\":\"layer\",\"kind\":\"%s\","
        "\"params\":{\"weight\":%s%s,%s}},"
        "{\"name\":\"output\",\"kind\":\"Output\"}],\"edges\":[]}",
        test->kind, weight, test->bias, geometry);
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(json != NULL && error == NULL);
    nir_graph_t graph;
    CHECK(nir_graph_load(json, &graph, &error) == test->accepted);
    if (test->accepted) {
        CHECK(error == NULL && graph.node_count == 3);
        CHECK(graph.input_index == 0 && graph.output_index == 2);
        check_accepted_output(&graph.nodes[1]);
    } else {
        CHECK(error != NULL && strstr(error, "bias") != NULL);
        CHECK(graph.nodes == NULL && graph.node_count == 0);
        CHECK(graph.edges == NULL && graph.edge_count == 0);
    }
    nir_graph_free(&graph);
    json_free(json);
    free(error);
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        if (strcmp(argv[1], CASES[i].name) == 0) {
            check_case(&CASES[i]);
            return 0;
        }
    }
    fprintf(stderr, "unknown case: %s\n", argv[1]);
    return 1;
}
