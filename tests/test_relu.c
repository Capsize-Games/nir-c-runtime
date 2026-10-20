#include <math.h>
#include <string.h>

#include "nir_c_runtime/ops.h"
#include "nir_c_runtime/runtime.h"
#include "test_util.h"

static void test_relu_operation_and_alias_rules(void) {
    size_t shape[2] = {2, 4};
    nir_tensor_t input = nir_tensor_alloc(shape, 2);
    CHECK(input.data != NULL);
    input.data[0] = -INFINITY;
    input.data[1] = -2.0f;
    input.data[2] = -0.0f;
    input.data[3] = 0.0f;
    input.data[4] = 0.5f;
    input.data[5] = 2.0f;
    input.data[6] = INFINITY;
    input.data[7] = NAN;

    nir_tensor_t output = nir_op_relu(&input);
    CHECK(output.data != NULL && output.ndim == 2);
    CHECK(output.shape[0] == 2 && output.shape[1] == 4);
    CHECK(output.data[0] == 0.0f && output.data[1] == 0.0f);
    CHECK(output.data[2] == 0.0f && output.data[3] == 0.0f);
    CHECK(output.data[4] == 0.5f && output.data[5] == 2.0f);
    CHECK(isinf(output.data[6]) && output.data[6] > 0.0f);
    CHECK(isnan(output.data[7]));

    CHECK(nir_op_relu_into(&input, &input));
    CHECK(input.data[0] == 0.0f && input.data[4] == 0.5f);
    nir_tensor_free(&output);
    nir_tensor_free(&input);
}

static void test_relu_into_rejects_partial_overlap_and_small_capacity(void) {
    float data[5] = {-1.0f, 2.0f, -3.0f, 0.0f, 0.0f};
    size_t shape[1] = {3};
    nir_tensor_t input = {0}, output = {0};
    CHECK(nir_tensor_bind(&input, data, 5, shape, 1));
    CHECK(nir_tensor_bind(&output, data + 1, 4, shape, 1));
    CHECK(!nir_op_relu_into(&input, &output));
    CHECK(data[0] == -1.0f && data[1] == 2.0f && data[2] == -3.0f);

    output = (nir_tensor_t){
        .data = data + 3,
        .capacity = 2,
        .ndim = 1,
        .shape = {3},
        .size = 3
    };
    CHECK(!nir_op_relu_into(&input, &output));
    CHECK(data[0] == -1.0f && data[1] == 2.0f && data[2] == -3.0f);
}

static void test_graph_loader_and_both_runtime_paths(void) {
    CHECK(NIR_NODE_DELAY == 10 && NIR_NODE_RELU == 11);
    const char *text =
        "{\"nodes\":["
        "{\"name\":\"input\",\"kind\":\"Input\"},"
        "{\"name\":\"relu\",\"kind\":\"ReLU\"},"
        "{\"name\":\"output\",\"kind\":\"Output\"}],"
        "\"edges\":["
        "{\"source\":\"input\",\"target\":\"relu\"},"
        "{\"source\":\"relu\",\"target\":\"output\"}]}";
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(json != NULL && error == NULL);
    nir_graph_t graph = {0};
    CHECK(nir_graph_load(json, &graph, &error));
    CHECK(graph.nodes[1].kind == NIR_NODE_RELU);
    json_free(json);

    nir_runtime_t allocating = {0};
    CHECK(nir_runtime_init(&allocating, &graph, &error));
    size_t shape[1] = {3};
    nir_tensor_t frame = nir_tensor_alloc(shape, 1);
    frame.data[0] = -1.0f;
    frame.data[1] = 0.25f;
    frame.data[2] = 4.0f;
    nir_tensor_t readout = {0};
    CHECK(nir_runtime_run(&allocating, &frame, 1, &readout, &error));
    CHECK(readout.size == 3 && readout.data[0] == 0.0f &&
          readout.data[1] == 0.25f && readout.data[2] == 4.0f);
    nir_tensor_free(&readout);
    nir_runtime_free(&allocating);

    nir_runtime_t workspace_runtime = {0};
    CHECK(nir_runtime_init(&workspace_runtime, &graph, &error));
    nir_tensor_t values[3] = {{0}}, li_state[3] = {{0}},
        delay_buffer[3] = {{0}}, sum = {0}, total = {0}, workspace_readout = {0};
    for (size_t i = 0; i < 3; i++) {
        values[i] = nir_tensor_alloc(shape, 1);
        li_state[i] = nir_tensor_alloc(shape, 1);
        delay_buffer[i] = nir_tensor_alloc(shape, 1);
    }
    sum = nir_tensor_alloc(shape, 1);
    total = nir_tensor_alloc(shape, 1);
    workspace_readout = nir_tensor_alloc(shape, 1);
    int li_primed[3] = {0}, delay_primed[3] = {0};
    nir_runtime_workspace_t workspace = {
        .values = values,
        .li_state = li_state,
        .li_primed = li_primed,
        .delay_buffer = delay_buffer,
        .delay_primed = delay_primed,
        .sum = &sum,
        .total = &total,
        .node_capacity = 3
    };
    CHECK(nir_runtime_run_workspace(
        &workspace_runtime, &frame, 1, &workspace, &workspace_readout, &error
    ));
    CHECK(workspace_readout.data[0] == 0.0f &&
          workspace_readout.data[1] == 0.25f &&
          workspace_readout.data[2] == 4.0f);

    nir_tensor_free(&workspace_readout);
    nir_tensor_free(&total);
    nir_tensor_free(&sum);
    for (size_t i = 0; i < 3; i++) {
        nir_tensor_free(&values[i]);
        nir_tensor_free(&li_state[i]);
        nir_tensor_free(&delay_buffer[i]);
    }
    nir_runtime_free(&workspace_runtime);
    nir_tensor_free(&frame);
    nir_graph_free(&graph);
    free(error);
}

int main(void) {
    test_relu_operation_and_alias_rules();
    test_relu_into_rejects_partial_overlap_and_small_capacity();
    test_graph_loader_and_both_runtime_paths();
    return 0;
}
