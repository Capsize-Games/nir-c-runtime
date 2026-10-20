#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/checkpoint.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/ops.h"
#include "nir_c_runtime/runtime.h"
#include "test_util.h"

static nir_tensor_t vector(const float *values, size_t count) {
    size_t shape[1] = {count};
    nir_tensor_t tensor = nir_tensor_alloc(shape, 1);
    CHECK(tensor.data != NULL);
    memcpy(tensor.data, values, count * sizeof(float));
    return tensor;
}

static void test_relu_forward_and_backward(void) {
    float values[] = {-INFINITY, -2.0f, -0.0f, 0.0f, 0.5f, 2.0f,
                      INFINITY, NAN};
    float upstream[] = {1, 2, 3, 4, 5, 6, 7, 8};
    nir_tensor_t input = vector(values, 8);
    nir_tensor_t grad_output = vector(upstream, 8);
    nir_tensor_t output = nir_op_relu(&input);
    CHECK(output.data != NULL && output.ndim == 1 && output.size == 8);
    CHECK(output.data[0] == 0.0f && output.data[1] == 0.0f);
    CHECK(output.data[2] == 0.0f && output.data[3] == 0.0f);
    CHECK(output.data[4] == 0.5f && output.data[5] == 2.0f);
    CHECK(isinf(output.data[6]) && output.data[6] > 0.0f);
    CHECK(isnan(output.data[7]));

    nir_tensor_t grad_input = nir_op_relu(&input);
    CHECK(nir_op_relu_backward(&input, &grad_output, &grad_input));
    float want[] = {0, 0, 0, 0, 5, 6, 7, 0};
    for (size_t i = 0; i < 8; i++) {
        CHECK(grad_input.data[i] == want[i]);
    }
    CHECK(nir_op_relu_backward(&input, &grad_output, &grad_output));
    CHECK(grad_output.data[0] == 0.0f && grad_output.data[4] == 5.0f);

    nir_tensor_free(&grad_input);
    nir_tensor_free(&output);
    nir_tensor_free(&grad_output);
    nir_tensor_free(&input);
}

static void test_relu_graph_loader_and_runtime(void) {
    const char *text =
        "{\"nodes\":["
        "{\"name\":\"input\",\"kind\":\"Input\"},"
        "{\"name\":\"relu\",\"kind\":\"ReLU\",\"params\":{}},"
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
    CHECK(NIR_NODE_RELU == NIR_NODE_LINEAR + 1);
    json_free(json);

    char *summary = nir_graph_export_summary(&graph, &error);
    CHECK(summary != NULL && error == NULL);
    json = json_parse(summary, &error);
    CHECK(json != NULL && error == NULL);
    nir_graph_t round_trip = {0};
    CHECK(nir_graph_load(json, &round_trip, &error));
    CHECK(round_trip.nodes[1].kind == NIR_NODE_RELU);
    nir_graph_free(&round_trip);
    json_free(json);
    free(summary);

    nir_runtime_t runtime = {0};
    CHECK(nir_runtime_init(&runtime, &graph, &error));
    float frame_data[] = {-1.0f, 0.25f, 4.0f};
    nir_tensor_t frame = vector(frame_data, 3), readout = {0};
    CHECK(nir_runtime_run(&runtime, &frame, 1, &readout, &error));
    CHECK(readout.data[0] == 0.0f && readout.data[1] == 0.25f &&
          readout.data[2] == 4.0f);

    float values_data[3][3] = {{0}}, li_data[3][3] = {{0}},
        delay_data[3][3] = {{0}}, sum_data[3] = {0},
        total_data[3] = {0}, workspace_output_data[3] = {0};
    nir_tensor_t values[3] = {0}, li_state[3] = {0}, delay_buffer[3] = {0};
    for (size_t i = 0; i < 3; i++) {
        CHECK(nir_tensor_bind(&values[i], values_data[i], 3,
            frame.shape, frame.ndim));
        CHECK(nir_tensor_bind(&li_state[i], li_data[i], 3,
            frame.shape, frame.ndim));
        CHECK(nir_tensor_bind(&delay_buffer[i], delay_data[i], 3,
            frame.shape, frame.ndim));
    }
    nir_tensor_t sum = {0}, total = {0}, workspace_output = {0};
    CHECK(nir_tensor_bind(&sum, sum_data, 3, frame.shape, frame.ndim));
    CHECK(nir_tensor_bind(&total, total_data, 3, frame.shape, frame.ndim));
    CHECK(nir_tensor_bind(&workspace_output, workspace_output_data, 3,
        frame.shape, frame.ndim));
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
        &runtime, &frame, 1, &workspace, &workspace_output, &error
    ));
    CHECK(memcmp(readout.data, workspace_output.data,
        3 * sizeof(float)) == 0);

    nir_tensor_free(&workspace_output);
    nir_tensor_free(&total);
    nir_tensor_free(&sum);
    for (size_t i = 0; i < 3; i++) {
        nir_tensor_free(&values[i]);
        nir_tensor_free(&li_state[i]);
        nir_tensor_free(&delay_buffer[i]);
    }
    nir_tensor_free(&readout);
    nir_tensor_free(&frame);
    nir_runtime_free(&runtime);
    nir_graph_free(&graph);
    free(error);
}

int main(void) {
    test_relu_forward_and_backward();
    test_relu_graph_loader_and_runtime();
    return 0;
}
