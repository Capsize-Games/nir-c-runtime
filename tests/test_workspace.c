#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/runtime.h"
#include "test_util.h"

static void bind_vector(nir_tensor_t *tensor, float *data, size_t size) {
    size_t shape[1] = {size};
    CHECK(nir_tensor_bind(tensor, data, size, shape, 1));
}

static void test_workspace_matches_allocating_runtime(void) {
    const char *text =
        "{\"nodes\":["
        "{\"name\":\"input\",\"kind\":\"Input\",\"params\":{}},"
        "{\"name\":\"fc\",\"kind\":\"Affine\",\"params\":{"
        "\"weight\":[[1,2],[-1,1]],\"bias\":[0.5,-0.5]}},"
        "{\"name\":\"output\",\"kind\":\"Output\",\"params\":{}}],"
        "\"edges\":[{\"source\":\"input\",\"target\":\"fc\"},"
        "{\"source\":\"fc\",\"target\":\"output\"}]}";
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(error == NULL);
    nir_graph_t graph;
    CHECK(nir_graph_load(json, &graph, &error));

    size_t frame_shape[1] = {2};
    nir_tensor_t frames[2] = {
        nir_tensor_alloc(frame_shape, 1), nir_tensor_alloc(frame_shape, 1)
    };
    frames[0].data[0] = 1.0f;
    frames[0].data[1] = 2.0f;
    frames[1].data[0] = -1.0f;
    frames[1].data[1] = 1.0f;

    nir_runtime_t reference;
    nir_runtime_t realtime;
    CHECK(nir_runtime_init(&reference, &graph, &error));
    CHECK(nir_runtime_init(&realtime, &graph, &error));
    nir_tensor_t expected;
    CHECK(nir_runtime_run(&reference, frames, 2, &expected, &error));

    float values_data[3][2] = {{0}};
    float li_data[3][2] = {{0}};
    float delay_data[3][2] = {{0}};
    float sum_data[2] = {0};
    float total_data[2] = {0};
    float actual_data[2] = {0};
    nir_tensor_t values[3] = {0};
    nir_tensor_t li_state[3] = {0};
    nir_tensor_t delay_buffer[3] = {0};
    for (size_t i = 0; i < 3; i++) {
        bind_vector(&values[i], values_data[i], 2);
        bind_vector(&li_state[i], li_data[i], 2);
        bind_vector(&delay_buffer[i], delay_data[i], 2);
    }
    nir_tensor_t sum = {0};
    nir_tensor_t total = {0};
    nir_tensor_t actual = {0};
    bind_vector(&sum, sum_data, 2);
    bind_vector(&total, total_data, 2);
    bind_vector(&actual, actual_data, 2);
    int li_primed[3] = {0};
    int delay_primed[3] = {0};
    nir_runtime_workspace_t workspace = {
        .values = values,
        .li_state = li_state,
        .li_primed = li_primed,
        .delay_buffer = delay_buffer,
        .delay_primed = delay_primed,
        .sum = &sum,
        .total = &total,
        .node_capacity = 3,
    };
    CHECK(nir_runtime_run_workspace(
        &realtime, frames, 2, &workspace, &actual, &error
    ));
    CHECK(actual.size == expected.size);
    for (size_t i = 0; i < actual.size; i++) {
        CHECK_NEAR(actual.data[i], expected.data[i], 1e-6);
    }

    nir_tensor_free(&actual);
    nir_tensor_free(&total);
    nir_tensor_free(&sum);
    for (size_t i = 0; i < 3; i++) {
        nir_tensor_free(&delay_buffer[i]);
        nir_tensor_free(&li_state[i]);
        nir_tensor_free(&values[i]);
    }
    nir_tensor_free(&expected);
    nir_tensor_free(&frames[0]);
    nir_tensor_free(&frames[1]);
    nir_runtime_free(&realtime);
    nir_runtime_free(&reference);
    nir_graph_free(&graph);
    json_free(json);
}

static void test_workspace_preserves_delay_and_li_state(void) {
    const char *text =
        "{\"nodes\":["
        "{\"name\":\"input\",\"kind\":\"Input\",\"params\":{}},"
        "{\"name\":\"delay\",\"kind\":\"Delay\",\"params\":{}},"
        "{\"name\":\"li\",\"kind\":\"LI\",\"params\":{"
        "\"tau\":1,\"r\":1,\"v_leak\":0}},"
        "{\"name\":\"output\",\"kind\":\"Output\",\"params\":{}}],"
        "\"edges\":[{\"source\":\"input\",\"target\":\"delay\"},"
        "{\"source\":\"delay\",\"target\":\"li\"},"
        "{\"source\":\"li\",\"target\":\"output\"}]}";
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(error == NULL);
    nir_graph_t graph;
    CHECK(nir_graph_load(json, &graph, &error));
    size_t shape[1] = {1};
    nir_tensor_t frames[2] = {
        nir_tensor_alloc(shape, 1), nir_tensor_alloc(shape, 1)
    };
    frames[0].data[0] = 2.0f;
    frames[1].data[0] = 4.0f;
    nir_runtime_t reference;
    nir_runtime_t realtime;
    CHECK(nir_runtime_init(&reference, &graph, &error));
    CHECK(nir_runtime_init(&realtime, &graph, &error));
    nir_tensor_t expected;
    CHECK(nir_runtime_run(&reference, frames, 2, &expected, &error));

    float values_data[4] = {0};
    float li_data[4] = {0};
    float delay_data[4] = {0};
    float sum_data[1] = {0};
    float total_data[1] = {0};
    float actual_data[1] = {0};
    nir_tensor_t values[4] = {0};
    nir_tensor_t li_state[4] = {0};
    nir_tensor_t delay_buffer[4] = {0};
    for (size_t i = 0; i < 4; i++) {
        bind_vector(&values[i], &values_data[i], 1);
        bind_vector(&li_state[i], &li_data[i], 1);
        bind_vector(&delay_buffer[i], &delay_data[i], 1);
    }
    nir_tensor_t sum = {0};
    nir_tensor_t total = {0};
    nir_tensor_t actual = {0};
    bind_vector(&sum, sum_data, 1);
    bind_vector(&total, total_data, 1);
    bind_vector(&actual, actual_data, 1);
    int li_primed[4] = {0};
    int delay_primed[4] = {0};
    nir_runtime_workspace_t workspace = {
        .values = values,
        .li_state = li_state,
        .li_primed = li_primed,
        .delay_buffer = delay_buffer,
        .delay_primed = delay_primed,
        .sum = &sum,
        .total = &total,
        .node_capacity = 4,
    };
    CHECK(nir_runtime_run_workspace(
        &realtime, frames, 2, &workspace, &actual, &error
    ));
    CHECK_NEAR(actual.data[0], expected.data[0], 1e-6);

    nir_runtime_workspace_reset(&workspace);
    CHECK(li_primed[2] == 0 && delay_primed[1] == 0);
    nir_tensor_free(&actual);
    nir_tensor_free(&total);
    nir_tensor_free(&sum);
    for (size_t i = 0; i < 4; i++) {
        nir_tensor_free(&delay_buffer[i]);
        nir_tensor_free(&li_state[i]);
        nir_tensor_free(&values[i]);
    }
    nir_tensor_free(&expected);
    nir_tensor_free(&frames[0]);
    nir_tensor_free(&frames[1]);
    nir_runtime_free(&realtime);
    nir_runtime_free(&reference);
    nir_graph_free(&graph);
    json_free(json);
}

int main(void) {
    test_workspace_matches_allocating_runtime();
    test_workspace_preserves_delay_and_li_state();
    return 0;
}
