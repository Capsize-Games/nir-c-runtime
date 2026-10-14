/* End-to-end validation for the neuron/linear kinds: load each
 * exported NIR graph fixture, run it through both the allocating and
 * the workspace runtimes, and compare against spikeforge's own
 * reference NirInterpreter output on the same input
 * (tools/export_fixture.py produced both from the same run).
 */
#include <stdio.h>
#include <string.h>

#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/runtime.h"
#include "test_util.h"

/* Every fixture tensor fits in this many floats; the fixtures are
 * tiny by design (see tools/export_fixture.py). */
#define FIXTURE_CAPACITY 64

static void fill_nested(
    const json_value_t *json, const size_t *shape, size_t ndim,
    float *data, size_t *index
) {
    if (ndim == 0) {
        data[(*index)++] = (float)json_as_number(json, 0.0);
        return;
    }
    for (size_t i = 0; i < shape[0]; i++) {
        fill_nested(
            json_array_get(json, i), shape + 1, ndim - 1, data, index
        );
    }
}

static nir_tensor_t *load_frames(
    const json_value_t *input_json, size_t num_steps,
    const size_t *shape, size_t ndim
) {
    nir_tensor_t *frames = (nir_tensor_t *)malloc(
        num_steps * sizeof(nir_tensor_t)
    );
    CHECK(frames != NULL);
    for (size_t t = 0; t < num_steps; t++) {
        frames[t] = nir_tensor_alloc(shape, ndim);
        CHECK(frames[t].data != NULL);
        size_t index = 0;
        fill_nested(
            json_array_get(input_json, t), shape, ndim,
            frames[t].data, &index
        );
        CHECK(index == frames[t].size);
    }
    return frames;
}

static double max_abs_diff(
    const nir_tensor_t *actual, const json_value_t *expected
) {
    double worst = 0.0;
    CHECK(actual->size == json_array_length(expected));
    for (size_t i = 0; i < actual->size; i++) {
        double want = json_as_number(json_array_get(expected, i), 0.0);
        double diff = actual->data[i] > want
            ? actual->data[i] - want : want - actual->data[i];
        if (diff > worst) {
            worst = diff;
        }
        CHECK_NEAR(actual->data[i], want, 1e-6);
    }
    return worst;
}

static void bind_slot(nir_tensor_t *tensor, float *data) {
    size_t shape[1] = {FIXTURE_CAPACITY};
    CHECK(nir_tensor_bind(tensor, data, FIXTURE_CAPACITY, shape, 1));
}

static void check_workspace_matches(
    nir_runtime_t *runtime, nir_graph_t *graph, nir_tensor_t *frames,
    size_t num_steps, const nir_tensor_t *expected, const char *stem
) {
    size_t n = graph->node_count;
    nir_tensor_t *values = (nir_tensor_t *)calloc(n, sizeof(nir_tensor_t));
    nir_tensor_t *li_state = (nir_tensor_t *)calloc(n, sizeof(nir_tensor_t));
    nir_tensor_t *syn_state = (nir_tensor_t *)calloc(
        n, sizeof(nir_tensor_t)
    );
    nir_tensor_t *delay = (nir_tensor_t *)calloc(n, sizeof(nir_tensor_t));
    int *li_primed = (int *)calloc(n, sizeof(int));
    int *delay_primed = (int *)calloc(n, sizeof(int));
    float *storage = (float *)calloc(
        n * 4 * FIXTURE_CAPACITY, sizeof(float)
    );
    CHECK(values && li_state && syn_state && delay);
    CHECK(li_primed && delay_primed && storage);
    for (size_t i = 0; i < n; i++) {
        bind_slot(&values[i], storage + (i * 4) * FIXTURE_CAPACITY);
        bind_slot(&li_state[i], storage + (i * 4 + 1) * FIXTURE_CAPACITY);
        bind_slot(&syn_state[i], storage + (i * 4 + 2) * FIXTURE_CAPACITY);
        bind_slot(&delay[i], storage + (i * 4 + 3) * FIXTURE_CAPACITY);
    }
    nir_tensor_t sum = {0}, total = {0}, actual = {0};
    float sum_data[FIXTURE_CAPACITY] = {0};
    float total_data[FIXTURE_CAPACITY] = {0};
    float actual_data[FIXTURE_CAPACITY] = {0};
    bind_slot(&sum, sum_data);
    bind_slot(&total, total_data);
    bind_slot(&actual, actual_data);
    nir_runtime_workspace_t workspace = {
        .values = values,
        .li_state = li_state,
        .li_primed = li_primed,
        .syn_state = syn_state,
        .delay_buffer = delay,
        .delay_primed = delay_primed,
        .sum = &sum,
        .total = &total,
        .node_capacity = n,
    };

    char *error = NULL;
    CHECK(nir_runtime_run_workspace(
        runtime, frames, num_steps, &workspace, &actual, &error
    ));
    CHECK(actual.size == expected->size);
    double worst = 0.0;
    for (size_t i = 0; i < actual.size; i++) {
        double diff = actual.data[i] > expected->data[i]
            ? actual.data[i] - expected->data[i]
            : expected->data[i] - actual.data[i];
        if (diff > worst) {
            worst = diff;
        }
        CHECK_NEAR(actual.data[i], expected->data[i], 1e-9);
    }
    printf("%s: workspace-vs-alloc max abs diff %g\n", stem, worst);

    nir_tensor_free(&actual);
    nir_tensor_free(&total);
    nir_tensor_free(&sum);
    free(storage);
    free(delay_primed);
    free(li_primed);
    for (size_t i = 0; i < n; i++) {
        nir_tensor_free(&delay[i]);
        nir_tensor_free(&syn_state[i]);
        nir_tensor_free(&li_state[i]);
        nir_tensor_free(&values[i]);
    }
    free(delay);
    free(syn_state);
    free(li_state);
    free(values);
}

static void check_fixture(const char *stem) {
    char graph_path[128], io_path[128];
    snprintf(graph_path, sizeof(graph_path), "fixtures/%s_graph.json", stem);
    snprintf(io_path, sizeof(io_path), "fixtures/%s_io.json", stem);
    char *graph_text = read_file(graph_path);
    char *io_text = read_file(io_path);

    char *error = NULL;
    json_value_t *graph_json = json_parse(graph_text, &error);
    CHECK(error == NULL);
    json_value_t *io_json = json_parse(io_text, &error);
    CHECK(error == NULL);

    nir_graph_t graph;
    CHECK(nir_graph_load(graph_json, &graph, &error));

    const json_value_t *shape_json = json_object_get(io_json, "shape");
    size_t dims = json_array_length(shape_json);
    CHECK(dims >= 2 && dims <= 1 + NIR_TENSOR_MAX_DIMS);
    size_t num_steps = (size_t)json_as_number(
        json_array_get(shape_json, 0), 0
    );
    size_t shape[NIR_TENSOR_MAX_DIMS];
    for (size_t i = 0; i < dims - 1; i++) {
        shape[i] = (size_t)json_as_number(
            json_array_get(shape_json, i + 1), 0
        );
    }
    nir_tensor_t *frames = load_frames(
        json_object_get(io_json, "input"), num_steps, shape, dims - 1
    );

    nir_runtime_t runtime;
    CHECK(nir_runtime_init(&runtime, &graph, &error));
    nir_tensor_t readout;
    CHECK(nir_runtime_run(&runtime, frames, num_steps, &readout, &error));
    double worst = max_abs_diff(
        &readout, json_object_get(io_json, "readout")
    );
    printf("%s: C-vs-reference max abs diff %g\n", stem, worst);

    nir_runtime_t realtime;
    CHECK(nir_runtime_init(&realtime, &graph, &error));
    check_workspace_matches(
        &realtime, &graph, frames, num_steps, &readout, stem
    );

    for (size_t t = 0; t < num_steps; t++) nir_tensor_free(&frames[t]);
    free(frames);
    nir_tensor_free(&readout);
    nir_runtime_free(&realtime);
    nir_runtime_free(&runtime);
    nir_graph_free(&graph);
    json_free(graph_json);
    json_free(io_json);
    free(graph_text);
    free(io_text);
}

int main(void) {
    check_fixture("lif_linear");
    check_fixture("cuba_conv");
    check_fixture("if_net");
    return 0;
}
