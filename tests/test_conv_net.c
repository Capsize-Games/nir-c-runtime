/* End-to-end validation: load the real exported NIR graph fixture,
 * run it, and compare against spikeforge's own reference
 * NirInterpreter's output on the same input (tools/export_fixture.py
 * produced both from the same run).
 */
#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/runtime.h"
#include "test_util.h"

static nir_tensor_t *load_frames(
    const json_value_t *input_json, size_t num_steps, size_t channels,
    size_t height, size_t width
) {
    nir_tensor_t *frames = (nir_tensor_t *)malloc(
        num_steps * sizeof(nir_tensor_t)
    );
    size_t shape[3] = {channels, height, width};
    for (size_t t = 0; t < num_steps; t++) {
        frames[t] = nir_tensor_alloc(shape, 3);
        const json_value_t *frame = json_array_get(input_json, t);
        size_t index = 0;
        for (size_t c = 0; c < channels; c++) {
            const json_value_t *row_c = json_array_get(frame, c);
            for (size_t h = 0; h < height; h++) {
                const json_value_t *row_h = json_array_get(row_c, h);
                for (size_t w = 0; w < width; w++) {
                    frames[t].data[index++] = (float)json_as_number(
                        json_array_get(row_h, w), 0.0
                    );
                }
            }
        }
    }
    return frames;
}

int main(void) {
    char *graph_text = read_file("fixtures/conv_net_graph.json");
    char *io_text = read_file("fixtures/conv_net_io.json");

    char *error = NULL;
    json_value_t *graph_json = json_parse(graph_text, &error);
    CHECK(error == NULL);
    json_value_t *io_json = json_parse(io_text, &error);
    CHECK(error == NULL);

    nir_graph_t graph;
    CHECK(nir_graph_load(graph_json, &graph, &error));

    const json_value_t *shape_json = json_object_get(io_json, "shape");
    size_t num_steps = (size_t)json_as_number(
        json_array_get(shape_json, 0), 0
    );
    size_t channels = (size_t)json_as_number(
        json_array_get(shape_json, 1), 0
    );
    size_t height = (size_t)json_as_number(
        json_array_get(shape_json, 2), 0
    );
    size_t width = (size_t)json_as_number(
        json_array_get(shape_json, 3), 0
    );
    nir_tensor_t *frames = load_frames(
        json_object_get(io_json, "input"), num_steps, channels, height, width
    );

    nir_runtime_t runtime;
    CHECK(nir_runtime_init(&runtime, &graph, &error));

    nir_tensor_t readout;
    CHECK(nir_runtime_run(&runtime, frames, num_steps, &readout, &error));

    const json_value_t *expected = json_object_get(io_json, "readout");
    CHECK(readout.size == json_array_length(expected));
    for (size_t i = 0; i < readout.size; i++) {
        double want = json_as_number(json_array_get(expected, i), 0.0);
        CHECK_NEAR(readout.data[i], want, 1e-3);
    }

    for (size_t t = 0; t < num_steps; t++) nir_tensor_free(&frames[t]);
    free(frames);
    nir_tensor_free(&readout);
    nir_runtime_free(&runtime);
    nir_graph_free(&graph);
    json_free(graph_json);
    json_free(io_json);
    free(graph_text);
    free(io_text);
    return 0;
}
