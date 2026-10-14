/* Initialized weights land on a graph node through
 * nir_graph_set_weights and run forward cleanly (#45). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/checkpoint.h"
#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/init.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/runtime.h"
#include "test_util.h"

static nir_graph_t load_fixture(const char *path) {
    nir_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    char *text = read_file(path);
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(error == NULL);
    CHECK(nir_graph_load(json, &graph, &error));
    CHECK(error == NULL);
    json_free(json);
    free(text);
    return graph;
}

static void fill_frames(nir_tensor_t frames[6]) {
    size_t shape[1] = {4};
    for (int t = 0; t < 6; t++) {
        frames[t] = nir_tensor_alloc(shape, 1);
        for (size_t i = 0; i < frames[t].size; i++) {
            frames[t].data[i] = (float)(t + 1) * 0.1f;
        }
    }
}

static void check_forward(nir_graph_t *graph) {
    nir_runtime_t runtime;
    char *error = NULL;
    CHECK(nir_runtime_init(&runtime, graph, &error));
    nir_tensor_t frames[6];
    fill_frames(frames);
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(nir_runtime_run(&runtime, frames, 6, &readout, &error));
    CHECK(error == NULL);
    CHECK(readout.size == 2);
    for (size_t i = 0; i < readout.size; i++) {
        CHECK(isfinite(readout.data[i]));
    }
    for (int t = 0; t < 6; t++) nir_tensor_free(&frames[t]);
    nir_tensor_free(&readout);
    nir_runtime_free(&runtime);
}

static void check_fc1_matches(
    nir_graph_t *graph, const float *fresh, size_t wn
) {
    const float *w = NULL;
    const float *b = NULL;
    size_t wn_back = 0;
    size_t bn = 0;
    CHECK(nir_graph_get_weights(
        graph, "fc1", &w, &wn_back, &b, &bn
    ));
    CHECK(wn_back == wn);
    CHECK(memcmp(w, fresh, wn * sizeof(float)) == 0);
}

static void set_fc1_xavier(nir_graph_t *graph, size_t wn) {
    size_t in = 0;
    size_t out = 0;
    size_t shape[2] = {3, 4};
    CHECK(nir_init_fans(shape, 2, &in, &out));
    float *fresh = (float *)malloc(wn * sizeof(float));
    CHECK(fresh != NULL);
    nir_rng_t rng;
    nir_rng_seed(&rng, 42);
    CHECK(nir_init_xavier_uniform(&rng, fresh, wn, in, out, 1.0f));
    char *error = NULL;
    CHECK(nir_graph_set_weights(
        graph, "fc1", fresh, wn, NULL, 0, &error
    ));
    CHECK(error == NULL);
    check_fc1_matches(graph, fresh, wn);
    free(fresh);
}

int main(void) {
    nir_graph_t graph = load_fixture("fixtures/lif_linear_graph.json");
    const float *w = NULL;
    const float *b = NULL;
    size_t wn = 0;
    size_t bn = 0;
    CHECK(nir_graph_get_weights(&graph, "fc1", &w, &wn, &b, &bn));
    CHECK(wn == 12 && b == NULL && bn == 0);
    set_fc1_xavier(&graph, wn);
    check_forward(&graph);
    nir_graph_free(&graph);
    return 0;
}
