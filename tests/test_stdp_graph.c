/* Graph-bound STDP (#48): the adapter trains a loaded
 * Affine/Linear node's weights from pre/post spike trains, matching
 * standalone nir_stdp_step exactly and moving checkpoint bytes.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/checkpoint.h"
#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/stdp_graph.h"
#include "nir_c_runtime/train.h"
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

/* fc2 in lif_linear_graph.json is Affine with out=2, in=3. */
static void test_affine_roundtrip(void) {
    nir_graph_t graph = load_fixture("fixtures/lif_linear_graph.json");
    char *error = NULL;
    char *before = nir_checkpoint_dumps(&graph, NULL, &error);
    CHECK(error == NULL && before != NULL);

    const float *w = NULL;
    const float *bias = NULL;
    size_t wn = 0;
    size_t bn = 0;
    CHECK(nir_graph_get_weights(&graph, "fc2", &w, &wn, &bias, &bn));
    CHECK(wn == 6 && bn == 2);
    float initial[6];
    float initial_bias[2];
    memcpy(initial, w, sizeof(initial));
    memcpy(initial_bias, bias, sizeof(initial_bias));

    nir_stdp_graph_t bound;
    memset(&bound, 0, sizeof(bound));
    CHECK(nir_stdp_graph_bind(
        &bound, &graph, "fc2", 2.0f, 2.0f, 0.01f, 0.012f, 0.0f,
        1.0f, &error
    ));
    CHECK(error == NULL);
    float pre1[] = {1.0f, 0.0f, 0.0f};
    float post1[] = {0.0f, 0.0f};
    CHECK(nir_stdp_graph_step(&bound, pre1, post1));
    float pre2[] = {0.0f, 0.0f, 0.0f};
    float post2[] = {1.0f, 0.0f};
    CHECK(nir_stdp_graph_step(&bound, pre2, post2));

    /* Only w[out=0][in=0] potentiates: a_plus * exp(-1/tau). */
    CHECK(nir_graph_get_weights(&graph, "fc2", &w, &wn, &bias, &bn));
    CHECK_NEAR(w[0] - initial[0], 0.01 * exp(-0.5), 1e-7);
    for (size_t i = 1; i < 6; i++) CHECK(w[i] == initial[i]);
    CHECK(memcmp(bias, initial_bias, sizeof(initial_bias)) == 0);

    char *after = nir_checkpoint_dumps(&graph, NULL, &error);
    CHECK(error == NULL && after != NULL);
    CHECK(strcmp(before, after) != 0);

    nir_stdp_graph_free(&bound);
    free(before);
    free(after);
    nir_graph_free(&graph);
}

/* The adapter matches standalone nir_stdp_step bit-exactly on the
 * same spike trains (weights modulo the [out][in] transpose, plus
 * the eligibility traces). */
static void test_equivalence(void) {
    nir_graph_t graph = load_fixture("fixtures/lif_linear_graph.json");
    char *error = NULL;
    nir_stdp_graph_t bound;
    memset(&bound, 0, sizeof(bound));
    CHECK(nir_stdp_graph_bind(
        &bound, &graph, "fc2", 3.0f, 4.0f, 0.02f, 0.015f, -1.0f,
        1.0f, &error
    ));
    CHECK(error == NULL);

    nir_stdp_t solo;
    CHECK(nir_stdp_init(
        &solo, 3, 2, 3.0f, 4.0f, 0.02f, 0.015f, -1.0f, 1.0f,
        &error
    ));
    CHECK(error == NULL);
    CHECK(nir_stdp_set_weight(&solo, bound.stdp.weight));

    float pre[][3] = {
        {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f},
    };
    float post[][2] = {
        {0.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f},
        {1.0f, 1.0f}, {0.0f, 1.0f}, {1.0f, 0.0f},
    };
    for (size_t t = 0; t < 6; t++) {
        CHECK(nir_stdp_graph_step(&bound, pre[t], post[t]));
        CHECK(nir_stdp_step(&solo, pre[t], post[t]));
    }
    for (size_t i = 0; i < 6; i++) {
        CHECK(bound.stdp.weight[i] == solo.weight[i]);
    }
    for (size_t i = 0; i < 3; i++) {
        CHECK(bound.stdp.pre_trace[i] == solo.pre_trace[i]);
    }
    for (size_t o = 0; o < 2; o++) {
        CHECK(bound.stdp.post_trace[o] == solo.post_trace[o]);
    }
    const float *w = NULL;
    const float *bias = NULL;
    size_t wn = 0;
    size_t bn = 0;
    CHECK(nir_graph_get_weights(&graph, "fc2", &w, &wn, &bias, &bn));
    for (size_t i = 0; i < 3; i++) {
        for (size_t o = 0; o < 2; o++) {
            CHECK(w[o * 3 + i] == solo.weight[i * 2 + o]);
        }
    }

    nir_stdp_free(&solo);
    nir_stdp_graph_free(&bound);
    nir_graph_free(&graph);
}

/* fc1 is Linear (in=4, out=3, no bias): same wiring, no bias. */
static void test_linear(void) {
    nir_graph_t graph = load_fixture("fixtures/lif_linear_graph.json");
    char *error = NULL;
    const float *w = NULL;
    const float *bias = NULL;
    size_t wn = 0;
    size_t bn = 0;
    CHECK(nir_graph_get_weights(&graph, "fc1", &w, &wn, &bias, &bn));
    CHECK(wn == 12 && bias == NULL && bn == 0);
    float initial[12];
    memcpy(initial, w, sizeof(initial));

    nir_stdp_graph_t bound;
    memset(&bound, 0, sizeof(bound));
    CHECK(nir_stdp_graph_bind(
        &bound, &graph, "fc1", 2.0f, 2.0f, 0.05f, 0.05f, -2.0f,
        2.0f, &error
    ));
    CHECK(error == NULL);
    float pre1[] = {0.0f, 1.0f, 0.0f, 1.0f};
    float post1[] = {1.0f, 0.0f, 0.0f};
    CHECK(nir_stdp_graph_step(&bound, pre1, post1));
    float pre2[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float post2[] = {0.0f, 1.0f, 1.0f};
    CHECK(nir_stdp_graph_step(&bound, pre2, post2));

    CHECK(nir_graph_get_weights(&graph, "fc1", &w, &wn, &bias, &bn));
    CHECK(memcmp(w, initial, sizeof(initial)) != 0);
    for (size_t i = 0; i < 12; i++) CHECK(isfinite(w[i]));

    nir_stdp_graph_free(&bound);
    nir_graph_free(&graph);
}

static void test_rejects(void) {
    nir_graph_t graph = load_fixture("fixtures/lif_linear_graph.json");
    nir_graph_t conv = load_fixture("fixtures/conv_net_graph.json");
    char *error = NULL;
    nir_stdp_graph_t bound;
    memset(&bound, 0, sizeof(bound));

    CHECK(!nir_stdp_graph_bind(
        &bound, &graph, "nope", 2.0f, 2.0f, 0.01f, 0.012f, 0.0f,
        1.0f, &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_stdp_graph_bind(
        &bound, &graph, "lif1", 2.0f, 2.0f, 0.01f, 0.012f, 0.0f,
        1.0f, &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_stdp_graph_bind(
        &bound, &conv, "conv1", 2.0f, 2.0f, 0.01f, 0.012f, 0.0f,
        1.0f, &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_stdp_graph_bind(
        NULL, &graph, "fc2", 2.0f, 2.0f, 0.01f, 0.012f, 0.0f,
        1.0f, &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_stdp_graph_bind(
        &bound, NULL, "fc2", 2.0f, 2.0f, 0.01f, 0.012f, 0.0f,
        1.0f, NULL
    ));
    CHECK(!nir_stdp_graph_bind(
        &bound, &graph, NULL, 2.0f, 2.0f, 0.01f, 0.012f, 0.0f,
        1.0f, NULL
    ));
    CHECK(!nir_stdp_graph_bind(
        &bound, &graph, "fc2", 0.0f, 2.0f, 0.01f, 0.012f, 0.0f,
        1.0f, &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;

    /* A zeroed binding refuses to step; free is safe on it. */
    float pre[] = {1.0f, 0.0f, 0.0f};
    float post[] = {0.0f, 1.0f};
    CHECK(!nir_stdp_graph_step(&bound, pre, post));
    CHECK(!nir_stdp_graph_step(NULL, pre, post));
    nir_stdp_graph_free(&bound);
    nir_stdp_graph_free(NULL);

    nir_graph_free(&graph);
    nir_graph_free(&conv);
}

static void test_step_rejects_nonfinite(void) {
    nir_graph_t graph = load_fixture("fixtures/lif_linear_graph.json");
    char *error = NULL;
    nir_stdp_graph_t bound;
    memset(&bound, 0, sizeof(bound));
    CHECK(nir_stdp_graph_bind(
        &bound, &graph, "fc2", 2.0f, 2.0f, 0.01f, 0.012f, 0.0f,
        1.0f, &error
    ));
    CHECK(error == NULL);
    const float *w = NULL;
    const float *bias = NULL;
    size_t wn = 0;
    size_t bn = 0;
    CHECK(nir_graph_get_weights(&graph, "fc2", &w, &wn, &bias, &bn));
    float initial[6];
    memcpy(initial, w, sizeof(initial));

    float bad[] = {1.0f, (float)INFINITY, 0.0f};
    float ok[] = {0.0f, 1.0f};
    CHECK(!nir_stdp_graph_step(&bound, bad, ok));
    CHECK(!nir_stdp_graph_step(&bound, NULL, ok));
    CHECK(!nir_stdp_graph_step(&bound, ok, NULL));
    CHECK(nir_graph_get_weights(&graph, "fc2", &w, &wn, &bias, &bn));
    CHECK(memcmp(w, initial, sizeof(initial)) == 0);

    nir_stdp_graph_free(&bound);
    nir_graph_free(&graph);
}

int main(void) {
    test_affine_roundtrip();
    test_equivalence();
    test_linear();
    test_rejects();
    test_step_rejects_nonfinite();
    return 0;
}
