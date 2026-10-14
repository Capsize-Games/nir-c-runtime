/* Checkpoint round-trips, weight accessors, summary export (#29). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/checkpoint.h"
#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/runtime.h"
#include "test_util.h"

static void check_writer(void) {
    nir_json_writer_t *w = nir_json_writer_new();
    CHECK(w != NULL);
    CHECK(nir_json_begin_object(w));
    CHECK(nir_json_key(w, "name"));
    CHECK(nir_json_string(w, "a\"b\\c\nd"));
    CHECK(nir_json_key(w, "w"));
    CHECK(nir_json_begin_array(w));
    CHECK(nir_json_float(w, 0.5f));
    CHECK(nir_json_double(w, 0.1));
    CHECK(nir_json_int(w, -7));
    CHECK(nir_json_bool(w, 1));
    CHECK(nir_json_null(w));
    CHECK(nir_json_end(w));
    CHECK(nir_json_end(w));
    const char *text = nir_json_writer_data(w);
    CHECK(text != NULL);
    CHECK(strcmp(text,
        "{\"name\":\"a\\\"b\\\\c\\nd\",\"w\":[0.5,0.10000000000000001,"
        "-7,true,null]}") == 0);
    CHECK(nir_json_writer_length(w) == strlen(text));
    CHECK(!nir_json_end(w)); /* unbalanced close fails */
    CHECK(nir_json_writer_data(w) == NULL);
    nir_json_writer_free(w);

    nir_json_writer_t *bad = nir_json_writer_new();
    CHECK(!nir_json_key(bad, "k")); /* key outside an object fails */
    CHECK(!nir_json_end(bad)); /* end with nothing open fails */
    nir_json_writer_free(bad);
}

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

static void check_weights_equal(
    const nir_graph_t *a, const nir_graph_t *b
) {
    CHECK(a->node_count == b->node_count);
    for (size_t i = 0; i < a->node_count; i++) {
        const float *wa = NULL;
        const float *wb = NULL;
        const float *ba = NULL;
        const float *bb = NULL;
        size_t wna = 0;
        size_t wnb = 0;
        size_t bna = 0;
        size_t bnb = 0;
        int ha = nir_graph_get_weights(
            a, a->nodes[i].name, &wa, &wna, &ba, &bna
        );
        int hb = nir_graph_get_weights(
            b, b->nodes[i].name, &wb, &wnb, &bb, &bnb
        );
        CHECK(ha == hb);
        if (!ha) {
            continue;
        }
        CHECK(wna == wnb && bna == bnb);
        CHECK(memcmp(wa, wb, wna * sizeof(float)) == 0);
        if (bna > 0) {
            CHECK(memcmp(ba, bb, bna * sizeof(float)) == 0);
        }
    }
}

static nir_tensor_t run_readout(nir_graph_t *graph) {
    nir_runtime_t runtime;
    char *error = NULL;
    CHECK(nir_runtime_init(&runtime, graph, &error));
    size_t shape[1] = {4};
    nir_tensor_t frames[6];
    for (int t = 0; t < 6; t++) {
        frames[t] = nir_tensor_alloc(shape, 1);
        for (size_t i = 0; i < frames[t].size; i++) {
            frames[t].data[i] = (float)(t + 1) * 0.1f;
        }
    }
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(nir_runtime_run(&runtime, frames, 6, &readout, &error));
    CHECK(error == NULL);
    for (int t = 0; t < 6; t++) {
        nir_tensor_free(&frames[t]);
    }
    nir_runtime_free(&runtime);
    return readout;
}

int main(void) {
    check_writer();

    char *error = NULL;
    nir_graph_t graph = load_fixture("fixtures/lif_linear_graph.json");

    /* Weight accessors: get, set, and every mismatch shape. */
    const float *w = NULL;
    const float *b = NULL;
    size_t wn = 0;
    size_t bn = 0;
    CHECK(nir_graph_get_weights(&graph, "fc1", &w, &wn, &b, &bn));
    CHECK(wn > 0 && b == NULL && bn == 0); /* Linear has no bias */
    CHECK(nir_graph_get_weights(&graph, "fc2", &w, &wn, &b, &bn));
    CHECK(wn > 0 && b != NULL && bn > 0); /* Affine has a bias */
    CHECK(!nir_graph_get_weights(&graph, "lif1", &w, &wn, &b, &bn));
    CHECK(!nir_graph_get_weights(&graph, "nope", &w, &wn, &b, &bn));

    float *fc1_before = (float *)malloc(wn * sizeof(float));
    CHECK(fc1_before != NULL);
    CHECK(nir_graph_get_weights(&graph, "fc2", &w, &wn, &b, &bn));
    memcpy(fc1_before, w, wn * sizeof(float));
    float *zeros = (float *)calloc(wn > bn ? wn : bn, sizeof(float));
    CHECK(zeros != NULL);
    CHECK(nir_graph_set_weights(&graph, "fc2", zeros, wn, zeros, bn,
        &error));
    CHECK(error == NULL);
    CHECK(!nir_graph_set_weights(&graph, "fc2", zeros, wn + 1, NULL, 0,
        &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_graph_set_weights(&graph, "fc1", zeros, 1, zeros, 1,
        &error)); /* Linear rejects bias data */
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_graph_set_weights(&graph, "lif1", zeros, 1, NULL, 0,
        &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_graph_set_weights(&graph, "nope", zeros, 1, NULL, 0,
        &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;

    /* Save with metadata, clobber, reload: bit-exact resume. */
    nir_checkpoint_meta_t meta;
    memset(&meta, 0, sizeof(meta));
    meta.topology = "lif_linear";
    meta.seed = 42;
    meta.step = 100;
    /* Restore fc2 first so the checkpoint holds trained weights. */
    CHECK(nir_graph_set_weights(&graph, "fc2", fc1_before, wn, b, bn,
        &error));
    char *dump = nir_checkpoint_dumps(&graph, &meta, &error);
    CHECK(error == NULL);
    CHECK(dump != NULL);
    CHECK(strstr(dump, "\"format\":\"spikeforge-c-checkpoint\"") != NULL);
    CHECK(strstr(dump, "\"version\":1") != NULL);
    CHECK(strstr(dump, "\"topology\":\"lif_linear\"") != NULL);
    CHECK(nir_graph_set_weights(&graph, "fc2", zeros, wn, zeros, bn,
        &error));
    nir_checkpoint_meta_t back;
    memset(&back, 0, sizeof(back));
    CHECK(nir_checkpoint_loads(dump, &graph, &back, &error));
    CHECK(error == NULL);
    CHECK(back.topology != NULL && strcmp(back.topology, "lif_linear") == 0);
    CHECK(back.seed == 42 && back.step == 100);
    nir_checkpoint_meta_free(&back);
    CHECK(nir_graph_get_weights(&graph, "fc2", &w, &wn, &b, &bn));
    CHECK(memcmp(w, fc1_before, wn * sizeof(float)) == 0);

    /* File round-trip through TMPDIR (never the source tree). */
    const char *tmp = getenv("TMPDIR");
    char path[512];
    sprintf(path, "%s/nir-checkpoint-test.json",
        tmp != NULL ? tmp : "/tmp");
    CHECK(nir_checkpoint_save(path, &graph, &meta, &error));
    CHECK(error == NULL);
    CHECK(nir_graph_set_weights(&graph, "fc2", zeros, wn, zeros, bn,
        &error));
    memset(&back, 0, sizeof(back));
    CHECK(nir_checkpoint_load(path, &graph, &back, &error));
    CHECK(error == NULL);
    CHECK(back.step == 100);
    nir_checkpoint_meta_free(&back);
    CHECK(nir_graph_get_weights(&graph, "fc2", &w, &wn, &b, &bn));
    CHECK(memcmp(w, fc1_before, wn * sizeof(float)) == 0);
    free(fc1_before);
    free(zeros);

    /* Rejects: bad version, bad format, missing node, bad shape. */
    char hacked[65536];
    strcpy(hacked, dump);
    char *ver = strstr(hacked, "\"version\":1");
    CHECK(ver != NULL);
    ver[strlen("\"version\":")] = '9';
    CHECK(!nir_checkpoint_loads(hacked, &graph, NULL, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_checkpoint_loads("{\"format\":\"nope\",\"version\":1,"
        "\"nodes\":[]}", &graph, NULL, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_checkpoint_loads("{\"format\":"
        "\"spikeforge-c-checkpoint\",\"version\":1,\"nodes\":[]}",
        &graph, NULL, &error)); /* drops every weighted node */
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_checkpoint_loads("not json", &graph, NULL, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;

    /* Non-finite weights refuse to serialize. */
    CHECK(nir_graph_get_weights(&graph, "fc2", &w, &wn, &b, &bn));
    float *mutable = (float *)w;
    float saved = mutable[0];
    mutable[0] = INFINITY;
    CHECK(nir_checkpoint_dumps(&graph, NULL, &error) == NULL);
    CHECK(error != NULL);
    free(error);
    error = NULL;
    mutable[0] = saved;

    /* Reverse bridge: export, reload, identical weights + readout. */
    nir_tensor_t before = run_readout(&graph);
    char *summary = nir_graph_export_summary(&graph, &error);
    CHECK(error == NULL);
    CHECK(summary != NULL);
    json_value_t *summary_json = json_parse(summary, &error);
    CHECK(error == NULL);
    nir_graph_t again;
    memset(&again, 0, sizeof(again));
    CHECK(nir_graph_load(summary_json, &again, &error));
    CHECK(error == NULL);
    check_weights_equal(&graph, &again);
    nir_tensor_t after = run_readout(&again);
    CHECK(before.size == after.size);
    for (size_t i = 0; i < before.size; i++) {
        CHECK(before.data[i] == after.data[i]);
    }
    nir_tensor_free(&before);
    nir_tensor_free(&after);
    nir_graph_free(&again);
    json_free(summary_json);
    free(summary);
    free(dump);

    nir_graph_free(&graph);
    return 0;
}
