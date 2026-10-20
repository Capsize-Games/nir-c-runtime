/* Checkpoint round-trips, weight accessors, summary export (#29),
 * plus optimizer-state resume (#43). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/checkpoint.h"
#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/runtime.h"
#include "nir_c_runtime/train.h"
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

static void fill_grad(float *g, size_t n, float base) {
    for (size_t i = 0; i < n; i++) {
        g[i] = base + 0.001f * (float)i;
    }
}

static float *copy_buf(const float *src, size_t n) {
    float *dst = (float *)malloc(n * sizeof(float));
    CHECK(dst != NULL);
    memcpy(dst, src, n * sizeof(float));
    return dst;
}

static char *copy_text(const char *src) {
    size_t n = strlen(src) + 1;
    char *dst = (char *)malloc(n);
    CHECK(dst != NULL);
    memcpy(dst, src, n);
    return dst;
}

/* SGD resume: K steps, save, N more steps vs fresh+load+N steps
 * are bit-identical in weights and velocities. */
static void check_opt_sgd_resume(void) {
    char *error = NULL;
    nir_graph_t a = load_fixture("fixtures/lif_linear_graph.json");
    const float *w1 = NULL;
    const float *b1 = NULL;
    const float *w2 = NULL;
    const float *b2 = NULL;
    size_t w1n = 0;
    size_t b1n = 0;
    size_t w2n = 0;
    size_t b2n = 0;
    CHECK(nir_graph_get_weights(&a, "fc1", &w1, &w1n, &b1, &b1n));
    CHECK(nir_graph_get_weights(&a, "fc2", &w2, &w2n, &b2, &b2n));
    CHECK(b1 == NULL && b1n == 0);
    nir_sgd_t s1w;
    nir_sgd_t s2w;
    nir_sgd_t s2b;
    CHECK(nir_sgd_init(&s1w, w1n, 0.1f, 0.9f, 0.01f, &error));
    CHECK(nir_sgd_init(&s2w, w2n, 0.1f, 0.9f, 0.01f, &error));
    CHECK(nir_sgd_init(&s2b, b2n, 0.1f, 0.9f, 0.01f, &error));
    float *g1w = copy_buf(w1, w1n);
    float *g2w = copy_buf(w2, w2n);
    float *g2b = copy_buf(b2, b2n);
    fill_grad(g1w, w1n, 0.1f);
    fill_grad(g2w, w2n, 0.1f);
    fill_grad(g2b, b2n, 0.2f);
    for (int t = 0; t < 3; t++) {
        CHECK(nir_sgd_step(&s1w, (float *)w1, g1w, w1n));
        CHECK(nir_sgd_step(&s2w, (float *)w2, g2w, w2n));
        CHECK(nir_sgd_step(&s2b, (float *)b2, g2b, b2n));
    }
    nir_checkpoint_opt_t opts[2];
    memset(opts, 0, sizeof(opts));
    opts[0].node = "fc1";
    opts[0].kind = NIR_CHECKPOINT_OPT_SGD;
    opts[0].sgd_weight = &s1w;
    opts[1].node = "fc2";
    opts[1].kind = NIR_CHECKPOINT_OPT_SGD;
    opts[1].sgd_weight = &s2w;
    opts[1].sgd_bias = &s2b;
    char *dump = nir_checkpoint_dumps_with_opt(
        &a, NULL, opts, 2, &error
    );
    CHECK(error == NULL && dump != NULL);
    CHECK(strstr(dump, "\"optimizers\"") != NULL);
    CHECK(strstr(dump, "\"velocity\"") != NULL);
    CHECK(strstr(dump, "\"velocity_bias\"") != NULL);
    for (int t = 0; t < 3; t++) {
        CHECK(nir_sgd_step(&s1w, (float *)w1, g1w, w1n));
        CHECK(nir_sgd_step(&s2w, (float *)w2, g2w, w2n));
        CHECK(nir_sgd_step(&s2b, (float *)b2, g2b, b2n));
    }
    float *a_w1 = copy_buf(w1, w1n);
    float *a_w2 = copy_buf(w2, w2n);
    float *a_b2 = copy_buf(b2, b2n);
    float *a_v1 = copy_buf(s1w.velocity, w1n);
    float *a_v2 = copy_buf(s2w.velocity, w2n);
    float *a_vb = copy_buf(s2b.velocity, b2n);

    nir_graph_t b = load_fixture("fixtures/lif_linear_graph.json");
    nir_sgd_t r1w;
    nir_sgd_t r2w;
    nir_sgd_t r2b;
    CHECK(nir_sgd_init(&r1w, w1n, 0.1f, 0.9f, 0.01f, &error));
    CHECK(nir_sgd_init(&r2w, w2n, 0.1f, 0.9f, 0.01f, &error));
    CHECK(nir_sgd_init(&r2b, b2n, 0.1f, 0.9f, 0.01f, &error));
    nir_checkpoint_opt_t ropts[2];
    memset(ropts, 0, sizeof(ropts));
    ropts[0].node = "fc1";
    ropts[0].kind = NIR_CHECKPOINT_OPT_SGD;
    ropts[0].sgd_weight = &r1w;
    ropts[1].node = "fc2";
    ropts[1].kind = NIR_CHECKPOINT_OPT_SGD;
    ropts[1].sgd_weight = &r2w;
    ropts[1].sgd_bias = &r2b;
    CHECK(nir_checkpoint_loads_with_opt(
        dump, &b, NULL, ropts, 2, &error
    ));
    CHECK(error == NULL);
    const float *bw1 = NULL;
    const float *bb1 = NULL;
    const float *bw2 = NULL;
    const float *bb2 = NULL;
    size_t bw1n = 0;
    size_t bb1n = 0;
    size_t bw2n = 0;
    size_t bb2n = 0;
    CHECK(nir_graph_get_weights(&b, "fc1", &bw1, &bw1n, &bb1, &bb1n));
    CHECK(nir_graph_get_weights(&b, "fc2", &bw2, &bw2n, &bb2, &bb2n));
    for (int t = 0; t < 3; t++) {
        CHECK(nir_sgd_step(&r1w, (float *)bw1, g1w, w1n));
        CHECK(nir_sgd_step(&r2w, (float *)bw2, g2w, w2n));
        CHECK(nir_sgd_step(&r2b, (float *)bb2, g2b, b2n));
    }
    CHECK(memcmp(bw1, a_w1, w1n * sizeof(float)) == 0);
    CHECK(memcmp(bw2, a_w2, w2n * sizeof(float)) == 0);
    CHECK(memcmp(bb2, a_b2, b2n * sizeof(float)) == 0);
    CHECK(memcmp(r1w.velocity, a_v1, w1n * sizeof(float)) == 0);
    CHECK(memcmp(r2w.velocity, a_v2, w2n * sizeof(float)) == 0);
    CHECK(memcmp(r2b.velocity, a_vb, b2n * sizeof(float)) == 0);

    /* File round-trip carries the same optimizer state. */
    const char *tmp = getenv("TMPDIR");
    char path[512];
    sprintf(path, "%s/nir-opt-test.json", tmp != NULL ? tmp : "/tmp");
    nir_graph_t c = load_fixture("fixtures/lif_linear_graph.json");
    nir_sgd_t f1w;
    nir_sgd_t f2w;
    nir_sgd_t f2b;
    CHECK(nir_sgd_init(&f1w, w1n, 0.1f, 0.9f, 0.01f, &error));
    CHECK(nir_sgd_init(&f2w, w2n, 0.1f, 0.9f, 0.01f, &error));
    CHECK(nir_sgd_init(&f2b, b2n, 0.1f, 0.9f, 0.01f, &error));
    nir_checkpoint_opt_t fopts[2];
    memset(fopts, 0, sizeof(fopts));
    fopts[0].node = "fc1";
    fopts[0].kind = NIR_CHECKPOINT_OPT_SGD;
    fopts[0].sgd_weight = &f1w;
    fopts[1].node = "fc2";
    fopts[1].kind = NIR_CHECKPOINT_OPT_SGD;
    fopts[1].sgd_weight = &f2w;
    fopts[1].sgd_bias = &f2b;
    CHECK(nir_checkpoint_save_with_opt(
        path, &a, NULL, opts, 2, &error
    ));
    CHECK(error == NULL);
    CHECK(nir_checkpoint_load_with_opt(
        path, &c, NULL, fopts, 2, &error
    ));
    CHECK(error == NULL);
    const float *cw1 = NULL;
    const float *cb1 = NULL;
    const float *cw2 = NULL;
    const float *cb2 = NULL;
    size_t cw1n = 0;
    size_t cb1n = 0;
    size_t cw2n = 0;
    size_t cb2n = 0;
    CHECK(nir_graph_get_weights(&c, "fc1", &cw1, &cw1n, &cb1, &cb1n));
    CHECK(nir_graph_get_weights(&c, "fc2", &cw2, &cw2n, &cb2, &cb2n));
    CHECK(memcmp(cw1, a_w1, w1n * sizeof(float)) == 0);
    CHECK(memcmp(f1w.velocity, a_v1, w1n * sizeof(float)) == 0);

    free(g1w);
    free(g2w);
    free(g2b);
    free(a_w1);
    free(a_w2);
    free(a_b2);
    free(a_v1);
    free(a_v2);
    free(a_vb);
    free(dump);
    nir_sgd_free(&s1w);
    nir_sgd_free(&s2w);
    nir_sgd_free(&s2b);
    nir_sgd_free(&r1w);
    nir_sgd_free(&r2w);
    nir_sgd_free(&r2b);
    nir_sgd_free(&f1w);
    nir_sgd_free(&f2w);
    nir_sgd_free(&f2b);
    nir_graph_free(&a);
    nir_graph_free(&b);
    nir_graph_free(&c);
}

/* Adam resume: weights, m/v, and step counters are bit-identical.
 * Bias correction makes the step counter load-bearing: a dropped
 * step would diverge on the very next update. */
static void check_opt_adam_resume(void) {
    char *error = NULL;
    nir_graph_t a = load_fixture("fixtures/lif_linear_graph.json");
    const float *w1 = NULL;
    const float *b1 = NULL;
    const float *w2 = NULL;
    const float *b2 = NULL;
    size_t w1n = 0;
    size_t b1n = 0;
    size_t w2n = 0;
    size_t b2n = 0;
    CHECK(nir_graph_get_weights(&a, "fc1", &w1, &w1n, &b1, &b1n));
    CHECK(nir_graph_get_weights(&a, "fc2", &w2, &w2n, &b2, &b2n));
    nir_adam_t s1w;
    nir_adam_t s2w;
    nir_adam_t s2b;
    CHECK(nir_adam_init(
        &s1w, w1n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    CHECK(nir_adam_init(
        &s2w, w2n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    CHECK(nir_adam_init(
        &s2b, b2n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    float *g1w = copy_buf(w1, w1n);
    float *g2w = copy_buf(w2, w2n);
    float *g2b = copy_buf(b2, b2n);
    fill_grad(g1w, w1n, 0.1f);
    fill_grad(g2w, w2n, 0.1f);
    fill_grad(g2b, b2n, 0.2f);
    for (int t = 0; t < 4; t++) {
        CHECK(nir_adam_step(&s1w, (float *)w1, g1w, w1n));
        CHECK(nir_adam_step(&s2w, (float *)w2, g2w, w2n));
        CHECK(nir_adam_step(&s2b, (float *)b2, g2b, b2n));
    }
    nir_checkpoint_opt_t opts[2];
    memset(opts, 0, sizeof(opts));
    opts[0].node = "fc1";
    opts[0].kind = NIR_CHECKPOINT_OPT_ADAM;
    opts[0].adam_weight = &s1w;
    opts[1].node = "fc2";
    opts[1].kind = NIR_CHECKPOINT_OPT_ADAM;
    opts[1].adam_weight = &s2w;
    opts[1].adam_bias = &s2b;
    char *dump = nir_checkpoint_dumps_with_opt(
        &a, NULL, opts, 2, &error
    );
    CHECK(error == NULL && dump != NULL);
    CHECK(strstr(dump, "\"kind\":\"adam\"") != NULL);
    CHECK(strstr(dump, "\"step\":4") != NULL);
    for (int t = 0; t < 4; t++) {
        CHECK(nir_adam_step(&s1w, (float *)w1, g1w, w1n));
        CHECK(nir_adam_step(&s2w, (float *)w2, g2w, w2n));
        CHECK(nir_adam_step(&s2b, (float *)b2, g2b, b2n));
    }
    CHECK(s1w.step == 8 && s2w.step == 8 && s2b.step == 8);
    float *a_w1 = copy_buf(w1, w1n);
    float *a_w2 = copy_buf(w2, w2n);
    float *a_b2 = copy_buf(b2, b2n);
    float *a_m1 = copy_buf(s1w.m, w1n);
    float *a_v1 = copy_buf(s1w.v, w1n);

    nir_graph_t b = load_fixture("fixtures/lif_linear_graph.json");
    nir_adam_t r1w;
    nir_adam_t r2w;
    nir_adam_t r2b;
    CHECK(nir_adam_init(
        &r1w, w1n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    CHECK(nir_adam_init(
        &r2w, w2n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    CHECK(nir_adam_init(
        &r2b, b2n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    nir_checkpoint_opt_t ropts[2];
    memset(ropts, 0, sizeof(ropts));
    ropts[0].node = "fc1";
    ropts[0].kind = NIR_CHECKPOINT_OPT_ADAM;
    ropts[0].adam_weight = &r1w;
    ropts[1].node = "fc2";
    ropts[1].kind = NIR_CHECKPOINT_OPT_ADAM;
    ropts[1].adam_weight = &r2w;
    ropts[1].adam_bias = &r2b;
    CHECK(nir_checkpoint_loads_with_opt(
        dump, &b, NULL, ropts, 2, &error
    ));
    CHECK(error == NULL);
    CHECK(r1w.step == 4 && r2w.step == 4 && r2b.step == 4);
    const float *bw1 = NULL;
    const float *bb1 = NULL;
    const float *bw2 = NULL;
    const float *bb2 = NULL;
    size_t bw1n = 0;
    size_t bb1n = 0;
    size_t bw2n = 0;
    size_t bb2n = 0;
    CHECK(nir_graph_get_weights(&b, "fc1", &bw1, &bw1n, &bb1, &bb1n));
    CHECK(nir_graph_get_weights(&b, "fc2", &bw2, &bw2n, &bb2, &bb2n));
    for (int t = 0; t < 4; t++) {
        CHECK(nir_adam_step(&r1w, (float *)bw1, g1w, w1n));
        CHECK(nir_adam_step(&r2w, (float *)bw2, g2w, w2n));
        CHECK(nir_adam_step(&r2b, (float *)bb2, g2b, b2n));
    }
    CHECK(r1w.step == 8 && r2w.step == 8 && r2b.step == 8);
    CHECK(memcmp(bw1, a_w1, w1n * sizeof(float)) == 0);
    CHECK(memcmp(bw2, a_w2, w2n * sizeof(float)) == 0);
    CHECK(memcmp(bb2, a_b2, b2n * sizeof(float)) == 0);
    CHECK(memcmp(r1w.m, a_m1, w1n * sizeof(float)) == 0);
    CHECK(memcmp(r1w.v, a_v1, w1n * sizeof(float)) == 0);

    free(g1w);
    free(g2w);
    free(g2b);
    free(a_w1);
    free(a_w2);
    free(a_b2);
    free(a_m1);
    free(a_v1);
    free(dump);
    nir_adam_free(&s1w);
    nir_adam_free(&s2w);
    nir_adam_free(&s2b);
    nir_adam_free(&r1w);
    nir_adam_free(&r2w);
    nir_adam_free(&r2b);
    nir_graph_free(&a);
    nir_graph_free(&b);
}

/* A v1 checkpoint (no optimizers section) still loads; caller
 * optimizers keep their fresh state. */
static void check_opt_v1_compat(void) {
    char *error = NULL;
    nir_graph_t a = load_fixture("fixtures/lif_linear_graph.json");
    char *dump = nir_checkpoint_dumps(&a, NULL, &error);
    CHECK(error == NULL && dump != NULL);
    char *ver = strstr(dump, "\"version\":2");
    CHECK(ver != NULL);
    ver[strlen("\"version\":")] = '1';
    CHECK(strstr(dump, "\"optimizers\"") == NULL);

    nir_graph_t b = load_fixture("fixtures/lif_linear_graph.json");
    const float *w1 = NULL;
    const float *b1 = NULL;
    size_t w1n = 0;
    size_t b1n = 0;
    CHECK(nir_graph_get_weights(&b, "fc1", &w1, &w1n, &b1, &b1n));
    nir_adam_t opt;
    CHECK(nir_adam_init(
        &opt, w1n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    nir_checkpoint_opt_t opts[1];
    memset(opts, 0, sizeof(opts));
    opts[0].node = "fc1";
    opts[0].kind = NIR_CHECKPOINT_OPT_ADAM;
    opts[0].adam_weight = &opt;
    CHECK(nir_checkpoint_loads_with_opt(
        dump, &b, NULL, opts, 1, &error
    ));
    CHECK(error == NULL);
    CHECK(opt.step == 0);
    for (size_t i = 0; i < w1n; i++) {
        CHECK(opt.m[i] == 0.0f && opt.v[i] == 0.0f);
    }
    /* Plain loads accepts v1 too. */
    nir_graph_t c = load_fixture("fixtures/lif_linear_graph.json");
    CHECK(nir_checkpoint_loads(dump, &c, NULL, &error));
    CHECK(error == NULL);

    free(dump);
    nir_adam_free(&opt);
    nir_graph_free(&a);
    nir_graph_free(&b);
    nir_graph_free(&c);
}

/* Plain loads restores weights from a v2-with-optimizers file,
 * ignoring the optimizer section. */
static void check_opt_plain_ignores(void) {
    char *error = NULL;
    nir_graph_t a = load_fixture("fixtures/lif_linear_graph.json");
    const float *w1 = NULL;
    const float *b1 = NULL;
    size_t w1n = 0;
    size_t b1n = 0;
    CHECK(nir_graph_get_weights(&a, "fc1", &w1, &w1n, &b1, &b1n));
    nir_sgd_t opt;
    CHECK(nir_sgd_init(&opt, w1n, 0.1f, 0.9f, 0.0f, &error));
    float *grad = copy_buf(w1, w1n);
    fill_grad(grad, w1n, 0.1f);
    CHECK(nir_sgd_step(&opt, (float *)w1, grad, w1n));
    nir_checkpoint_opt_t opts[1];
    memset(opts, 0, sizeof(opts));
    opts[0].node = "fc1";
    opts[0].kind = NIR_CHECKPOINT_OPT_SGD;
    opts[0].sgd_weight = &opt;
    char *dump = nir_checkpoint_dumps_with_opt(
        &a, NULL, opts, 1, &error
    );
    CHECK(error == NULL && dump != NULL);
    CHECK(strstr(dump, "\"optimizers\"") != NULL);

    nir_graph_t b = load_fixture("fixtures/lif_linear_graph.json");
    CHECK(nir_checkpoint_loads(dump, &b, NULL, &error));
    CHECK(error == NULL);
    const float *bw1 = NULL;
    const float *bb1 = NULL;
    size_t bw1n = 0;
    size_t bb1n = 0;
    CHECK(nir_graph_get_weights(&b, "fc1", &bw1, &bw1n, &bb1, &bb1n));
    CHECK(memcmp(bw1, w1, w1n * sizeof(float)) == 0);

    free(grad);
    free(dump);
    nir_sgd_free(&opt);
    nir_graph_free(&a);
    nir_graph_free(&b);
}

static void check_opt_rejects(void) {
    char *error = NULL;
    nir_graph_t g = load_fixture("fixtures/lif_linear_graph.json");
    const float *w1 = NULL;
    const float *b1 = NULL;
    const float *w2 = NULL;
    const float *b2 = NULL;
    size_t w1n = 0;
    size_t b1n = 0;
    size_t w2n = 0;
    size_t b2n = 0;
    CHECK(nir_graph_get_weights(&g, "fc1", &w1, &w1n, &b1, &b1n));
    CHECK(nir_graph_get_weights(&g, "fc2", &w2, &w2n, &b2, &b2n));
    nir_sgd_t sw;
    nir_adam_t aw;
    nir_adam_t aw2;
    nir_adam_t ab2;
    CHECK(nir_sgd_init(&sw, w1n, 0.1f, 0.9f, 0.0f, &error));
    CHECK(nir_adam_init(
        &aw, w1n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    CHECK(nir_adam_init(
        &aw2, w2n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    CHECK(nir_adam_init(
        &ab2, b2n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));

    /* Save rejects: unknown node, unweighted node, duplicates,
     * mixed kinds, and count mismatches. */
    nir_checkpoint_opt_t bad;
    memset(&bad, 0, sizeof(bad));
    bad.node = "nope";
    bad.kind = NIR_CHECKPOINT_OPT_SGD;
    bad.sgd_weight = &sw;
    CHECK(nir_checkpoint_dumps_with_opt(
        &g, NULL, &bad, 1, &error
    ) == NULL);
    CHECK(error != NULL);
    free(error);
    error = NULL;
    bad.node = "lif1";
    CHECK(nir_checkpoint_dumps_with_opt(
        &g, NULL, &bad, 1, &error
    ) == NULL);
    CHECK(error != NULL);
    free(error);
    error = NULL;
    nir_checkpoint_opt_t dup[2];
    memset(dup, 0, sizeof(dup));
    dup[0].node = "fc1";
    dup[0].kind = NIR_CHECKPOINT_OPT_SGD;
    dup[0].sgd_weight = &sw;
    dup[1].node = "fc1";
    dup[1].kind = NIR_CHECKPOINT_OPT_SGD;
    dup[1].sgd_weight = &sw;
    CHECK(nir_checkpoint_dumps_with_opt(
        &g, NULL, dup, 2, &error
    ) == NULL);
    CHECK(error != NULL);
    free(error);
    error = NULL;
    nir_checkpoint_opt_t mixed;
    memset(&mixed, 0, sizeof(mixed));
    mixed.node = "fc1";
    mixed.kind = NIR_CHECKPOINT_OPT_SGD;
    mixed.sgd_weight = &sw;
    mixed.adam_weight = &aw;
    CHECK(nir_checkpoint_dumps_with_opt(
        &g, NULL, &mixed, 1, &error
    ) == NULL);
    CHECK(error != NULL);
    free(error);
    error = NULL;
    nir_sgd_t wrong;
    CHECK(nir_sgd_init(&wrong, w1n + 1, 0.1f, 0.0f, 0.0f, &error));
    nir_checkpoint_opt_t mismatch;
    memset(&mismatch, 0, sizeof(mismatch));
    mismatch.node = "fc1";
    mismatch.kind = NIR_CHECKPOINT_OPT_SGD;
    mismatch.sgd_weight = &wrong;
    CHECK(nir_checkpoint_dumps_with_opt(
        &g, NULL, &mismatch, 1, &error
    ) == NULL);
    CHECK(error != NULL);
    free(error);
    error = NULL;
    nir_sgd_free(&wrong);

    /* A valid Adam dump for load-reject hacks. */
    nir_checkpoint_opt_t good[2];
    memset(good, 0, sizeof(good));
    good[0].node = "fc1";
    good[0].kind = NIR_CHECKPOINT_OPT_ADAM;
    good[0].adam_weight = &aw;
    good[1].node = "fc2";
    good[1].kind = NIR_CHECKPOINT_OPT_ADAM;
    good[1].adam_weight = &aw2;
    good[1].adam_bias = &ab2;
    float *gw1 = copy_buf(w1, w1n);
    fill_grad(gw1, w1n, 0.1f);
    CHECK(nir_adam_step(&aw, (float *)w1, gw1, w1n));
    free(gw1);
    char *dump = nir_checkpoint_dumps_with_opt(
        &g, NULL, good, 2, &error
    );
    CHECK(error == NULL && dump != NULL);

    /* Load rejects a kind mismatch (file adam vs caller sgd). */
    nir_graph_t h = load_fixture("fixtures/lif_linear_graph.json");
    nir_sgd_t rs;
    CHECK(nir_sgd_init(&rs, w1n, 0.1f, 0.9f, 0.0f, &error));
    nir_checkpoint_opt_t as_sgd[1];
    memset(as_sgd, 0, sizeof(as_sgd));
    as_sgd[0].node = "fc1";
    as_sgd[0].kind = NIR_CHECKPOINT_OPT_SGD;
    as_sgd[0].sgd_weight = &rs;
    /* Caller omits fc2 as well: file fc1 hits the kind check
     * first only if iteration meets fc1 first; to force the kind
     * path, pass both nodes with fc1 mistyped. */
    nir_checkpoint_opt_t kind_mix[2];
    memset(kind_mix, 0, sizeof(kind_mix));
    kind_mix[0].node = "fc1";
    kind_mix[0].kind = NIR_CHECKPOINT_OPT_SGD;
    kind_mix[0].sgd_weight = &rs;
    kind_mix[1].node = "fc2";
    kind_mix[1].kind = NIR_CHECKPOINT_OPT_ADAM;
    kind_mix[1].adam_weight = &aw2;
    kind_mix[1].adam_bias = &ab2;
    CHECK(!nir_checkpoint_loads_with_opt(
        dump, &h, NULL, kind_mix, 2, &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;

    /* Load rejects file entries with no caller entry. */
    CHECK(!nir_checkpoint_loads_with_opt(
        dump, &h, NULL, as_sgd, 1, &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    nir_sgd_free(&rs);

    /* Load rejects an unknown optimizer node. */
    char *hacked = copy_text(dump);
    char *fc2key = strstr(hacked, "\"fc2\":{\"kind\"");
    CHECK(fc2key != NULL);
    memcpy(fc2key + 1, "nope", 4);
    nir_graph_t u = load_fixture("fixtures/lif_linear_graph.json");
    nir_adam_t u1;
    nir_adam_t u2;
    nir_adam_t ub2;
    CHECK(nir_adam_init(
        &u1, w1n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    CHECK(nir_adam_init(
        &u2, w2n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    CHECK(nir_adam_init(
        &ub2, b2n, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    nir_checkpoint_opt_t uopts[2];
    memset(uopts, 0, sizeof(uopts));
    uopts[0].node = "fc1";
    uopts[0].kind = NIR_CHECKPOINT_OPT_ADAM;
    uopts[0].adam_weight = &u1;
    uopts[1].node = "fc2";
    uopts[1].kind = NIR_CHECKPOINT_OPT_ADAM;
    uopts[1].adam_weight = &u2;
    uopts[1].adam_bias = &ub2;
    CHECK(!nir_checkpoint_loads_with_opt(
        hacked, &u, NULL, uopts, 2, &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    free(hacked);

    /* Load rejects a truncated optimizer buffer. */
    hacked = copy_text(dump);
    char *vel = strstr(hacked, "\"m\":[");
    CHECK(vel != NULL);
    char *end = strchr(vel, ']');
    CHECK(end != NULL);
    memmove(vel + 5, end, strlen(end) + 1);
    memcpy(vel + 5, "0.0]", 4);
    CHECK(!nir_checkpoint_loads_with_opt(
        hacked, &u, NULL, uopts, 2, &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    free(hacked);

    /* Load rejects an invalid Adam step. */
    hacked = copy_text(dump);
    char *step = strstr(hacked, "\"step\":1");
    CHECK(step != NULL);
    step[strlen("\"step\":")] = '-';
    CHECK(!nir_checkpoint_loads_with_opt(
        hacked, &u, NULL, uopts, 2, &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    free(hacked);

    free(dump);
    nir_sgd_free(&sw);
    nir_adam_free(&aw);
    nir_adam_free(&aw2);
    nir_adam_free(&ab2);
    nir_adam_free(&u1);
    nir_adam_free(&u2);
    nir_adam_free(&ub2);
    nir_graph_free(&g);
    nir_graph_free(&h);
    nir_graph_free(&u);
}

int main(void) {
    check_writer();

    char *error = NULL;
    nir_graph_t graph = load_fixture("fixtures/lif_linear_graph.json");
    size_t lif1 = nir_graph_find(&graph, "lif1");
    CHECK(lif1 != SIZE_MAX && graph.nodes[lif1].kind == NIR_NODE_LIF);
    graph.nodes[lif1].params.lif.reset_mode = NIR_LIF_RESET_SUBTRACTIVE;

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
    CHECK(strstr(dump, "\"version\":2") != NULL);
    CHECK(strstr(dump, "\"topology\":\"lif_linear\"") != NULL);
    CHECK(nir_graph_set_weights(&graph, "fc2", zeros, wn, zeros, bn,
        &error));
    nir_checkpoint_meta_t back;
    memset(&back, 0, sizeof(back));
    CHECK(nir_checkpoint_loads(dump, &graph, &back, &error));
    CHECK(error == NULL);
    lif1 = nir_graph_find(&graph, "lif1");
    CHECK(graph.nodes[lif1].params.lif.reset_mode ==
          NIR_LIF_RESET_SUBTRACTIVE);
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
    char *ver = strstr(hacked, "\"version\":2");
    CHECK(ver != NULL);
    ver[strlen("\"version\":")] = '9';
    CHECK(!nir_checkpoint_loads(hacked, &graph, NULL, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_checkpoint_loads("{\"format\":\"nope\",\"version\":2,"
        "\"nodes\":[]}", &graph, NULL, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_checkpoint_loads("{\"format\":"
        "\"spikeforge-c-checkpoint\",\"version\":1,\"nodes\":[]}",
        &graph, NULL, &error)); /* v1 version ok, drops every node */
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
    CHECK(strstr(summary, "\"reset_mode\":\"subtractive\"") != NULL);
    json_value_t *summary_json = json_parse(summary, &error);
    CHECK(error == NULL);
    nir_graph_t again;
    memset(&again, 0, sizeof(again));
    CHECK(nir_graph_load(summary_json, &again, &error));
    CHECK(error == NULL);
    size_t again_lif1 = nir_graph_find(&again, "lif1");
    CHECK(again.nodes[again_lif1].params.lif.reset_mode ==
          NIR_LIF_RESET_SUBTRACTIVE);
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

    check_opt_sgd_resume();
    check_opt_adam_resume();
    check_opt_v1_compat();
    check_opt_plain_ignores();
    check_opt_rejects();

    nir_graph_free(&graph);
    return 0;
}
