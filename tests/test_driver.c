/* End-to-end driver: loss falls on the tiny fixture,
 * checkpoints resume bit-identically, and the SGD/rate/
 * schedule paths run cleanly. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/checkpoint.h"
#include "nir_c_runtime/data.h"
#include "nir_c_runtime/driver.h"
#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "test_util.h"

static nir_graph_t load_graph(const char *path) {
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

static void base_config(nir_driver_config_t *c) {
    memset(c, 0, sizeof(*c));
    c->num_steps = 4;
    c->surrogate = NIR_SURROGATE_FAST_SIGMOID;
    c->slope = 25.0f;
    c->encode = NIR_DRIVER_ENCODE_REPEAT;
    c->encode_gain = 1.0f;
    c->batch_size = 2;
    c->seed = 7;
    c->opt_kind = NIR_DRIVER_OPT_ADAM;
    c->lr = 0.01f;
    c->weight_decay = 0.0f;
    c->beta1 = 0.9f;
    c->beta2 = 0.999f;
    c->eps = 1e-8f;
    c->sched = NIR_DRIVER_SCHED_NONE;
}

static void load_tiny(
    uint8_t **px, uint8_t **lb, size_t *count,
    size_t *rows, size_t *cols
) {
    char *error = NULL;
    size_t nlabels = 0;
    CHECK(nir_idx_read_images(
        "fixtures/mnist_tiny_images.idx", px, count,
        rows, cols, &error));
    CHECK(error == NULL);
    CHECK(nir_idx_read_labels(
        "fixtures/mnist_tiny_labels.idx", lb, &nlabels,
        &error));
    CHECK(error == NULL);
    CHECK(nlabels == *count);
}

static void temp_path(const char *name, char *out) {
    const char *tmp = getenv("TMPDIR");
    if (tmp == NULL) {
        tmp = "/tmp";
    }
    sprintf(out, "%s/nir-driver-test-%s", tmp, name);
}

static void test_train_decreases(void) {
    nir_graph_t graph = load_graph(
        "../examples/mnist_tiny_graph.json");
    uint8_t *px = NULL;
    uint8_t *lb = NULL;
    size_t count = 0;
    size_t rows = 0;
    size_t cols = 0;
    load_tiny(&px, &lb, &count, &rows, &cols);
    nir_driver_config_t config;
    base_config(&config);
    nir_driver_t driver;
    memset(&driver, 0, sizeof(driver));
    char *error = NULL;
    CHECK(nir_driver_init(&driver, &graph, &config,
        &error));
    CHECK(error == NULL);
    nir_driver_stats_t before;
    CHECK(nir_driver_eval(&driver, px, lb, count, rows,
        cols, &before, &error));
    CHECK(error == NULL);
    CHECK(isfinite(before.loss));
    for (int e = 0; e < 20; e++) {
        nir_driver_stats_t st;
        CHECK(nir_driver_train_epoch(&driver, px, lb,
            count, rows, cols, &st, &error));
        CHECK(error == NULL);
        CHECK(isfinite(st.loss));
    }
    nir_driver_stats_t after;
    CHECK(nir_driver_eval(&driver, px, lb, count, rows,
        cols, &after, &error));
    CHECK(error == NULL);
    CHECK(isfinite(after.loss));
    /* Modest bound: 20 Adam epochs beat init by 0.5. */
    CHECK(after.loss + 0.5f < before.loss);
    CHECK(nir_driver_step(&driver) == 40);
    nir_driver_free(&driver);
    free(px);
    free(lb);
    nir_graph_free(&graph);
}

static void test_checkpoint_resume(void) {
    nir_graph_t graph = load_graph(
        "../examples/mnist_tiny_graph.json");
    uint8_t *px = NULL;
    uint8_t *lb = NULL;
    size_t count = 0;
    size_t rows = 0;
    size_t cols = 0;
    load_tiny(&px, &lb, &count, &rows, &cols);
    nir_driver_config_t config;
    base_config(&config);
    nir_driver_t driver;
    memset(&driver, 0, sizeof(driver));
    char *error = NULL;
    CHECK(nir_driver_init(&driver, &graph, &config,
        &error));
    for (int e = 0; e < 5; e++) {
        nir_driver_stats_t st;
        CHECK(nir_driver_train_epoch(&driver, px, lb,
            count, rows, cols, &st, &error));
    }
    nir_driver_stats_t trained;
    CHECK(nir_driver_eval(&driver, px, lb, count, rows,
        cols, &trained, &error));
    char path[512];
    temp_path("resume.json", path);
    CHECK(nir_driver_save(&driver, path, "mnist_tiny",
        &error));
    CHECK(error == NULL);
    unsigned long want_step = nir_driver_step(&driver);
    CHECK(want_step == 10);
    nir_driver_free(&driver);
    nir_graph_free(&graph);
    nir_graph_t graph2 = load_graph(
        "../examples/mnist_tiny_graph.json");
    nir_driver_t driver2;
    memset(&driver2, 0, sizeof(driver2));
    CHECK(nir_driver_init(&driver2, &graph2, &config,
        &error));
    CHECK(nir_driver_load(&driver2, path, &error));
    CHECK(error == NULL);
    CHECK(nir_driver_step(&driver2) == want_step);
    nir_driver_stats_t resumed;
    CHECK(nir_driver_eval(&driver2, px, lb, count, rows,
        cols, &resumed, &error));
    CHECK(resumed.loss == trained.loss);
    CHECK(resumed.accuracy == trained.accuracy);
    nir_driver_free(&driver2);
    nir_graph_free(&graph2);
    free(px);
    free(lb);
}

static void test_sgd_rate_sched(void) {
    nir_graph_t graph = load_graph(
        "../examples/mnist_tiny_graph.json");
    uint8_t *px = NULL;
    uint8_t *lb = NULL;
    size_t count = 0;
    size_t rows = 0;
    size_t cols = 0;
    load_tiny(&px, &lb, &count, &rows, &cols);
    char *error = NULL;
    nir_driver_config_t sgd;
    base_config(&sgd);
    sgd.opt_kind = NIR_DRIVER_OPT_SGD;
    sgd.momentum = 0.9f;
    nir_driver_t d1;
    memset(&d1, 0, sizeof(d1));
    CHECK(nir_driver_init(&d1, &graph, &sgd, &error));
    for (int e = 0; e < 2; e++) {
        nir_driver_stats_t st;
        CHECK(nir_driver_train_epoch(&d1, px, lb,
            count, rows, cols, &st, &error));
        CHECK(isfinite(st.loss));
    }
    nir_driver_free(&d1);
    nir_driver_config_t rate;
    base_config(&rate);
    rate.encode = NIR_DRIVER_ENCODE_RATE;
    nir_driver_t d2;
    memset(&d2, 0, sizeof(d2));
    CHECK(nir_driver_init(&d2, &graph, &rate, &error));
    for (int e = 0; e < 2; e++) {
        nir_driver_stats_t st;
        CHECK(nir_driver_train_epoch(&d2, px, lb,
            count, rows, cols, &st, &error));
        CHECK(isfinite(st.loss));
    }
    nir_driver_free(&d2);
    nir_driver_config_t sched;
    base_config(&sched);
    sched.sched = NIR_DRIVER_SCHED_STEP;
    sched.sched_gamma = 0.9f;
    sched.sched_step_size = 5;
    nir_driver_t d3;
    memset(&d3, 0, sizeof(d3));
    CHECK(nir_driver_init(&d3, &graph, &sched, &error));
    nir_driver_stats_t st;
    CHECK(nir_driver_train_epoch(&d3, px, lb, count,
        rows, cols, &st, &error));
    CHECK(isfinite(st.loss));
    nir_driver_free(&d3);
    nir_graph_free(&graph);
    free(px);
    free(lb);
}

static void test_errors(void) {
    nir_graph_t graph = load_graph(
        "../examples/mnist_tiny_graph.json");
    uint8_t *px = NULL;
    uint8_t *lb = NULL;
    size_t count = 0;
    size_t rows = 0;
    size_t cols = 0;
    load_tiny(&px, &lb, &count, &rows, &cols);
    char *error = NULL;
    nir_driver_config_t bad;
    base_config(&bad);
    nir_driver_t driver;
    memset(&driver, 0, sizeof(driver));
    bad.num_steps = 0;
    CHECK(!nir_driver_init(&driver, &graph, &bad,
        &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    base_config(&bad);
    bad.batch_size = 0;
    CHECK(!nir_driver_init(&driver, &graph, &bad,
        &error));
    free(error);
    error = NULL;
    base_config(&bad);
    bad.lr = -1.0f;
    CHECK(!nir_driver_init(&driver, &graph, &bad,
        &error));
    free(error);
    error = NULL;
    base_config(&bad);
    bad.slope = 0.0f;
    CHECK(!nir_driver_init(&driver, &graph, &bad,
        &error));
    free(error);
    error = NULL;
    nir_driver_config_t good;
    base_config(&good);
    CHECK(nir_driver_init(&driver, &graph, &good,
        &error));
    CHECK(error == NULL);
    nir_driver_stats_t st;
    CHECK(!nir_driver_train_epoch(&driver, NULL, lb,
        count, rows, cols, &st, &error));
    free(error);
    error = NULL;
    uint8_t bad_labels[3] = {0, 10, 1};
    CHECK(!nir_driver_train_epoch(&driver, px,
        bad_labels, count, rows, cols, &st, &error));
    free(error);
    error = NULL;
    CHECK(!nir_driver_eval(&driver, px, bad_labels,
        count, rows, cols, &st, &error));
    free(error);
    error = NULL;
    CHECK(!nir_driver_save(&driver, NULL, "t", &error));
    free(error);
    error = NULL;
    CHECK(!nir_driver_load(&driver, NULL, &error));
    free(error);
    error = NULL;
    CHECK(nir_driver_step(NULL) == 0);
    nir_driver_free(&driver);
    nir_driver_free(&driver);
    nir_driver_free(NULL);
    nir_graph_free(&graph);
    free(px);
    free(lb);
}

int main(void) {
    test_train_decreases();
    test_checkpoint_resume();
    test_sgd_rate_sched();
    test_errors();
    return 0;
}
