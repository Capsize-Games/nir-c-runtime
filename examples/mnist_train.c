/* Worked MNIST training example (#44): zero Python at
 * train time. Loads a committed graph plus the tiny IDX
 * fixture, trains with the C driver (encode, tape BPTT,
 * minibatch mean, Adam), prints loss/accuracy per epoch,
 * and saves a checkpoint with optimizer state.
 *
 * Usage: mnist_train <graph.json> <images.idx>
 *        <labels.idx> <out-checkpoint.json> [epochs]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/checkpoint.h"
#include "nir_c_runtime/data.h"
#include "nir_c_runtime/driver.h"
#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/train.h"

static char *read_whole(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)size + 1);
    if (buf == NULL) {
        fclose(f);
        return NULL;
    }
    if (size > 0 &&
        fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[size] = '\0';
    fclose(f);
    return buf;
}

static int load_graph(
    const char *path, nir_graph_t *graph
) {
    char *text = read_whole(path);
    if (text == NULL) {
        fprintf(stderr, "cannot open graph: %s\n", path);
        return 0;
    }
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    if (error != NULL) {
        fprintf(stderr, "graph parse failed: %s\n", error);
        free(error);
        free(text);
        return 0;
    }
    memset(graph, 0, sizeof(*graph));
    int ok = nir_graph_load(json, graph, &error);
    if (!ok) {
        fprintf(stderr, "graph load failed: %s\n",
            error != NULL ? error : "unknown");
        free(error);
    }
    json_free(json);
    free(text);
    return ok;
}

static void fill_config(nir_driver_config_t *c) {
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
    c->momentum = 0.0f;
    c->weight_decay = 0.0f;
    c->beta1 = 0.9f;
    c->beta2 = 0.999f;
    c->eps = 1e-8f;
    c->sched = NIR_DRIVER_SCHED_NONE;
}

static int load_dataset(
    const char *img_path, const char *lbl_path,
    uint8_t **px, uint8_t **lb, size_t *count,
    size_t *rows, size_t *cols
) {
    char *error = NULL;
    size_t nlabels = 0;
    if (!nir_idx_read_images(img_path, px, count, rows,
            cols, &error)) {
        fprintf(stderr, "images failed: %s\n",
            error != NULL ? error : "unknown");
        free(error);
        return 0;
    }
    if (!nir_idx_read_labels(lbl_path, lb, &nlabels,
            &error)) {
        fprintf(stderr, "labels failed: %s\n",
            error != NULL ? error : "unknown");
        free(error);
        free(*px);
        *px = NULL;
        return 0;
    }
    if (nlabels != *count) {
        fprintf(stderr, "images/labels count mismatch\n");
        free(*px);
        free(*lb);
        *px = NULL;
        *lb = NULL;
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 6) {
        fprintf(stderr,
            "usage: %s <graph> <images> <labels> <out>"
            " [epochs]\n",
            argv[0]);
        return 2;
    }
    int epochs = 20;
    if (argc == 6) {
        epochs = atoi(argv[5]);
        if (epochs <= 0 || epochs > 10000) {
            fprintf(stderr, "epochs must be 1..10000\n");
            return 2;
        }
    }
    nir_graph_t graph;
    if (!load_graph(argv[1], &graph)) {
        return 1;
    }
    uint8_t *px = NULL;
    uint8_t *lb = NULL;
    size_t count = 0;
    size_t rows = 0;
    size_t cols = 0;
    if (!load_dataset(argv[2], argv[3], &px, &lb, &count,
            &rows, &cols)) {
        nir_graph_free(&graph);
        return 1;
    }
    nir_driver_config_t config;
    fill_config(&config);
    nir_driver_t driver;
    memset(&driver, 0, sizeof(driver));
    char *error = NULL;
    if (!nir_driver_init(&driver, &graph, &config,
            &error)) {
        fprintf(stderr, "driver init failed: %s\n",
            error != NULL ? error : "unknown");
        free(error);
        free(px);
        free(lb);
        nir_graph_free(&graph);
        return 1;
    }
    nir_driver_stats_t stats;
    if (!nir_driver_eval(&driver, px, lb, count, rows,
            cols, &stats, &error)) {
        fprintf(stderr, "initial eval failed: %s\n",
            error != NULL ? error : "unknown");
        free(error);
        nir_driver_free(&driver);
        free(px);
        free(lb);
        nir_graph_free(&graph);
        return 1;
    }
    printf("initial loss %.6f acc %.3f\n", stats.loss,
        stats.accuracy);
    for (int e = 0; e < epochs; e++) {
        if (!nir_driver_train_epoch(&driver, px, lb,
                count, rows, cols, &stats, &error)) {
            fprintf(stderr, "epoch %d failed: %s\n", e,
                error != NULL ? error : "unknown");
            free(error);
            nir_driver_free(&driver);
            free(px);
            free(lb);
            nir_graph_free(&graph);
            return 1;
        }
        printf("epoch %d loss %.6f acc %.3f\n", e + 1,
            stats.loss, stats.accuracy);
    }
    if (!nir_driver_save(&driver, argv[4], "mnist_tiny",
            &error)) {
        fprintf(stderr, "checkpoint save failed: %s\n",
            error != NULL ? error : "unknown");
        free(error);
        nir_driver_free(&driver);
        free(px);
        free(lb);
        nir_graph_free(&graph);
        return 1;
    }
    printf("saved %s step %lu\n", argv[4],
        nir_driver_step(&driver));
    nir_driver_free(&driver);
    free(px);
    free(lb);
    nir_graph_free(&graph);
    return 0;
}
