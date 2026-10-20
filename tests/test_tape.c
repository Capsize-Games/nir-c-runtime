/* Graph-level BPTT tape (#39).
 *
 * Forward retention matches nir_runtime_run bit for bit (small
 * Delay+LI graph plus the real exported conv_net with Delay
 * reset cycles). Multi-step gradients match central finite
 * differences on smooth topologies covering every stateless
 * kind (Affine, Linear, Conv2d, Conv1d, Flatten, AvgPool2d,
 * SumPool2d, Scale), LI recurrence, and Delay edges. The
 * single-step spiking chains (Threshold, LIF, IF, Scale) match
 * float64 closed forms of the documented surrogate rules, and
 * a scalar Delay graph proves credit flows into earlier steps
 * (its last frame has exactly zero gradient).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/checkpoint.h"
#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/loss.h"
#include "nir_c_runtime/runtime.h"
#include "nir_c_runtime/tape.h"
#include "nir_c_runtime/train.h"
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

static nir_tensor_t make_vector(const float *values, size_t n) {
    size_t shape[1] = {n};
    nir_tensor_t t = nir_tensor_alloc(shape, 1);
    CHECK(t.data != NULL);
    memcpy(t.data, values, n * sizeof(float));
    return t;
}

static void check_vector(
    const nir_tensor_t *t, const double *want, size_t n, double tol
) {
    CHECK(t->size == n);
    for (size_t i = 0; i < n; i++) {
        CHECK_NEAR(t->data[i], want[i], tol);
    }
}

static void check_finite(const nir_tensor_t *t) {
    CHECK(t != NULL && t->data != NULL);
    for (size_t i = 0; i < t->size; i++) {
        CHECK(isfinite(t->data[i]));
    }
}

static unsigned lcg_next(unsigned *state) {
    *state = *state * 1103515245u + 12345u;
    return (*state >> 16) & 0x7fffu;
}

static double fast_sigmoid(double slope, double x) {
    double d = 1.0 + fabs(slope * x);
    return slope / (d * d);
}

/* One fresh-runtime MSE evaluation: the finite-difference
 * oracle. A new runtime per call keeps temporal state
 * unprimed, matching the tape's fresh forward. */
static float run_loss(
    nir_graph_t *graph, nir_tensor_t *frames, size_t steps,
    const nir_tensor_t *target
) {
    char *error = NULL;
    nir_runtime_t runtime;
    CHECK(nir_runtime_init(&runtime, graph, &error));
    CHECK(error == NULL);
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(nir_runtime_run(&runtime, frames, steps, &readout, &error));
    CHECK(error == NULL);
    size_t shape[1] = {readout.size};
    nir_tensor_t grad = nir_tensor_alloc(shape, 1);
    CHECK(grad.data != NULL);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&readout, target, &loss, &grad));
    nir_tensor_free(&readout);
    nir_tensor_free(&grad);
    nir_runtime_free(&runtime);
    return loss;
}

/* Flat parameter views of a weighted node for FD perturbation:
 * weight count (0 when unweighted), plus the bias view (NULL/0
 * when the node has no bias). */
static void node_param_views(
    nir_node_t *node, float **w, size_t *nw, float **b, size_t *nb
) {
    *w = NULL;
    *nw = 0;
    *b = NULL;
    *nb = 0;
    switch (node->kind) {
        case NIR_NODE_AFFINE:
            *w = node->params.affine.weight;
            *nw = node->params.affine.out_features *
                node->params.affine.in_features;
            *b = node->params.affine.bias;
            if (*b != NULL) {
                *nb = node->params.affine.out_features;
            }
            break;
        case NIR_NODE_LINEAR:
            *w = node->params.linear.weight;
            *nw = node->params.linear.out_features *
                node->params.linear.in_features;
            break;
        case NIR_NODE_CONV2D:
            *w = node->params.conv2d.weight;
            *nw = node->params.conv2d.out_channels *
                node->params.conv2d.in_channels *
                node->params.conv2d.kernel_h *
                node->params.conv2d.kernel_w;
            *b = node->params.conv2d.bias;
            if (*b != NULL) {
                *nb = node->params.conv2d.out_channels;
            }
            break;
        case NIR_NODE_CONV1D:
            *w = node->params.conv1d.weight;
            *nw = node->params.conv1d.out_channels *
                node->params.conv1d.in_channels *
                node->params.conv1d.kernel;
            *b = node->params.conv1d.bias;
            if (*b != NULL) {
                *nb = node->params.conv1d.out_channels;
            }
            break;
        default:
            break;
    }
}

static void check_fd_buffer(
    nir_graph_t *graph, nir_tensor_t *frames, size_t steps,
    const nir_tensor_t *target, float *buf, const float *analytic,
    size_t count, double eps, double tol
) {
    for (size_t k = 0; k < count; k++) {
        float save = buf[k];
        buf[k] = (float)(save + eps);
        float up = run_loss(graph, frames, steps, target);
        buf[k] = (float)(save - eps);
        float down = run_loss(graph, frames, steps, target);
        buf[k] = save;
        CHECK_NEAR(analytic[k], (up - down) / (2.0 * eps), tol);
    }
}

/* Central-difference check of every parameter gradient, every
 * Scale gradient, and every input-frame gradient against the
 * runtime forward. */
static void check_fd(
    nir_graph_t *graph, nir_tape_t *tape, nir_tensor_t *frames,
    size_t steps, const nir_tensor_t *target, double eps, double tol
) {
    for (size_t i = 0; i < graph->node_count; i++) {
        nir_node_t *node = &graph->nodes[i];
        if (node->kind == NIR_NODE_SCALE) {
            float got = 0.0f;
            CHECK(nir_tape_scale_grad(tape, i, &got));
            float save = node->params.scale.scale;
            node->params.scale.scale = (float)(save + eps);
            float up = run_loss(graph, frames, steps, target);
            node->params.scale.scale = (float)(save - eps);
            float down = run_loss(graph, frames, steps, target);
            node->params.scale.scale = save;
            CHECK_NEAR(got, (up - down) / (2.0 * eps), tol);
            continue;
        }
        float *w = NULL, *b = NULL;
        size_t nw = 0, nb = 0;
        node_param_views(node, &w, &nw, &b, &nb);
        if (nw == 0) {
            continue;
        }
        const nir_tensor_t *gw = nir_tape_weight_grad(tape, i);
        CHECK(gw != NULL && gw->size == nw);
        check_fd_buffer(
            graph, frames, steps, target, w, gw->data, nw, eps,
            tol
        );
        if (nb > 0) {
            const nir_tensor_t *gb = nir_tape_bias_grad(tape, i);
            CHECK(gb != NULL && gb->size == nb);
            check_fd_buffer(
                graph, frames, steps, target, b, gb->data, nb,
                eps, tol
            );
        }
    }
    for (size_t t = 0; t < steps; t++) {
        const nir_tensor_t *gi = nir_tape_input_grad(tape, t);
        CHECK(gi != NULL && gi->size == frames[t].size);
        check_fd_buffer(
            graph, frames, steps, target, frames[t].data,
            gi->data, frames[t].size, eps, tol
        );
    }
}

/* Run one graph through a fresh runtime and through the tape,
 * and require bit-identical readouts. */
static void check_forward_exact(
    nir_graph_t *graph, nir_tensor_t *frames, size_t steps
) {
    char *error = NULL;
    nir_runtime_t runtime;
    CHECK(nir_runtime_init(&runtime, graph, &error));
    CHECK(error == NULL);
    nir_tensor_t want;
    memset(&want, 0, sizeof(want));
    CHECK(nir_runtime_run(&runtime, frames, steps, &want, &error));
    CHECK(error == NULL);
    nir_runtime_free(&runtime);
    nir_tape_t tape;
    memset(&tape, 0, sizeof(tape));
    CHECK(nir_tape_init(
        &tape, graph, steps, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error == NULL);
    nir_tensor_t got;
    memset(&got, 0, sizeof(got));
    CHECK(nir_tape_forward(&tape, frames, &got, &error));
    CHECK(error == NULL);
    CHECK(got.size == want.size);
    for (size_t i = 0; i < got.size; i++) {
        CHECK(got.data[i] == want.data[i]);
    }
    nir_tensor_free(&want);
    nir_tensor_free(&got);
    nir_tape_free(&tape);
}

static void test_forward_matches_runtime(void) {
    nir_graph_t graph = load_graph(
        "fixtures/tape_delay_li_graph.json"
    );
    float fdat[4][2] = {
        {0.5f, -0.25f},
        {-0.75f, 0.5f},
        {0.25f, 0.75f},
        {-0.125f, -0.5f},
    };
    nir_tensor_t frames[4];
    for (size_t t = 0; t < 4; t++) {
        frames[t] = make_vector(fdat[t], 2);
    }
    check_forward_exact(&graph, frames, 4);
    for (size_t t = 0; t < 4; t++) {
        nir_tensor_free(&frames[t]);
    }
    nir_graph_free(&graph);
}

/* The real exported conv_net, whose Threshold/Delay/Scale reset
 * loops are genuine Delay feedback cycles. */
static void test_forward_conv_net(void) {
    char *graph_text = read_file("fixtures/conv_net_graph.json");
    char *io_text = read_file("fixtures/conv_net_io.json");
    char *error = NULL;
    json_value_t *graph_json = json_parse(graph_text, &error);
    CHECK(error == NULL);
    json_value_t *io_json = json_parse(io_text, &error);
    CHECK(error == NULL);
    nir_graph_t graph;
    memset(&graph, 0, sizeof(graph));
    CHECK(nir_graph_load(graph_json, &graph, &error));
    CHECK(error == NULL);
    const json_value_t *shape_json = json_object_get(io_json, "shape");
    size_t steps = (size_t)json_as_number(
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
    nir_tensor_t *frames = (nir_tensor_t *)calloc(
        steps, sizeof(nir_tensor_t)
    );
    CHECK(frames != NULL);
    size_t shape[3] = {channels, height, width};
    const json_value_t *input_json = json_object_get(io_json, "input");
    for (size_t t = 0; t < steps; t++) {
        frames[t] = nir_tensor_alloc(shape, 3);
        CHECK(frames[t].data != NULL);
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
    check_forward_exact(&graph, frames, steps);
    for (size_t t = 0; t < steps; t++) {
        nir_tensor_free(&frames[t]);
    }
    free(frames);
    nir_graph_free(&graph);
    json_free(graph_json);
    json_free(io_json);
    free(graph_text);
    free(io_text);
}

static void test_fd_delay_li(void) {
    nir_graph_t graph = load_graph(
        "fixtures/tape_delay_li_graph.json"
    );
    size_t steps = 4;
    float fdat[4][2] = {
        {0.5f, -0.25f},
        {-0.75f, 0.5f},
        {0.25f, 0.75f},
        {-0.125f, -0.5f},
    };
    nir_tensor_t frames[4];
    for (size_t t = 0; t < steps; t++) {
        frames[t] = make_vector(fdat[t], 2);
    }
    float tv[] = {0.5f, -0.5f};
    nir_tensor_t target = make_vector(tv, 2);
    char *error = NULL;
    nir_tape_t tape;
    memset(&tape, 0, sizeof(tape));
    CHECK(nir_tape_init(
        &tape, &graph, steps, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error == NULL);
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(nir_tape_forward(&tape, frames, &readout, &error));
    CHECK(error == NULL);
    size_t gshape[1] = {readout.size};
    nir_tensor_t dloss = nir_tensor_alloc(gshape, 1);
    CHECK(dloss.data != NULL);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&readout, &target, &loss, &dloss));
    CHECK(nir_tape_backward(&tape, &dloss, &error));
    CHECK(error == NULL);
    /* Float32 central differences at eps=1e-3 agree to ~1e-5
     * here; 1e-3 separates rounding from formula errors. */
    check_fd(&graph, &tape, frames, steps, &target, 1e-3, 1e-3);
    /* A second backward from the same retained pass
     * reproduces every gradient bit for bit: buffers are
     * re-zeroed, never accumulated into. */
    size_t fc1 = nir_graph_find(&graph, "fc1");
    size_t fc2 = nir_graph_find(&graph, "fc2");
    float snap_w1[6], snap_b1[3], snap_w2[6], snap_b2[2];
    memcpy(snap_w1, nir_tape_weight_grad(&tape, fc1)->data,
        sizeof(snap_w1));
    memcpy(snap_b1, nir_tape_bias_grad(&tape, fc1)->data,
        sizeof(snap_b1));
    memcpy(snap_w2, nir_tape_weight_grad(&tape, fc2)->data,
        sizeof(snap_w2));
    memcpy(snap_b2, nir_tape_bias_grad(&tape, fc2)->data,
        sizeof(snap_b2));
    CHECK(nir_tape_backward(&tape, &dloss, &error));
    CHECK(error == NULL);
    CHECK(memcmp(snap_w1, nir_tape_weight_grad(&tape, fc1)->data,
        sizeof(snap_w1)) == 0);
    CHECK(memcmp(snap_b1, nir_tape_bias_grad(&tape, fc1)->data,
        sizeof(snap_b1)) == 0);
    CHECK(memcmp(snap_w2, nir_tape_weight_grad(&tape, fc2)->data,
        sizeof(snap_w2)) == 0);
    CHECK(memcmp(snap_b2, nir_tape_bias_grad(&tape, fc2)->data,
        sizeof(snap_b2)) == 0);
    for (size_t t = 0; t < steps; t++) {
        nir_tensor_free(&frames[t]);
    }
    nir_tensor_free(&target);
    nir_tensor_free(&readout);
    nir_tensor_free(&dloss);
    nir_tape_free(&tape);
    nir_graph_free(&graph);
}

static void test_delay_flow(void) {
    nir_graph_t graph = load_graph(
        "fixtures/tape_scalar_delay_graph.json"
    );
    size_t steps = 4;
    double f[4] = {0.25, -0.5, 0.75, 1.0};
    nir_tensor_t frames[4];
    for (size_t t = 0; t < steps; t++) {
        float v = (float)f[t];
        frames[t] = make_vector(&v, 1);
    }
    float tv[] = {1.0f};
    nir_tensor_t target = make_vector(tv, 1);
    char *error = NULL;
    nir_tape_t tape;
    memset(&tape, 0, sizeof(tape));
    CHECK(nir_tape_init(
        &tape, &graph, steps, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error == NULL);
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(nir_tape_forward(&tape, frames, &readout, &error));
    CHECK(error == NULL);
    size_t gshape[1] = {1};
    nir_tensor_t dloss = nir_tensor_alloc(gshape, 1);
    CHECK(dloss.data != NULL);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&readout, &target, &loss, &dloss));
    CHECK(nir_tape_backward(&tape, &dloss, &error));
    CHECK(error == NULL);
    /* Closed form: step 0 reads the unprimed Delay as 0, later
     * steps read the previous frame through it. */
    double w = 2.0, b = 0.5, tgt = 1.0;
    double o0 = b;
    double o1 = w * f[0] + b;
    double o2 = w * f[1] + b;
    double o3 = w * f[2] + b;
    double ro = (o0 + o1 + o2 + o3) / 4.0;
    double want_loss = (ro - tgt) * (ro - tgt);
    double g = 2.0 * (ro - tgt);
    CHECK_NEAR(readout.data[0], ro, 1e-6);
    CHECK_NEAR(loss, want_loss, 1e-6);
    size_t fc = nir_graph_find(&graph, "fc1");
    const nir_tensor_t *gw = nir_tape_weight_grad(&tape, fc);
    const nir_tensor_t *gb = nir_tape_bias_grad(&tape, fc);
    CHECK(gw != NULL && gb != NULL);
    CHECK_NEAR(gw->data[0], g / 4.0 * (f[0] + f[1] + f[2]), 1e-6);
    CHECK_NEAR(gb->data[0], g, 1e-6);
    for (size_t t = 0; t < 3; t++) {
        const nir_tensor_t *gi = nir_tape_input_grad(&tape, t);
        CHECK(gi != NULL);
        /* The only path from these frames to the loss crosses
         * the Delay edge into a later step. */
        CHECK_NEAR(gi->data[0], g / 4.0 * w, 1e-6);
    }
    /* The last frame feeds only the Delay buffer, which no
     * later step reads: exactly zero gradient. */
    const nir_tensor_t *glast = nir_tape_input_grad(&tape, 3);
    CHECK(glast != NULL && glast->data[0] == 0.0f);
    for (size_t t = 0; t < steps; t++) {
        nir_tensor_free(&frames[t]);
    }
    nir_tensor_free(&target);
    nir_tensor_free(&readout);
    nir_tensor_free(&dloss);
    nir_tape_free(&tape);
    nir_graph_free(&graph);
}

static void test_dense_relu_training_and_finite_difference(void) {
    nir_graph_t graph = load_graph(
        "fixtures/tape_relu_dense_graph.json"
    );
    const size_t steps = 2;
    float x0[] = {1.0f, 2.0f};
    float x1[] = {-1.0f, 0.5f};
    nir_tensor_t frames[2] = {
        make_vector(x0, 2), make_vector(x1, 2)
    };
    float target_value[] = {0.4f};
    nir_tensor_t target = make_vector(target_value, 1);

    nir_tape_t tape = {0};
    char *error = NULL;
    CHECK(nir_tape_init(
        &tape, &graph, steps, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    nir_tensor_t readout = {0};
    CHECK(nir_tape_forward(&tape, frames, &readout, &error));
    size_t output_shape[1] = {readout.size};
    nir_tensor_t grad = nir_tensor_alloc(output_shape, 1);
    CHECK(grad.data != NULL);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&readout, &target, &loss, &grad));
    CHECK(nir_tape_backward(&tape, &grad, &error));
    check_fd(&graph, &tape, frames, steps, &target, 1e-3, 1e-3);

    size_t fc1 = nir_graph_find(&graph, "fc1");
    char *checkpoint = nir_checkpoint_dumps(&graph, NULL, &error);
    CHECK(checkpoint != NULL && error == NULL);
    char *summary = nir_graph_export_summary(&graph, &error);
    CHECK(summary != NULL && error == NULL);
    json_value_t *summary_json = json_parse(summary, &error);
    CHECK(summary_json != NULL && error == NULL);
    nir_graph_t resumed_graph = {0};
    CHECK(nir_graph_load(summary_json, &resumed_graph, &error));
    CHECK(nir_graph_find(&resumed_graph, "relu") != (size_t)-1);
    CHECK(resumed_graph.nodes[nir_graph_find(
        &resumed_graph, "relu")].kind == NIR_NODE_RELU);
    size_t resumed_fc1 = nir_graph_find(&resumed_graph, "fc1");
    resumed_graph.nodes[resumed_fc1].params.affine.weight[0] = -9.0f;
    CHECK(nir_checkpoint_loads(
        checkpoint, &resumed_graph, NULL, &error
    ));
    CHECK(resumed_graph.nodes[resumed_fc1].params.affine.weight[0] ==
          graph.nodes[fc1].params.affine.weight[0]);
    nir_tape_t resumed_tape = {0};
    CHECK(nir_tape_init(
        &resumed_tape, &resumed_graph, steps,
        NIR_SURROGATE_FAST_SIGMOID, 25.0f, &error
    ));
    nir_tensor_t resumed_readout = {0};
    CHECK(nir_tape_forward(
        &resumed_tape, frames, &resumed_readout, &error
    ));
    size_t resumed_grad_shape[1] = {resumed_readout.size};
    nir_tensor_t resumed_grad = nir_tensor_alloc(resumed_grad_shape, 1);
    CHECK(resumed_grad.data != NULL);
    CHECK(nir_mse_loss(
        &resumed_readout, &target, &loss, &resumed_grad
    ));
    CHECK(nir_tape_backward(&resumed_tape, &resumed_grad, &error));
    nir_tensor_free(&resumed_grad);
    nir_tensor_free(&resumed_readout);
    nir_tape_free(&resumed_tape);
    nir_graph_free(&resumed_graph);
    json_free(summary_json);
    free(summary);
    free(checkpoint);

    const nir_tensor_t *weight_grad = nir_tape_weight_grad(&tape, fc1);
    CHECK(weight_grad != NULL && weight_grad->size == 4);
    CHECK(weight_grad->data[0] != 0.0f || weight_grad->data[1] != 0.0f ||
          weight_grad->data[2] != 0.0f || weight_grad->data[3] != 0.0f);
    nir_sgd_t optimizer;
    CHECK(nir_sgd_init(&optimizer, 4, 0.01f, 0.0f, 0.0f, &error));
    float old_weight = graph.nodes[fc1].params.affine.weight[0];
    CHECK(nir_sgd_step(
        &optimizer, graph.nodes[fc1].params.affine.weight,
        weight_grad->data, weight_grad->size
    ));
    CHECK(graph.nodes[fc1].params.affine.weight[0] != old_weight);
    nir_sgd_free(&optimizer);

    nir_tensor_free(&grad);
    nir_tensor_free(&readout);
    nir_tape_free(&tape);
    nir_tensor_free(&target);
    for (size_t t = 0; t < steps; t++) {
        nir_tensor_free(&frames[t]);
    }
    nir_graph_free(&graph);
    free(error);
}

/* One-step Affine/Threshold/LIF/Scale/Affine chain against a
 * float64 closed form of the documented surrogate rules. */
static void test_step_lif(void) {
    nir_graph_t graph = load_graph("fixtures/tape_step_graph.json");
    float xv[] = {1.5f, -0.25f};
    nir_tensor_t frame = make_vector(xv, 2);
    float tv[] = {0.3f, 0.7f};
    nir_tensor_t target = make_vector(tv, 2);
    char *error = NULL;
    nir_tape_t tape;
    memset(&tape, 0, sizeof(tape));
    CHECK(nir_tape_init(
        &tape, &graph, 1, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error == NULL);
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(nir_tape_forward(&tape, &frame, &readout, &error));
    CHECK(error == NULL);
    nir_tensor_t dloss = make_vector(tv, 2);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&readout, &target, &loss, &dloss));
    CHECK(nir_tape_backward(&tape, &dloss, &error));
    CHECK(error == NULL);
    double w1[2][2] = {{0.5, -0.5}, {0.25, 0.75}};
    double b1[2] = {0.0, 0.1};
    double w2[2][2] = {{1.0, -1.0}, {-0.5, 0.5}};
    double b2[2] = {0.1, -0.1};
    double x[2] = {1.5, -0.25};
    double tgt[2] = {0.3, 0.7};
    double hp[2], h[2], v[2], s[2], c[2], lg[2];
    for (size_t o = 0; o < 2; o++) {
        hp[o] = w1[o][0] * x[0] + w1[o][1] * x[1] + b1[o];
        h[o] = hp[o] > 0.5 ? 1.0 : 0.0;
    }
    double decay = exp(-1.0 / 2.0);
    for (size_t o = 0; o < 2; o++) {
        v[o] = 3.0 * (1.0 - decay) * h[o];
        s[o] = v[o] > 1.0 ? 1.0 : 0.0;
        c[o] = 2.0 * s[o];
    }
    CHECK(s[0] == 1.0 && s[1] == 0.0);
    for (size_t o = 0; o < 2; o++) {
        lg[o] = w2[o][0] * c[0] + w2[o][1] * c[1] + b2[o];
    }
    double dl[2] = {lg[0] - tgt[0], lg[1] - tgt[1]};
    double dc[2], ds[2], g[2], dh[2], dhp[2];
    double gscale = 0.0;
    for (size_t i = 0; i < 2; i++) {
        dc[i] = w2[0][i] * dl[0] + w2[1][i] * dl[1];
        gscale += dc[i] * s[i];
        ds[i] = dc[i] * 2.0;
        g[i] = ds[i] * fast_sigmoid(25.0, v[i] - 1.0);
        dh[i] = g[i] * 3.0 * (1.0 - decay);
        dhp[i] = dh[i] * fast_sigmoid(25.0, hp[i] - 0.5);
    }
    double want_w1[4], want_w2[4], want_dx[2];
    for (size_t o = 0; o < 2; o++) {
        for (size_t i = 0; i < 2; i++) {
            want_w1[o * 2 + i] = dhp[o] * x[i];
            want_w2[o * 2 + i] = dl[o] * c[i];
        }
    }
    for (size_t i = 0; i < 2; i++) {
        want_dx[i] = w1[0][i] * dhp[0] + w1[1][i] * dhp[1];
    }
    size_t fc1 = nir_graph_find(&graph, "fc1");
    size_t fc2 = nir_graph_find(&graph, "fc2");
    size_t sc = nir_graph_find(&graph, "sc");
    check_vector(nir_tape_weight_grad(&tape, fc1), want_w1, 4, 1e-5);
    check_vector(nir_tape_bias_grad(&tape, fc1), dhp, 2, 1e-5);
    check_vector(nir_tape_weight_grad(&tape, fc2), want_w2, 4, 1e-5);
    check_vector(nir_tape_bias_grad(&tape, fc2), dl, 2, 1e-5);
    float got_scale = 0.0f;
    CHECK(nir_tape_scale_grad(&tape, sc, &got_scale));
    CHECK_NEAR(got_scale, gscale, 1e-5);
    check_vector(nir_tape_input_grad(&tape, 0), want_dx, 2, 1e-5);
    nir_tensor_free(&frame);
    nir_tensor_free(&target);
    nir_tensor_free(&readout);
    nir_tensor_free(&dloss);
    nir_tape_free(&tape);
    nir_graph_free(&graph);
}

static void test_step_if(void) {
    nir_graph_t graph = load_graph("fixtures/tape_if_graph.json");
    float xv[] = {1.5f, -0.25f};
    nir_tensor_t frame = make_vector(xv, 2);
    float tv[] = {0.3f, 0.7f};
    nir_tensor_t target = make_vector(tv, 2);
    char *error = NULL;
    nir_tape_t tape;
    memset(&tape, 0, sizeof(tape));
    CHECK(nir_tape_init(
        &tape, &graph, 1, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error == NULL);
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(nir_tape_forward(&tape, &frame, &readout, &error));
    CHECK(error == NULL);
    nir_tensor_t dloss = make_vector(tv, 2);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&readout, &target, &loss, &dloss));
    CHECK(nir_tape_backward(&tape, &dloss, &error));
    CHECK(error == NULL);
    double w1[2][2] = {{0.5, -0.5}, {0.25, 0.75}};
    double b1[2] = {0.0, 0.1};
    double w2[2][2] = {{1.0, -1.0}, {-0.5, 0.5}};
    double b2[2] = {0.1, -0.1};
    double x[2] = {1.5, -0.25};
    double tgt[2] = {0.3, 0.7};
    double hp[2], m[2], s[2], lg[2];
    for (size_t o = 0; o < 2; o++) {
        hp[o] = w1[o][0] * x[0] + w1[o][1] * x[1] + b1[o];
        m[o] = 2.0 * hp[o];
        s[o] = m[o] > 1.0 ? 1.0 : 0.0;
    }
    CHECK(s[0] == 1.0 && s[1] == 0.0);
    for (size_t o = 0; o < 2; o++) {
        lg[o] = w2[o][0] * s[0] + w2[o][1] * s[1] + b2[o];
    }
    double dl[2] = {lg[0] - tgt[0], lg[1] - tgt[1]};
    double ds[2], g[2], dh[2];
    for (size_t i = 0; i < 2; i++) {
        ds[i] = w2[0][i] * dl[0] + w2[1][i] * dl[1];
        g[i] = ds[i] * fast_sigmoid(25.0, m[i] - 1.0);
        dh[i] = g[i] * 2.0;
    }
    double want_w1[4], want_w2[4], want_dx[2];
    for (size_t o = 0; o < 2; o++) {
        for (size_t i = 0; i < 2; i++) {
            want_w1[o * 2 + i] = dh[o] * x[i];
            want_w2[o * 2 + i] = dl[o] * s[i];
        }
    }
    for (size_t i = 0; i < 2; i++) {
        want_dx[i] = w1[0][i] * dh[0] + w1[1][i] * dh[1];
    }
    size_t fc1 = nir_graph_find(&graph, "fc1");
    size_t fc2 = nir_graph_find(&graph, "fc2");
    check_vector(nir_tape_weight_grad(&tape, fc1), want_w1, 4, 1e-5);
    check_vector(nir_tape_bias_grad(&tape, fc1), dh, 2, 1e-5);
    check_vector(nir_tape_weight_grad(&tape, fc2), want_w2, 4, 1e-5);
    check_vector(nir_tape_bias_grad(&tape, fc2), dl, 2, 1e-5);
    check_vector(nir_tape_input_grad(&tape, 0), want_dx, 2, 1e-5);
    nir_tensor_free(&frame);
    nir_tensor_free(&target);
    nir_tensor_free(&readout);
    nir_tensor_free(&dloss);
    nir_tape_free(&tape);
    nir_graph_free(&graph);
}

static void test_subtractive_lif_reset_bptt(void) {
    const char *text =
        "{\"nodes\":["
        "{\"name\":\"input\",\"kind\":\"Input\"},"
        "{\"name\":\"lif\",\"kind\":\"LIF\",\"params\":{"
        "\"tau\":1,\"r\":1,\"v_leak\":0,\"v_threshold\":0.8,"
        "\"v_reset\":0,\"reset_mode\":\"subtractive\"}},"
        "{\"name\":\"output\",\"kind\":\"Output\"}],"
        "\"edges\":[{\"source\":\"input\",\"target\":\"lif\"},"
        "{\"source\":\"lif\",\"target\":\"output\"}]}";
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(json != NULL && error == NULL);
    nir_graph_t graph;
    CHECK(nir_graph_load(json, &graph, &error));

    float x0[] = {2.0f}, x1[] = {1.0f};
    nir_tensor_t frames[2] = {
        make_vector(x0, 1), make_vector(x1, 1)
    };
    nir_tape_t tape;
    memset(&tape, 0, sizeof(tape));
    CHECK(nir_tape_init(
        &tape, &graph, 2, NIR_SURROGATE_FAST_SIGMOID, 2.0f, &error
    ));
    nir_tensor_t readout = {0};
    CHECK(nir_tape_forward(&tape, frames, &readout, &error));
    CHECK_NEAR(readout.data[0], 1.0f, 1e-6);
    size_t lif_index = nir_graph_find(&graph, "lif");
    size_t n = graph.node_count;
    double decay = exp(-1.0);
    double c = 1.0 - decay;
    double membrane0 = 2.0 * c;
    double stored0 = membrane0 - 0.8;
    double membrane1 = decay * stored0 + c;
    double stored1 = membrane1 - 0.8;
    CHECK_NEAR(tape.values[lif_index].data[0], 1.0f, 1e-6);
    CHECK_NEAR(tape.stored[lif_index].data[0], stored0, 1e-6);
    CHECK_NEAR(tape.values[n + lif_index].data[0], 1.0f, 1e-6);
    CHECK_NEAR(tape.stored[n + lif_index].data[0], stored1, 1e-6);

    size_t target_shape[1] = {1};
    nir_tensor_t grad_readout = nir_tensor_alloc(target_shape, 1);
    grad_readout.data[0] = 2.0f; /* d((readout - 0)^2)/d(readout) */
    CHECK(nir_tape_backward(&tape, &grad_readout, &error));
    double s0 = fast_sigmoid(2.0, membrane0 - 0.8);
    double s1 = fast_sigmoid(2.0, membrane1 - 0.8);
    /* Output averaging seeds each step with 1.0. Step 1's stored
     * gradient crosses the immediate subtractive reset at step 0. */
    double reset_gradient = s1 * decay;
    double membrane_gradient0 = s0 + reset_gradient * (1.0 - 0.8 * s0);
    double want_x0 = membrane_gradient0 * c;
    double want_x1 = s1 * c;
    CHECK_NEAR(nir_tape_input_grad(&tape, 0)->data[0], want_x0, 1e-6);
    CHECK_NEAR(nir_tape_input_grad(&tape, 1)->data[0], want_x1, 1e-6);

    nir_tensor_free(&grad_readout);
    nir_tensor_free(&readout);
    nir_tape_free(&tape);
    nir_tensor_free(&frames[0]);
    nir_tensor_free(&frames[1]);
    nir_graph_free(&graph);
    json_free(json);
}

static void fill_lcg(nir_tensor_t *t, unsigned *state) {
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] =
            ((float)(int)(lcg_next(state) % 2001u) - 1000.0f) /
            2000.0f;
    }
}

static void test_fd_conv(void) {
    nir_graph_t graph = load_graph(
        "fixtures/tape_conv_graph.json"
    );
    size_t steps = 2;
    size_t shape[3] = {1, 5, 5};
    nir_tensor_t frames[2];
    unsigned state = 0xC0FFEEu;
    for (size_t t = 0; t < steps; t++) {
        frames[t] = nir_tensor_alloc(shape, 3);
        CHECK(frames[t].data != NULL);
        fill_lcg(&frames[t], &state);
    }
    float tv[] = {0.3f, -0.2f};
    nir_tensor_t target = make_vector(tv, 2);
    char *error = NULL;
    nir_tape_t tape;
    memset(&tape, 0, sizeof(tape));
    CHECK(nir_tape_init(
        &tape, &graph, steps, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error == NULL);
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(nir_tape_forward(&tape, frames, &readout, &error));
    CHECK(error == NULL);
    size_t gshape[1] = {readout.size};
    nir_tensor_t dloss = nir_tensor_alloc(gshape, 1);
    CHECK(dloss.data != NULL);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&readout, &target, &loss, &dloss));
    CHECK(nir_tape_backward(&tape, &dloss, &error));
    CHECK(error == NULL);
    check_fd(&graph, &tape, frames, steps, &target, 1e-3, 1e-3);
    for (size_t t = 0; t < steps; t++) {
        nir_tensor_free(&frames[t]);
    }
    nir_tensor_free(&target);
    nir_tensor_free(&readout);
    nir_tensor_free(&dloss);
    nir_tape_free(&tape);
    nir_graph_free(&graph);
}

static void test_fd_conv1d(void) {
    nir_graph_t graph = load_graph(
        "fixtures/tape_conv1d_graph.json"
    );
    size_t steps = 2;
    size_t shape[2] = {2, 4};
    nir_tensor_t frames[2];
    unsigned state = 0x1DECAFEu;
    for (size_t t = 0; t < steps; t++) {
        frames[t] = nir_tensor_alloc(shape, 2);
        CHECK(frames[t].data != NULL);
        fill_lcg(&frames[t], &state);
    }
    float tv[] = {-0.4f, 0.6f};
    nir_tensor_t target = make_vector(tv, 2);
    char *error = NULL;
    nir_tape_t tape;
    memset(&tape, 0, sizeof(tape));
    CHECK(nir_tape_init(
        &tape, &graph, steps, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error == NULL);
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(nir_tape_forward(&tape, frames, &readout, &error));
    CHECK(error == NULL);
    size_t gshape[1] = {readout.size};
    nir_tensor_t dloss = nir_tensor_alloc(gshape, 1);
    CHECK(dloss.data != NULL);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&readout, &target, &loss, &dloss));
    CHECK(nir_tape_backward(&tape, &dloss, &error));
    CHECK(error == NULL);
    check_fd(&graph, &tape, frames, steps, &target, 1e-3, 1e-3);
    for (size_t t = 0; t < steps; t++) {
        nir_tensor_free(&frames[t]);
    }
    nir_tensor_free(&target);
    nir_tensor_free(&readout);
    nir_tensor_free(&dloss);
    nir_tape_free(&tape);
    nir_graph_free(&graph);
}

/* Multi-step Threshold/LIF/IF/CubaLIF chain: spikes are not
 * finite-differenceable, so this pins successful wiring (no
 * crash, finite gradients, deterministic re-runs) while the
 * single-step tests above pin exactness. */
static void test_spike_smoke(void) {
    nir_graph_t graph = load_graph(
        "fixtures/tape_spike_graph.json"
    );
    size_t steps = 4;
    float fdat[4][2] = {
        {0.5f, -0.25f},
        {-0.75f, 0.5f},
        {0.25f, 0.75f},
        {-0.125f, -0.5f},
    };
    nir_tensor_t frames[4];
    for (size_t t = 0; t < steps; t++) {
        frames[t] = make_vector(fdat[t], 2);
    }
    float tv[] = {0.5f, -0.5f};
    nir_tensor_t target = make_vector(tv, 2);
    char *error = NULL;
    nir_surrogate_kind_t kind;
    CHECK(nir_surrogate_from_name("atan", &kind, &error));
    CHECK(error == NULL && kind == NIR_SURROGATE_ATAN);
    nir_tape_t tape;
    memset(&tape, 0, sizeof(tape));
    CHECK(nir_tape_init(&tape, &graph, steps, kind, 2.0f, &error));
    CHECK(error == NULL);
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(nir_tape_forward(&tape, frames, &readout, &error));
    CHECK(error == NULL);
    check_finite(&readout);
    size_t gshape[1] = {readout.size};
    nir_tensor_t dloss = nir_tensor_alloc(gshape, 1);
    CHECK(dloss.data != NULL);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&readout, &target, &loss, &dloss));
    CHECK(isfinite(loss));
    CHECK(nir_tape_backward(&tape, &dloss, &error));
    CHECK(error == NULL);
    size_t fc1 = nir_graph_find(&graph, "fc1");
    size_t fc2 = nir_graph_find(&graph, "fc2");
    check_finite(nir_tape_weight_grad(&tape, fc1));
    check_finite(nir_tape_bias_grad(&tape, fc1));
    check_finite(nir_tape_weight_grad(&tape, fc2));
    check_finite(nir_tape_bias_grad(&tape, fc2));
    for (size_t t = 0; t < steps; t++) {
        check_finite(nir_tape_input_grad(&tape, t));
    }
    float snap1[6], snap2[6];
    memcpy(snap1, nir_tape_weight_grad(&tape, fc1)->data,
        sizeof(snap1));
    memcpy(snap2, nir_tape_weight_grad(&tape, fc2)->data,
        sizeof(snap2));
    CHECK(nir_tape_backward(&tape, &dloss, &error));
    CHECK(error == NULL);
    CHECK(memcmp(snap1, nir_tape_weight_grad(&tape, fc1)->data,
        sizeof(snap1)) == 0);
    CHECK(memcmp(snap2, nir_tape_weight_grad(&tape, fc2)->data,
        sizeof(snap2)) == 0);
    for (size_t t = 0; t < steps; t++) {
        nir_tensor_free(&frames[t]);
    }
    nir_tensor_free(&target);
    nir_tensor_free(&readout);
    nir_tensor_free(&dloss);
    nir_tape_free(&tape);
    nir_graph_free(&graph);
}

static void test_errors(void) {
    nir_graph_t graph = load_graph(
        "fixtures/tape_delay_li_graph.json"
    );
    char *error = NULL;
    nir_tape_t tape;
    memset(&tape, 0, sizeof(tape));
    CHECK(!nir_tape_init(
        NULL, &graph, 4, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_tape_init(
        &tape, NULL, 4, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    nir_tape_free(&tape);
    CHECK(!nir_tape_init(
        &tape, &graph, 0, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    nir_tape_free(&tape);
    const float bad_slopes[] = {0.0f, -2.0f, (float)INFINITY,
        (float)NAN};
    for (size_t i = 0; i < 4; i++) {
        CHECK(!nir_tape_init(
            &tape, &graph, 4, NIR_SURROGATE_ATAN, bad_slopes[i],
            &error
        ));
        CHECK(error != NULL);
        free(error);
        error = NULL;
        nir_tape_free(&tape);
    }
    CHECK(!nir_tape_init(
        &tape, &graph, 4, (nir_surrogate_kind_t)99, 2.0f,
        &error
    ));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    nir_tape_free(&tape);
    /* A cycle no Delay breaks fails init, like the runtime. */
    const char *cyclic =
        "{\"nodes\":["
        "{\"name\":\"input\",\"kind\":\"Input\",\"params\":{}},"
        "{\"name\":\"fc\",\"kind\":\"Affine\","
        "\"params\":{\"weight\":[[1.0]],\"bias\":[0.0]}},"
        "{\"name\":\"output\",\"kind\":\"Output\",\"params\":{}}],"
        "\"edges\":["
        "{\"source\":\"input\",\"target\":\"fc\"},"
        "{\"source\":\"fc\",\"target\":\"fc\"},"
        "{\"source\":\"fc\",\"target\":\"output\"}]}";
    json_value_t *cyclic_json = json_parse(cyclic, &error);
    CHECK(error == NULL);
    nir_graph_t loop;
    memset(&loop, 0, sizeof(loop));
    CHECK(nir_graph_load(cyclic_json, &loop, &error));
    CHECK(error == NULL);
    CHECK(!nir_tape_init(
        &tape, &loop, 2, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error != NULL && strstr(error, "Delay") != NULL);
    free(error);
    error = NULL;
    nir_tape_free(&tape);
    json_free(cyclic_json);
    nir_graph_free(&loop);
    /* Sequencing on a live tape. */
    CHECK(nir_tape_init(
        &tape, &graph, 4, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &error
    ));
    CHECK(error == NULL);
    float fv[] = {0.5f, -0.25f};
    nir_tensor_t frame = make_vector(fv, 2);
    nir_tensor_t readout;
    memset(&readout, 0, sizeof(readout));
    CHECK(!nir_tape_forward(&tape, NULL, &readout, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_tape_forward(&tape, &frame, NULL, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_tape_forward(NULL, &frame, &readout, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    size_t fc1 = nir_graph_find(&graph, "fc1");
    size_t li1 = nir_graph_find(&graph, "li1");
    CHECK(!nir_tape_backward(&tape, &frame, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    CHECK(nir_tape_weight_grad(&tape, fc1) == NULL);
    CHECK(nir_tape_bias_grad(&tape, fc1) == NULL);
    float scale_out = 0.0f;
    CHECK(!nir_tape_scale_grad(&tape, fc1, &scale_out));
    CHECK(nir_tape_input_grad(&tape, 0) == NULL);
    /* Misshapen frames fail the forward and retain
     * nothing: backward still refuses afterwards. */
    float bv[] = {1.0f, 2.0f, 3.0f};
    nir_tensor_t badframes[4];
    for (size_t t = 0; t < 4; t++) {
        badframes[t] = make_vector(bv, 3);
    }
    CHECK(!nir_tape_forward(&tape, badframes, &readout, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    for (size_t t = 0; t < 4; t++) {
        nir_tensor_free(&badframes[t]);
    }
    CHECK(!nir_tape_backward(&tape, &frame, &error));
    free(error);
    error = NULL;
    nir_tensor_t frames[4];
    for (size_t t = 0; t < 4; t++) {
        frames[t] = make_vector(fv, 2);
    }
    CHECK(nir_tape_forward(&tape, frames, &readout, &error));
    CHECK(error == NULL);
    /* Still no gradients until backward runs. */
    CHECK(nir_tape_weight_grad(&tape, fc1) == NULL);
    CHECK(nir_tape_input_grad(&tape, 0) == NULL);
    CHECK(!nir_tape_backward(&tape, NULL, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    float wv[] = {1.0f, 2.0f, 3.0f};
    nir_tensor_t wrong = make_vector(wv, 3);
    CHECK(!nir_tape_backward(&tape, &wrong, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    size_t rshape[1] = {readout.size};
    nir_tensor_t dloss = nir_tensor_alloc(rshape, 1);
    CHECK(dloss.data != NULL);
    for (size_t i = 0; i < dloss.size; i++) {
        dloss.data[i] = 1.0f;
    }
    CHECK(nir_tape_backward(&tape, &dloss, &error));
    CHECK(error == NULL);
    /* Accessor kind and range guards. */
    CHECK(nir_tape_weight_grad(&tape, li1) == NULL);
    CHECK(nir_tape_bias_grad(&tape, li1) == NULL);
    CHECK(nir_tape_weight_grad(&tape, graph.node_count) == NULL);
    CHECK(nir_tape_bias_grad(&tape, graph.node_count) == NULL);
    CHECK(nir_tape_weight_grad(NULL, fc1) == NULL);
    CHECK(!nir_tape_scale_grad(&tape, fc1, &scale_out));
    CHECK(!nir_tape_scale_grad(&tape, graph.node_count, &scale_out));
    CHECK(!nir_tape_scale_grad(&tape, fc1, NULL));
    CHECK(nir_tape_input_grad(&tape, 4) == NULL);
    CHECK(nir_tape_input_grad(NULL, 0) == NULL);
    CHECK(nir_tape_input_grad(&tape, 0) != NULL);
    /* A re-forward discards the old gradients. */
    nir_tensor_free(&readout);
    CHECK(nir_tape_forward(&tape, frames, &readout, &error));
    CHECK(error == NULL);
    CHECK(nir_tape_weight_grad(&tape, fc1) == NULL);
    nir_tensor_free(&frame);
    nir_tensor_free(&wrong);
    for (size_t t = 0; t < 4; t++) {
        nir_tensor_free(&frames[t]);
    }
    nir_tensor_free(&readout);
    nir_tensor_free(&dloss);
    nir_tape_free(&tape);
    nir_tape_free(&tape);
    nir_tape_free(NULL);
    nir_graph_free(&graph);
    memset(&tape, 0, sizeof(tape));
    nir_tape_free(&tape);
}

int main(void) {
    test_forward_matches_runtime();
    test_forward_conv_net();
    test_fd_delay_li();
    test_delay_flow();
    test_dense_relu_training_and_finite_difference();
    test_step_lif();
    test_subtractive_lif_reset_bptt();
    test_step_if();
    test_fd_conv();
    test_fd_conv1d();
    test_spike_smoke();
    test_errors();
    return 0;
}
