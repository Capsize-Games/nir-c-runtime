/* Backward ops for spatial/linear nodes (Conv2d, Conv1d, Flatten,
 * AvgPool2d, SumPool2d, Scale) and spiking neurons (LIF/IF/CubaLIF).
 *
 * Linear ops are checked against hand-computed closed forms plus a
 * central-difference sweep of the forward op (stride/padding/
 * dilation/multi-channel index paths); spiking-neuron grads are
 * checked against float64 evaluations of the documented surrogate
 * chain rule on both the spike-fired and no-spike branches, with
 * the branch labels pinned by the forward op.
 */
#include <math.h>
#include <string.h>

#include "nir_c_runtime/ops.h"
#include "test_util.h"

static nir_tensor_t make_tensor(
    const float *values, const size_t *shape, size_t ndim
) {
    nir_tensor_t t = nir_tensor_alloc(shape, ndim);
    CHECK(t.data != NULL);
    memcpy(t.data, values, t.size * sizeof(float));
    return t;
}

static void check_data(
    const nir_tensor_t *t, const double *want, size_t n, double tol
) {
    CHECK(t->size == n);
    for (size_t i = 0; i < n; i++) {
        CHECK_NEAR(t->data[i], want[i], tol);
    }
}

static unsigned lcg_next(unsigned *state) {
    *state = *state * 1103515245u + 12345u;
    return (*state >> 16) & 0x7fffu;
}

static void fill_lcg(float *data, size_t n, unsigned *state) {
    for (size_t i = 0; i < n; i++) {
        data[i] = ((float)(int)(lcg_next(state) % 2001u) - 1000.0f)
            / 1000.0f;
    }
}

static double fast_sigmoid_grad(double slope, double x) {
    double d = 1.0 + fabs(slope * x);
    return slope / (d * d);
}

static double atan_grad(double slope, double x) {
    double d = 1.5707963267948966 * slope * x;
    return (slope / 2.0) / (1.0 + d * d);
}

static void test_conv2d_backward_hand(void) {
    /* x = 1..9 row-major, w = [[1,0],[0,-1]], b = 0.5,
     * gy = [[1,2],[3,4]]. */
    float weight[4] = {1.0f, 0.0f, 0.0f, -1.0f};
    float bias[1] = {0.5f};
    nir_conv2d_params_t params = {
        1, 1, 2, 2, weight, bias, 1, 1, 0, 0, 1, 1
    };
    float xv[9];
    for (size_t i = 0; i < 9; i++) xv[i] = (float)(i + 1);
    size_t xshape[3] = {1, 3, 3};
    nir_tensor_t input = make_tensor(xv, xshape, 3);
    float gv[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t gshape[3] = {1, 2, 2};
    nir_tensor_t grad_out = make_tensor(gv, gshape, 3);
    size_t wshape[1] = {4};
    size_t bshape[1] = {1};
    nir_tensor_t grad_w = nir_tensor_alloc(wshape, 1);
    nir_tensor_t grad_b = nir_tensor_alloc(bshape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(xshape, 3);
    CHECK(nir_op_conv2d_backward(
        &params, &input, &grad_out, &grad_w, &grad_b, &grad_x
    ));
    double want_w[] = {37.0, 47.0, 67.0, 77.0};
    double want_b[] = {10.0};
    double want_x[] = {1.0, 2.0, 0.0, 3.0, 3.0, -2.0, 0.0, -3.0,
        -4.0};
    CHECK(grad_w.ndim == 1);
    check_data(&grad_w, want_w, 4, 1e-6);
    check_data(&grad_b, want_b, 1, 1e-6);
    CHECK(grad_x.ndim == 3 && grad_x.shape[1] == 3);
    check_data(&grad_x, want_x, 9, 1e-6);
    /* Weight/bias grads accumulate; grad_input is overwritten. */
    CHECK(nir_op_conv2d_backward(
        &params, &input, &grad_out, &grad_w, &grad_b, &grad_x
    ));
    double want_w2[] = {74.0, 94.0, 134.0, 154.0};
    double want_b2[] = {20.0};
    check_data(&grad_w, want_w2, 4, 1e-6);
    check_data(&grad_b, want_b2, 1, 1e-6);
    check_data(&grad_x, want_x, 9, 1e-6);
    /* Unbiased nodes take no bias slot; a mismatch is rejected. */
    params.bias = NULL;
    memset(grad_w.data, 0, 4 * sizeof(float));
    CHECK(nir_op_conv2d_backward(
        &params, &input, &grad_out, &grad_w, NULL, &grad_x
    ));
    check_data(&grad_w, want_w, 4, 1e-6);
    check_data(&grad_x, want_x, 9, 1e-6);
    CHECK(!nir_op_conv2d_backward(
        &params, &input, &grad_out, &grad_w, &grad_b, &grad_x
    ));
    params.bias = bias;
    CHECK(!nir_op_conv2d_backward(
        &params, &input, &grad_out, &grad_w, NULL, &grad_x
    ));
    nir_tensor_free(&input);
    nir_tensor_free(&grad_out);
    nir_tensor_free(&grad_w);
    nir_tensor_free(&grad_b);
    nir_tensor_free(&grad_x);
}

/* Central-difference gradient of dot(conv2d(params, input), gy)
 * with respect to one element, perturbed through `slot`. */
static double conv2d_fd_elem(
    nir_conv2d_params_t *params, nir_tensor_t *input,
    const nir_tensor_t *gy, float *slot, double eps
) {
    nir_tensor_t out = nir_tensor_alloc(gy->shape, gy->ndim);
    CHECK(out.data != NULL);
    float orig = *slot;
    *slot = orig + (float)eps;
    CHECK(nir_op_conv2d_into(params, input, &out));
    double plus = 0.0;
    for (size_t i = 0; i < out.size; i++) {
        plus += (double)out.data[i] * (double)gy->data[i];
    }
    *slot = orig - (float)eps;
    CHECK(nir_op_conv2d_into(params, input, &out));
    double minus = 0.0;
    for (size_t i = 0; i < out.size; i++) {
        minus += (double)out.data[i] * (double)gy->data[i];
    }
    *slot = orig;
    nir_tensor_free(&out);
    return (plus - minus) / (2.0 * eps);
}

static void check_conv2d_fd(
    size_t in_ch, size_t out_ch, size_t h, size_t w, size_t kh,
    size_t kw, int sh, int sw, int ph, int pw, int dh, int dw,
    unsigned seed
) {
    size_t wc = out_ch * in_ch * kh * kw;
    float *weight = malloc(wc * sizeof(float));
    float *bias = malloc(out_ch * sizeof(float));
    CHECK(weight != NULL && bias != NULL);
    fill_lcg(weight, wc, &seed);
    fill_lcg(bias, out_ch, &seed);
    nir_conv2d_params_t params = {
        out_ch, in_ch, kh, kw, weight, bias, sh, sw, ph, pw, dh,
        dw
    };
    size_t xshape[3] = {in_ch, h, w};
    nir_tensor_t input = nir_tensor_alloc(xshape, 3);
    CHECK(input.data != NULL);
    fill_lcg(input.data, input.size, &seed);
    nir_tensor_t probe = nir_op_conv2d(&params, &input);
    CHECK(probe.data != NULL);
    nir_tensor_t gy = nir_tensor_alloc(probe.shape, probe.ndim);
    CHECK(gy.data != NULL);
    fill_lcg(gy.data, gy.size, &seed);
    size_t wshape[1] = {wc};
    size_t bshape[1] = {out_ch};
    nir_tensor_t grad_w = nir_tensor_alloc(wshape, 1);
    nir_tensor_t grad_b = nir_tensor_alloc(bshape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(xshape, 3);
    CHECK(nir_op_conv2d_backward(
        &params, &input, &gy, &grad_w, &grad_b, &grad_x
    ));
    double eps = 0.01;
    double tol = 0.01;
    for (size_t i = 0; i < input.size; i++) {
        double num = conv2d_fd_elem(
            &params, &input, &gy, &input.data[i], eps
        );
        CHECK_NEAR(grad_x.data[i], num, tol);
    }
    for (size_t i = 0; i < wc; i++) {
        double num = conv2d_fd_elem(
            &params, &input, &gy, &weight[i], eps
        );
        CHECK_NEAR(grad_w.data[i], num, tol);
    }
    for (size_t i = 0; i < out_ch; i++) {
        double num = conv2d_fd_elem(
            &params, &input, &gy, &bias[i], eps
        );
        CHECK_NEAR(grad_b.data[i], num, tol);
    }
    nir_tensor_free(&input);
    nir_tensor_free(&probe);
    nir_tensor_free(&gy);
    nir_tensor_free(&grad_w);
    nir_tensor_free(&grad_b);
    nir_tensor_free(&grad_x);
    free(weight);
    free(bias);
}

static void test_conv2d_backward_finite_diff(void) {
    /* Strided, padded, multi-channel. */
    check_conv2d_fd(2, 2, 5, 5, 3, 3, 2, 2, 1, 1, 1, 1, 42u);
    /* Dilated with padding. */
    check_conv2d_fd(1, 1, 5, 5, 3, 3, 1, 1, 2, 2, 2, 2, 7u);
}

static void test_conv1d_backward_hand(void) {
    /* x = [1,2,3,4], w = [1,0,-1], b = 0.5, pad 1,
     * gy = [1,0,-1,2]. */
    float weight[3] = {1.0f, 0.0f, -1.0f};
    float bias[1] = {0.5f};
    nir_conv1d_params_t params = {1, 1, 3, weight, bias, 1, 1, 1};
    float xv[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t xshape[2] = {1, 4};
    nir_tensor_t input = make_tensor(xv, xshape, 2);
    float gv[4] = {1.0f, 0.0f, -1.0f, 2.0f};
    size_t gshape[2] = {1, 4};
    nir_tensor_t grad_out = make_tensor(gv, gshape, 2);
    size_t wshape[1] = {3};
    size_t bshape[1] = {1};
    nir_tensor_t grad_w = nir_tensor_alloc(wshape, 1);
    nir_tensor_t grad_b = nir_tensor_alloc(bshape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(xshape, 2);
    CHECK(nir_op_conv1d_backward(
        &params, &input, &grad_out, &grad_w, &grad_b, &grad_x
    ));
    double want_w[] = {4.0, 6.0, -2.0};
    double want_b[] = {2.0};
    double want_x[] = {0.0, -2.0, 2.0, 1.0};
    check_data(&grad_w, want_w, 3, 1e-6);
    check_data(&grad_b, want_b, 1, 1e-6);
    CHECK(grad_x.ndim == 2 && grad_x.shape[1] == 4);
    check_data(&grad_x, want_x, 4, 1e-6);
    /* Accumulation and the bias-slot rule mirror conv2d. */
    CHECK(nir_op_conv1d_backward(
        &params, &input, &grad_out, &grad_w, &grad_b, &grad_x
    ));
    double want_w2[] = {8.0, 12.0, -4.0};
    check_data(&grad_w, want_w2, 3, 1e-6);
    check_data(&grad_x, want_x, 4, 1e-6);
    params.bias = NULL;
    CHECK(!nir_op_conv1d_backward(
        &params, &input, &grad_out, &grad_w, &grad_b, &grad_x
    ));
    CHECK(nir_op_conv1d_backward(
        &params, &input, &grad_out, &grad_w, NULL, &grad_x
    ));
    nir_tensor_free(&input);
    nir_tensor_free(&grad_out);
    nir_tensor_free(&grad_w);
    nir_tensor_free(&grad_b);
    nir_tensor_free(&grad_x);
}

static double conv1d_fd_elem(
    nir_conv1d_params_t *params, nir_tensor_t *input,
    const nir_tensor_t *gy, float *slot, double eps
) {
    nir_tensor_t out = nir_tensor_alloc(gy->shape, gy->ndim);
    CHECK(out.data != NULL);
    float orig = *slot;
    *slot = orig + (float)eps;
    CHECK(nir_op_conv1d_into(params, input, &out));
    double plus = 0.0;
    for (size_t i = 0; i < out.size; i++) {
        plus += (double)out.data[i] * (double)gy->data[i];
    }
    *slot = orig - (float)eps;
    CHECK(nir_op_conv1d_into(params, input, &out));
    double minus = 0.0;
    for (size_t i = 0; i < out.size; i++) {
        minus += (double)out.data[i] * (double)gy->data[i];
    }
    *slot = orig;
    nir_tensor_free(&out);
    return (plus - minus) / (2.0 * eps);
}

static void test_conv1d_backward_finite_diff(void) {
    /* Strided, padded, multi-channel. */
    size_t in_ch = 2, out_ch = 3, len = 7, k = 3;
    size_t wc = out_ch * in_ch * k;
    float *weight = malloc(wc * sizeof(float));
    float *bias = malloc(out_ch * sizeof(float));
    CHECK(weight != NULL && bias != NULL);
    unsigned seed = 99u;
    fill_lcg(weight, wc, &seed);
    fill_lcg(bias, out_ch, &seed);
    nir_conv1d_params_t params = {
        out_ch, in_ch, k, weight, bias, 2, 1, 1
    };
    size_t xshape[2] = {in_ch, len};
    nir_tensor_t input = nir_tensor_alloc(xshape, 2);
    CHECK(input.data != NULL);
    fill_lcg(input.data, input.size, &seed);
    nir_tensor_t probe = nir_op_conv1d(&params, &input);
    CHECK(probe.data != NULL);
    nir_tensor_t gy = nir_tensor_alloc(probe.shape, probe.ndim);
    CHECK(gy.data != NULL);
    fill_lcg(gy.data, gy.size, &seed);
    size_t wshape[1] = {wc};
    size_t bshape[1] = {out_ch};
    nir_tensor_t grad_w = nir_tensor_alloc(wshape, 1);
    nir_tensor_t grad_b = nir_tensor_alloc(bshape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(xshape, 2);
    CHECK(nir_op_conv1d_backward(
        &params, &input, &gy, &grad_w, &grad_b, &grad_x
    ));
    double eps = 0.01;
    double tol = 0.01;
    for (size_t i = 0; i < input.size; i++) {
        double num = conv1d_fd_elem(
            &params, &input, &gy, &input.data[i], eps
        );
        CHECK_NEAR(grad_x.data[i], num, tol);
    }
    for (size_t i = 0; i < wc; i++) {
        double num = conv1d_fd_elem(
            &params, &input, &gy, &weight[i], eps
        );
        CHECK_NEAR(grad_w.data[i], num, tol);
    }
    for (size_t i = 0; i < out_ch; i++) {
        double num = conv1d_fd_elem(
            &params, &input, &gy, &bias[i], eps
        );
        CHECK_NEAR(grad_b.data[i], num, tol);
    }
    nir_tensor_free(&input);
    nir_tensor_free(&probe);
    nir_tensor_free(&gy);
    nir_tensor_free(&grad_w);
    nir_tensor_free(&grad_b);
    nir_tensor_free(&grad_x);
    free(weight);
    free(bias);
}

static void test_flatten_backward(void) {
    float xv[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    size_t xshape[3] = {2, 1, 2};
    nir_tensor_t input = make_tensor(xv, xshape, 3);
    float gv[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    size_t gshape[1] = {4};
    nir_tensor_t grad_out = make_tensor(gv, gshape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(gshape, 1);
    CHECK(nir_op_flatten_backward(&input, &grad_out, &grad_x));
    CHECK(grad_x.ndim == 3 && grad_x.shape[0] == 2 &&
        grad_x.shape[1] == 1 && grad_x.shape[2] == 2);
    double want[] = {10.0, 20.0, 30.0, 40.0};
    check_data(&grad_x, want, 4, 1e-9);
    /* Size and rank mismatches are rejected. */
    size_t short_shape[1] = {3};
    nir_tensor_t short_gy = nir_tensor_alloc(short_shape, 1);
    CHECK(!nir_op_flatten_backward(&input, &short_gy, &grad_x));
    size_t rank2[2] = {2, 2};
    nir_tensor_t rank2_gy = nir_tensor_alloc(rank2, 2);
    CHECK(!nir_op_flatten_backward(&input, &rank2_gy, &grad_x));
    CHECK(!nir_op_flatten_backward(NULL, &grad_out, &grad_x));
    nir_tensor_free(&input);
    nir_tensor_free(&grad_out);
    nir_tensor_free(&grad_x);
    nir_tensor_free(&short_gy);
    nir_tensor_free(&rank2_gy);
}

static void test_pool_backward(void) {
    /* 3x3 input 1..9, 2x2 window, stride 1:
     * gy = [[1,2],[3,4]] scatters over four windows. */
    float xv[9];
    for (size_t i = 0; i < 9; i++) xv[i] = (float)(i + 1);
    size_t xshape[3] = {1, 3, 3};
    nir_tensor_t input = make_tensor(xv, xshape, 3);
    float gv[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t gshape[3] = {1, 2, 2};
    nir_tensor_t grad_out = make_tensor(gv, gshape, 3);
    nir_pool2d_params_t params = {2, 2, 1, 1, 0, 0};
    nir_tensor_t grad_avg = nir_tensor_alloc(xshape, 3);
    nir_tensor_t grad_sum = nir_tensor_alloc(xshape, 3);
    /* Prefill with junk: grad_input is overwritten, not added. */
    memset(grad_avg.data, 0x7f, 9 * sizeof(float));
    CHECK(nir_op_avgpool2d_backward(
        &params, &input, &grad_out, &grad_avg
    ));
    CHECK(nir_op_sumpool2d_backward(
        &params, &input, &grad_out, &grad_sum
    ));
    double want_avg[] = {0.25, 0.75, 0.5, 1.0, 2.5, 1.5, 0.75,
        1.75, 1.0};
    double want_sum[] = {1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0,
        7.0, 4.0};
    check_data(&grad_avg, want_avg, 9, 1e-6);
    check_data(&grad_sum, want_sum, 9, 1e-6);
    /* Stride 2, pad 1: padded cells receive nothing and the avg
     * divisor stays the full window area (count-include-pad). */
    nir_pool2d_params_t padded = {2, 2, 2, 2, 1, 1};
    nir_tensor_t grad_pad = nir_tensor_alloc(xshape, 3);
    CHECK(nir_op_avgpool2d_backward(
        &padded, &input, &grad_out, &grad_pad
    ));
    double want_pad[] = {0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 0.75,
        1.0, 1.0};
    check_data(&grad_pad, want_pad, 9, 1e-6);
    /* Bad geometry and mismatched grads are rejected. */
    nir_pool2d_params_t bad = {0, 2, 1, 1, 0, 0};
    CHECK(!nir_op_avgpool2d_backward(
        &bad, &input, &grad_out, &grad_pad
    ));
    CHECK(!nir_op_sumpool2d_backward(
        NULL, &input, &grad_out, &grad_pad
    ));
    size_t wrong[3] = {1, 3, 3};
    nir_tensor_t wrong_gy = nir_tensor_alloc(wrong, 3);
    CHECK(!nir_op_avgpool2d_backward(
        &params, &input, &wrong_gy, &grad_pad
    ));
    nir_tensor_free(&input);
    nir_tensor_free(&grad_out);
    nir_tensor_free(&grad_avg);
    nir_tensor_free(&grad_sum);
    nir_tensor_free(&grad_pad);
    nir_tensor_free(&wrong_gy);
}

static void test_scale_backward(void) {
    nir_scale_params_t params = {-2.0f};
    float xv[2] = {3.0f, 0.5f};
    float gv[2] = {1.0f, -4.0f};
    size_t shape[1] = {2};
    nir_tensor_t input = make_tensor(xv, shape, 1);
    nir_tensor_t grad_out = make_tensor(gv, shape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(shape, 1);
    float grad_scale = 0.0f;
    CHECK(nir_op_scale_backward(
        &params, &input, &grad_out, &grad_x, &grad_scale
    ));
    double want_x[] = {-2.0, 8.0};
    check_data(&grad_x, want_x, 2, 1e-9);
    /* 1*3 + (-4)*0.5 = 1, accumulating into the slot. */
    CHECK_NEAR(grad_scale, 1.0, 1e-6);
    CHECK(nir_op_scale_backward(
        &params, &input, &grad_out, &grad_x, &grad_scale
    ));
    CHECK_NEAR(grad_scale, 2.0, 1e-6);
    /* The scale slot is opt-in: without it no input is needed. */
    CHECK(nir_op_scale_backward(
        &params, NULL, &grad_out, &grad_x, NULL
    ));
    check_data(&grad_x, want_x, 2, 1e-9);
    /* Exact aliasing follows the elementwise rule. */
    memcpy(grad_out.data, gv, sizeof(gv));
    CHECK(nir_op_scale_backward(
        &params, NULL, &grad_out, &grad_out, NULL
    ));
    check_data(&grad_out, want_x, 2, 1e-9);
    /* A requested scale slot needs a matching input. */
    memcpy(grad_out.data, gv, sizeof(gv));
    CHECK(!nir_op_scale_backward(
        &params, NULL, &grad_out, &grad_x, &grad_scale
    ));
    size_t one[1] = {1};
    nir_tensor_t short_x = nir_tensor_alloc(one, 1);
    CHECK(!nir_op_scale_backward(
        &params, &short_x, &grad_out, &grad_x, &grad_scale
    ));
    nir_scale_params_t bad = {(float)INFINITY};
    CHECK(!nir_op_scale_backward(
        &bad, &input, &grad_out, &grad_x, NULL
    ));
    nir_tensor_free(&input);
    nir_tensor_free(&grad_out);
    nir_tensor_free(&grad_x);
    nir_tensor_free(&short_x);
}

static void check_lif_case(
    float x, const float *vprev, float gy_spike, int want_spike
) {
    nir_lif_params_t params = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    nir_tensor_t v_prev = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_stored = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_p = nir_tensor_alloc(shape, 1);
    nir_tensor_t spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t stored = nir_tensor_alloc(shape, 1);
    input.data[0] = x;
    if (vprev != NULL) {
        v_prev.data[0] = *vprev;
    }
    grad_spike.data[0] = gy_spike;
    /* A nonzero stored gradient must be ignored: the reset
     * branch carries zero gradient. */
    grad_stored.data[0] = 123.0f;
    /* Pin the branch label with the forward op. */
    CHECK(nir_op_lif_into(
        &params, &input, vprev != NULL ? &v_prev : NULL, &spike,
        &stored
    ));
    CHECK_NEAR(spike.data[0], (double)want_spike, 1e-9);
    CHECK(nir_op_lif_backward(
        &params, &input, vprev != NULL ? &v_prev : NULL,
        &grad_spike, &grad_stored, NIR_SURROGATE_FAST_SIGMOID,
        25.0f, &grad_x, &grad_p
    ));
    double decay = exp(-1.0);
    double prev = vprev != NULL ? (double)*vprev : 0.0;
    double membrane = prev * decay + (1.0 - decay) * (double)x;
    double g = (double)gy_spike *
        fast_sigmoid_grad(25.0, membrane - 1.0);
    CHECK_NEAR(grad_x.data[0], g * (1.0 - decay), 1e-6);
    CHECK_NEAR(grad_p.data[0], g * decay, 1e-6);
    /* NULL stored grad and NULL prev grad agree. */
    nir_tensor_t grad_x2 = nir_tensor_alloc(shape, 1);
    CHECK(nir_op_lif_backward(
        &params, &input, vprev != NULL ? &v_prev : NULL,
        &grad_spike, NULL, NIR_SURROGATE_FAST_SIGMOID, 25.0f,
        &grad_x2, NULL
    ));
    CHECK_NEAR(grad_x2.data[0], grad_x.data[0], 1e-9);
    nir_tensor_free(&input);
    nir_tensor_free(&v_prev);
    nir_tensor_free(&grad_spike);
    nir_tensor_free(&grad_stored);
    nir_tensor_free(&grad_x);
    nir_tensor_free(&grad_p);
    nir_tensor_free(&grad_x2);
    nir_tensor_free(&spike);
    nir_tensor_free(&stored);
}

static void test_lif_backward(void) {
    /* Below threshold: no spike. Above: spike with reset. */
    check_lif_case(0.5f, NULL, 2.0f, 0);
    float rest = 0.0f;
    check_lif_case(0.5f, &rest, 2.0f, 0);
    check_lif_case(2.0f, NULL, -1.5f, 1);
    /* Atan surrogate on the no-spike branch. */
    nir_lif_params_t params = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_p = nir_tensor_alloc(shape, 1);
    input.data[0] = 0.5f;
    grad_spike.data[0] = 1.0f;
    CHECK(nir_op_lif_backward(
        &params, &input, NULL, &grad_spike, NULL,
        NIR_SURROGATE_ATAN, 2.0f, &grad_x, &grad_p
    ));
    double decay = exp(-1.0);
    double membrane = (1.0 - decay) * 0.5;
    double g = atan_grad(2.0, membrane - 1.0);
    CHECK_NEAR(grad_x.data[0], g * (1.0 - decay), 1e-6);
    CHECK_NEAR(grad_p.data[0], g * decay, 1e-6);
    /* Rejections: bad surrogate, bad params, bad wiring. */
    CHECK(!nir_op_lif_backward(
        &params, &input, NULL, &grad_spike, NULL,
        NIR_SURROGATE_ATAN, 0.0f, &grad_x, &grad_p
    ));
    CHECK(!nir_op_lif_backward(
        &params, &input, NULL, &grad_spike, NULL,
        (nir_surrogate_kind_t)99, 2.0f, &grad_x, &grad_p
    ));
    nir_lif_params_t bad_tau = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    CHECK(!nir_op_lif_backward(
        &bad_tau, &input, NULL, &grad_spike, NULL,
        NIR_SURROGATE_ATAN, 2.0f, &grad_x, &grad_p
    ));
    nir_lif_params_t bad_r = {
        1.0f, (float)NAN, 0.0f, 1.0f, 0.0f
    };
    CHECK(!nir_op_lif_backward(
        &bad_r, &input, NULL, &grad_spike, NULL,
        NIR_SURROGATE_ATAN, 2.0f, &grad_x, &grad_p
    ));
    CHECK(!nir_op_lif_backward(
        NULL, &input, NULL, &grad_spike, NULL,
        NIR_SURROGATE_ATAN, 2.0f, &grad_x, &grad_p
    ));
    /* The two outputs must not overlap each other. */
    CHECK(!nir_op_lif_backward(
        &params, &input, NULL, &grad_spike, NULL,
        NIR_SURROGATE_ATAN, 2.0f, &grad_x, &grad_x
    ));
    /* A misshapen stored gradient is rejected, not ignored. */
    size_t two[1] = {2};
    nir_tensor_t wrong_stored = nir_tensor_alloc(two, 1);
    CHECK(!nir_op_lif_backward(
        &params, &input, NULL, &grad_spike, &wrong_stored,
        NIR_SURROGATE_ATAN, 2.0f, &grad_x, &grad_p
    ));
    nir_tensor_free(&input);
    nir_tensor_free(&grad_spike);
    nir_tensor_free(&grad_x);
    nir_tensor_free(&grad_p);
    nir_tensor_free(&wrong_stored);
}

static void check_if_case(
    float x, float vprev, int use_prev, float gy_spike,
    int want_spike
) {
    nir_if_params_t params = {2.0f, 2.0f, 0.5f};
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    nir_tensor_t v_prev = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_stored = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_p = nir_tensor_alloc(shape, 1);
    nir_tensor_t spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t stored = nir_tensor_alloc(shape, 1);
    input.data[0] = x;
    v_prev.data[0] = vprev;
    grad_spike.data[0] = gy_spike;
    grad_stored.data[0] = -7.0f;
    const nir_tensor_t *prev = use_prev ? &v_prev : NULL;
    CHECK(nir_op_if_into(
        &params, &input, prev, &spike, &stored
    ));
    CHECK_NEAR(spike.data[0], (double)want_spike, 1e-9);
    CHECK(nir_op_if_backward(
        &params, &input, prev, &grad_spike, &grad_stored,
        NIR_SURROGATE_FAST_SIGMOID, 25.0f, &grad_x, &grad_p
    ));
    double have_prev = use_prev ? (double)vprev : 0.0;
    double membrane = have_prev + 2.0 * (double)x;
    double g = (double)gy_spike *
        fast_sigmoid_grad(25.0, membrane - 2.0);
    CHECK_NEAR(grad_x.data[0], g * 2.0, 1e-6);
    CHECK_NEAR(grad_p.data[0], g, 1e-6);
    nir_tensor_free(&input);
    nir_tensor_free(&v_prev);
    nir_tensor_free(&grad_spike);
    nir_tensor_free(&grad_stored);
    nir_tensor_free(&grad_x);
    nir_tensor_free(&grad_p);
    nir_tensor_free(&spike);
    nir_tensor_free(&stored);
}

static void test_if_backward(void) {
    /* v_prev + 2x = 2 hits the threshold exactly: strict, so
     * no spike. 2 + 2*0.5 = 3 crosses. */
    check_if_case(1.0f, 0.0f, 0, 1.5f, 0);
    check_if_case(0.5f, 2.0f, 1, -1.0f, 1);
    /* Atan surrogate with NULL prev grad. */
    nir_if_params_t params = {2.0f, 2.0f, 0.5f};
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(shape, 1);
    input.data[0] = 1.0f;
    grad_spike.data[0] = 1.0f;
    CHECK(nir_op_if_backward(
        &params, &input, NULL, &grad_spike, NULL,
        NIR_SURROGATE_ATAN, 2.0f, &grad_x, NULL
    ));
    double g = atan_grad(2.0, 0.0);
    CHECK_NEAR(grad_x.data[0], g * 2.0, 1e-6);
    /* Rejections mirror lif_backward. */
    nir_if_params_t bad_r = {
        (float)INFINITY, 2.0f, 0.5f
    };
    CHECK(!nir_op_if_backward(
        &bad_r, &input, NULL, &grad_spike, NULL,
        NIR_SURROGATE_ATAN, 2.0f, &grad_x, NULL
    ));
    CHECK(!nir_op_if_backward(
        &params, NULL, NULL, &grad_spike, NULL,
        NIR_SURROGATE_ATAN, 2.0f, &grad_x, NULL
    ));
    nir_tensor_free(&input);
    nir_tensor_free(&grad_spike);
    nir_tensor_free(&grad_x);
}

static void check_cuba_case(
    float x, const float *syn_prev, const float *mem_prev,
    float gy_spike, int want_spike
) {
    nir_cuba_params_t params = {
        1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.5f
    };
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    nir_tensor_t syn = nir_tensor_alloc(shape, 1);
    nir_tensor_t mem = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_stored = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_s = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_m = nir_tensor_alloc(shape, 1);
    nir_tensor_t spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t syn_out = nir_tensor_alloc(shape, 1);
    nir_tensor_t stored = nir_tensor_alloc(shape, 1);
    input.data[0] = x;
    if (syn_prev != NULL) {
        syn.data[0] = *syn_prev;
        mem.data[0] = *mem_prev;
    }
    grad_spike.data[0] = gy_spike;
    grad_stored.data[0] = 5.0f;
    const nir_tensor_t *sp = syn_prev != NULL ? &syn : NULL;
    const nir_tensor_t *mp = syn_prev != NULL ? &mem : NULL;
    CHECK(nir_op_cuba_into(
        &params, &input, sp, mp, &spike, &syn_out, &stored
    ));
    CHECK_NEAR(spike.data[0], (double)want_spike, 1e-9);
    CHECK(nir_op_cuba_backward(
        &params, &input, sp, mp, &grad_spike, &grad_stored,
        NIR_SURROGATE_FAST_SIGMOID, 25.0f, &grad_x, &grad_s,
        &grad_m
    ));
    double syn_decay = exp(-1.0);
    double mem_decay = exp(-0.5);
    double have_syn = syn_prev != NULL ? (double)*syn_prev : 0.0;
    double have_mem = syn_prev != NULL ? (double)*mem_prev : 0.0;
    double syn_v = have_syn * syn_decay +
        0.5 * (1.0 - syn_decay) * (double)x;
    double membrane = have_mem * mem_decay +
        (1.0 - mem_decay) * syn_v;
    double g = (double)gy_spike *
        fast_sigmoid_grad(25.0, membrane - 1.0);
    double dsyn = g * (1.0 - mem_decay);
    CHECK_NEAR(grad_x.data[0], dsyn * 0.5 * (1.0 - syn_decay),
        1e-6);
    CHECK_NEAR(grad_s.data[0], dsyn * syn_decay, 1e-6);
    CHECK_NEAR(grad_m.data[0], g * mem_decay, 1e-6);
    /* Either state gradient may be omitted. */
    CHECK(nir_op_cuba_backward(
        &params, &input, sp, mp, &grad_spike, NULL,
        NIR_SURROGATE_FAST_SIGMOID, 25.0f, &grad_x, NULL, NULL
    ));
    CHECK_NEAR(grad_x.data[0], dsyn * 0.5 * (1.0 - syn_decay),
        1e-6);
    nir_tensor_free(&input);
    nir_tensor_free(&syn);
    nir_tensor_free(&mem);
    nir_tensor_free(&grad_spike);
    nir_tensor_free(&grad_stored);
    nir_tensor_free(&grad_x);
    nir_tensor_free(&grad_s);
    nir_tensor_free(&grad_m);
    nir_tensor_free(&spike);
    nir_tensor_free(&syn_out);
    nir_tensor_free(&stored);
}

static void test_cuba_backward(void) {
    /* From rest: x = 1 stays quiet, x = 10 spikes. */
    check_cuba_case(1.0f, NULL, NULL, 1.0f, 0);
    check_cuba_case(10.0f, NULL, NULL, -2.0f, 1);
    /* Primed dual state on the no-spike branch. */
    float syn_prev = 0.25f;
    float mem_prev = 0.1f;
    check_cuba_case(1.0f, &syn_prev, &mem_prev, 1.0f, 0);
    /* Split or misshapen state is rejected, like the forward. */
    nir_cuba_params_t params = {
        1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.5f
    };
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    nir_tensor_t syn = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(shape, 1);
    CHECK(!nir_op_cuba_backward(
        &params, &input, &syn, NULL, &grad_spike, NULL,
        NIR_SURROGATE_FAST_SIGMOID, 25.0f, &grad_x, NULL, NULL
    ));
    size_t two[1] = {2};
    nir_tensor_t wide = nir_tensor_alloc(two, 1);
    CHECK(!nir_op_cuba_backward(
        &params, &input, &wide, &wide, &grad_spike, NULL,
        NIR_SURROGATE_FAST_SIGMOID, 25.0f, &grad_x, NULL, NULL
    ));
    nir_cuba_params_t bad_w = {
        1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 0.0f, (float)NAN
    };
    CHECK(!nir_op_cuba_backward(
        &bad_w, &input, NULL, NULL, &grad_spike, NULL,
        NIR_SURROGATE_FAST_SIGMOID, 25.0f, &grad_x, NULL, NULL
    ));
    nir_tensor_free(&input);
    nir_tensor_free(&syn);
    nir_tensor_free(&grad_spike);
    nir_tensor_free(&grad_x);
    nir_tensor_free(&wide);
}

static void test_backward_rejects_bad_wiring(void) {
    float weight[1] = {1.0f};
    float bias[1] = {0.0f};
    nir_conv2d_params_t conv = {
        1, 1, 1, 1, weight, bias, 1, 1, 0, 0, 1, 1
    };
    size_t xshape[3] = {1, 2, 2};
    size_t wshape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(xshape, 3);
    nir_tensor_t grad_out = nir_tensor_alloc(xshape, 3);
    nir_tensor_t grad_w = nir_tensor_alloc(wshape, 1);
    nir_tensor_t grad_b = nir_tensor_alloc(wshape, 1);
    nir_tensor_t grad_x = nir_tensor_alloc(xshape, 3);
    /* NULL slots are rejected. */
    CHECK(!nir_op_conv2d_backward(
        NULL, &input, &grad_out, &grad_w, &grad_b, &grad_x
    ));
    CHECK(!nir_op_conv2d_backward(
        &conv, &input, &grad_out, NULL, &grad_b, &grad_x
    ));
    CHECK(!nir_op_conv2d_backward(
        &conv, &input, &grad_out, &grad_w, &grad_b, NULL
    ));
    /* A mis-shaped output gradient is rejected. */
    size_t wrong[3] = {1, 3, 3};
    nir_tensor_t wrong_gy = nir_tensor_alloc(wrong, 3);
    CHECK(!nir_op_conv2d_backward(
        &conv, &input, &wrong_gy, &grad_w, &grad_b, &grad_x
    ));
    /* Spatial outputs must be disjoint from every input. */
    nir_tensor_t aliased = {0};
    CHECK(nir_tensor_bind(
        &aliased, input.data, input.capacity, xshape, 3
    ));
    CHECK(!nir_op_conv2d_backward(
        &conv, &input, &grad_out, &grad_w, &grad_b, &aliased
    ));
    /* Sibling outputs must not overlap each other. */
    float buf[8] = {0};
    nir_tensor_t gw_bind = {0};
    nir_tensor_t gx_bind = {0};
    CHECK(nir_tensor_bind(&gw_bind, buf, 8, wshape, 1));
    CHECK(nir_tensor_bind(&gx_bind, buf, 8, xshape, 3));
    CHECK(!nir_op_conv2d_backward(
        &conv, &input, &grad_out, &gw_bind, &grad_b, &gx_bind
    ));
    nir_tensor_free(&input);
    nir_tensor_free(&grad_out);
    nir_tensor_free(&grad_w);
    nir_tensor_free(&grad_b);
    nir_tensor_free(&grad_x);
    nir_tensor_free(&wrong_gy);
    nir_tensor_free(&aliased);
    nir_tensor_free(&gw_bind);
    nir_tensor_free(&gx_bind);
}

int main(void) {
    test_conv2d_backward_hand();
    test_conv2d_backward_finite_diff();
    test_conv1d_backward_hand();
    test_conv1d_backward_finite_diff();
    test_flatten_backward();
    test_pool_backward();
    test_scale_backward();
    test_lif_backward();
    test_if_backward();
    test_cuba_backward();
    test_backward_rejects_bad_wiring();
    return 0;
}
