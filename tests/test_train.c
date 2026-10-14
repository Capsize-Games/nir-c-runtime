/* C-side training math: surrogate hook, layer backward ops,
 * cross-entropy loss, SGD/Adam optimizers, and the STDP port.
 *
 * Reference values are float64 closed-form evaluations of the
 * documented update rules (snnTorch fast_sigmoid/atan gradients,
 * torch.optim.SGD/Adam, Song/Abbott STDP, softmax cross-entropy),
 * kept to tolerances that separate float32 rounding from formula
 * errors. A committed torch-fixture generator under tools/ is
 * follow-up work for the tools owner.
 */
#include <math.h>
#include <string.h>

#include "nir_c_runtime/ops.h"
#include "nir_c_runtime/train.h"
#include "test_util.h"

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

static void test_surrogate_names(void) {
    nir_surrogate_kind_t kind;
    char *error = NULL;
    CHECK(nir_surrogate_from_name("fast_sigmoid", &kind, &error));
    CHECK(error == NULL && kind == NIR_SURROGATE_FAST_SIGMOID);
    CHECK(nir_surrogate_from_name("atan", &kind, &error));
    CHECK(error == NULL && kind == NIR_SURROGATE_ATAN);
    CHECK(!nir_surrogate_from_name("bogus", &kind, &error));
    CHECK(error != NULL && strstr(error, "bogus") != NULL);
    free(error);
    error = NULL;
    CHECK(!nir_surrogate_from_name(NULL, &kind, &error));
    CHECK(error != NULL);
    free(error);
    CHECK(!nir_surrogate_from_name("atan", NULL, NULL));
}

static void test_surrogate_values(void) {
    const struct {
        nir_surrogate_kind_t kind;
        float slope;
        float x;
        double want;
    } cases[] = {
        {NIR_SURROGATE_FAST_SIGMOID, 25.0f, -1.0f, 0.03698224852071006},
        {NIR_SURROGATE_FAST_SIGMOID, 25.0f, -0.1f, 2.0408163265306123},
        {NIR_SURROGATE_FAST_SIGMOID, 25.0f, 0.0f, 25.0},
        {NIR_SURROGATE_FAST_SIGMOID, 25.0f, 0.1f, 2.0408163265306123},
        {NIR_SURROGATE_FAST_SIGMOID, 25.0f, 1.0f, 0.03698224852071006},
        {NIR_SURROGATE_ATAN, 2.0f, -1.0f, 0.09199966835037524},
        {NIR_SURROGATE_ATAN, 2.0f, -0.1f, 0.9101698376462755},
        {NIR_SURROGATE_ATAN, 2.0f, 0.0f, 1.0},
        {NIR_SURROGATE_ATAN, 2.0f, 0.1f, 0.9101698376462755},
        {NIR_SURROGATE_ATAN, 2.0f, 1.0f, 0.09199966835037524},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        float got = 0.0f;
        CHECK(nir_surrogate_grad(
            cases[i].kind, cases[i].slope, cases[i].x, &got
        ));
        CHECK_NEAR(got, cases[i].want, 1e-6);
    }
    float out = 0.0f;
    CHECK(!nir_surrogate_grad(NIR_SURROGATE_ATAN, 0.0f, 0.0f, &out));
    CHECK(!nir_surrogate_grad(NIR_SURROGATE_ATAN, -2.0f, 0.0f, &out));
    CHECK(!nir_surrogate_grad(
        NIR_SURROGATE_ATAN, (float)INFINITY, 0.0f, &out
    ));
    CHECK(!nir_surrogate_grad(
        NIR_SURROGATE_ATAN, (float)NAN, 0.0f, &out
    ));
    CHECK(!nir_surrogate_grad(
        NIR_SURROGATE_ATAN, 2.0f, (float)NAN, &out
    ));
    CHECK(!nir_surrogate_grad((nir_surrogate_kind_t)99, 2.0f, 0.0f, &out));
    CHECK(!nir_surrogate_grad(NIR_SURROGATE_ATAN, 2.0f, 0.0f, NULL));
}

static void test_threshold_backward(void) {
    nir_threshold_params_t params = {0.5f};
    float m[] = {0.2f, 0.5f, 0.8f};
    float g[] = {1.0f, 1.0f, 1.0f};
    nir_tensor_t membrane = make_vector(m, 3);
    nir_tensor_t grad_out = make_vector(g, 3);
    size_t shape[1] = {3};
    nir_tensor_t grad_in = nir_tensor_alloc(shape, 1);
    CHECK(grad_in.data != NULL);
    CHECK(nir_op_threshold_backward(
        &params, &membrane, &grad_out, NIR_SURROGATE_FAST_SIGMOID,
        25.0f, &grad_in
    ));
    double want[] = {0.3460207612456747, 25.0, 0.3460207612456747};
    check_vector(&grad_in, want, 3, 1e-6);
    /* Exact aliasing follows the elementwise rule. */
    memcpy(grad_out.data, g, sizeof(g));
    CHECK(nir_op_threshold_backward(
        &params, &membrane, &grad_out, NIR_SURROGATE_FAST_SIGMOID,
        25.0f, &grad_out
    ));
    check_vector(&grad_out, want, 3, 1e-6);
    CHECK(!nir_op_threshold_backward(
        &params, &membrane, &grad_out, NIR_SURROGATE_FAST_SIGMOID,
        0.0f, &grad_in
    ));
    nir_tensor_free(&membrane);
    nir_tensor_free(&grad_out);
    nir_tensor_free(&grad_in);
}

static void test_li_backward(void) {
    nir_li_params_t params = {2.0f, 1.0f, 0.0f};
    float g[] = {1.0f, -2.0f};
    nir_tensor_t grad_out = make_vector(g, 2);
    size_t shape[1] = {2};
    nir_tensor_t grad_in = nir_tensor_alloc(shape, 1);
    nir_tensor_t grad_prev = nir_tensor_alloc(shape, 1);
    CHECK(grad_in.data != NULL && grad_prev.data != NULL);
    CHECK(nir_op_li_backward(&params, &grad_out, &grad_in, &grad_prev));
    double want_in[] = {0.3934693402873666, -0.7869386805747332};
    double want_prev[] = {0.6065306597126334, -1.2130613194252668};
    check_vector(&grad_in, want_in, 2, 1e-6);
    check_vector(&grad_prev, want_prev, 2, 1e-6);
    CHECK(nir_op_li_backward(&params, &grad_out, &grad_in, NULL));
    check_vector(&grad_in, want_in, 2, 1e-6);
    nir_li_params_t bad = {0.0f, 1.0f, 0.0f};
    CHECK(!nir_op_li_backward(&bad, &grad_out, &grad_in, NULL));
    nir_tensor_free(&grad_out);
    nir_tensor_free(&grad_in);
    nir_tensor_free(&grad_prev);
}

static void test_affine_backward(void) {
    float w[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {0.5f, -0.5f};
    nir_affine_params_t params = {2, 2, w, b};
    float xv[] = {0.5f, -1.0f};
    float gv[] = {1.0f, -0.5f};
    nir_tensor_t input = make_vector(xv, 2);
    nir_tensor_t grad_out = make_vector(gv, 2);
    size_t wshape[2] = {2, 2};
    size_t vshape[1] = {2};
    nir_tensor_t grad_w = nir_tensor_alloc(wshape, 2);
    nir_tensor_t grad_b = nir_tensor_alloc(vshape, 1);
    nir_tensor_t grad_in = nir_tensor_alloc(vshape, 1);
    CHECK(grad_w.data != NULL && grad_b.data != NULL);
    CHECK(grad_in.data != NULL);
    CHECK(nir_op_affine_backward(
        &params, &input, &grad_out, &grad_w, &grad_b, &grad_in
    ));
    double want_w[] = {0.5, -1.0, -0.25, 0.5};
    double want_b[] = {1.0, -0.5};
    double want_x[] = {-0.5, 0.0};
    check_vector(&grad_w, want_w, 4, 1e-6);
    check_vector(&grad_b, want_b, 2, 1e-6);
    check_vector(&grad_in, want_x, 2, 1e-6);
    /* Weight/bias gradients accumulate across calls. */
    CHECK(nir_op_affine_backward(
        &params, &input, &grad_out, &grad_w, &grad_b, &grad_in
    ));
    double want_w2[] = {1.0, -2.0, -0.5, 1.0};
    double want_b2[] = {2.0, -1.0};
    check_vector(&grad_w, want_w2, 4, 1e-6);
    check_vector(&grad_b, want_b2, 2, 1e-6);
    /* A biased node requires a bias gradient slot. */
    CHECK(!nir_op_affine_backward(
        &params, &input, &grad_out, &grad_w, NULL, &grad_in
    ));
    params.bias = NULL;
    memset(grad_w.data, 0, 4 * sizeof(float));
    CHECK(nir_op_affine_backward(
        &params, &input, &grad_out, &grad_w, NULL, &grad_in
    ));
    check_vector(&grad_w, want_w, 4, 1e-6);
    nir_tensor_free(&input);
    nir_tensor_free(&grad_out);
    nir_tensor_free(&grad_w);
    nir_tensor_free(&grad_b);
    nir_tensor_free(&grad_in);
}

static void test_linear_backward(void) {
    float w[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    nir_linear_params_t params = {2, 3, w};
    float xv[] = {1.0f, 0.0f, -1.0f};
    float gv[] = {2.0f, -1.0f};
    nir_tensor_t input = make_vector(xv, 3);
    nir_tensor_t grad_out = make_vector(gv, 2);
    size_t wshape[2] = {2, 3};
    size_t xshape[1] = {3};
    nir_tensor_t grad_w = nir_tensor_alloc(wshape, 2);
    nir_tensor_t grad_in = nir_tensor_alloc(xshape, 1);
    CHECK(grad_w.data != NULL && grad_in.data != NULL);
    CHECK(nir_op_linear_backward(
        &params, &input, &grad_out, &grad_w, &grad_in
    ));
    double want_w[] = {2.0, 0.0, -2.0, -1.0, 0.0, 1.0};
    double want_x[] = {-2.0, -1.0, 0.0};
    check_vector(&grad_w, want_w, 6, 1e-6);
    check_vector(&grad_in, want_x, 3, 1e-6);
    nir_tensor_free(&input);
    nir_tensor_free(&grad_out);
    nir_tensor_free(&grad_w);
    nir_tensor_free(&grad_in);
}

static void test_cross_entropy(void) {
    float lv[] = {1.0f, 2.0f, 3.0f};
    nir_tensor_t logits = make_vector(lv, 3);
    size_t shape[1] = {3};
    nir_tensor_t grad = nir_tensor_alloc(shape, 1);
    CHECK(grad.data != NULL);
    float loss = 0.0f;
    CHECK(nir_cross_entropy(&logits, 2, &loss, &grad));
    CHECK_NEAR(loss, 0.40760596444438046, 1e-6);
    double want[] = {0.09003057317038046, 0.24472847105479764,
        -0.3347590442251782};
    check_vector(&grad, want, 3, 1e-6);
    CHECK(!nir_cross_entropy(&logits, 3, &loss, &grad));
    nir_tensor_free(&logits);
    nir_tensor_free(&grad);
}

static void test_sgd(void) {
    nir_sgd_t opt;
    char *error = NULL;
    CHECK(nir_sgd_init(&opt, 2, 0.1f, 0.9f, 0.01f, &error));
    CHECK(error == NULL);
    float w[] = {1.0f, -2.0f};
    float g[] = {0.1f, 0.2f};
    CHECK(nir_sgd_step(&opt, w, g, 2));
    CHECK_NEAR(w[0], 0.989, 1e-6);
    CHECK_NEAR(w[1], -2.018, 1e-6);
    CHECK(nir_sgd_step(&opt, w, g, 2));
    CHECK_NEAR(w[0], 0.9681109999999999, 1e-6);
    CHECK_NEAR(w[1], -2.0521819999999997, 1e-6);
    CHECK(!nir_sgd_step(&opt, w, g, 1));
    nir_sgd_free(&opt);
    CHECK(nir_sgd_init(&opt, 1, 0.5f, 0.0f, 0.0f, &error));
    float w0[] = {2.0f};
    float g0[] = {1.0f};
    CHECK(nir_sgd_step(&opt, w0, g0, 1));
    CHECK_NEAR(w0[0], 1.5, 1e-6);
    nir_sgd_free(&opt);
    CHECK(!nir_sgd_init(&opt, 2, -0.1f, 0.0f, 0.0f, &error));
    CHECK(error != NULL);
    free(error);
}

static void test_adam(void) {
    nir_adam_t opt;
    char *error = NULL;
    CHECK(nir_adam_init(
        &opt, 2, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    CHECK(error == NULL);
    float w[] = {1.0f, -2.0f};
    float g[] = {0.1f, 0.2f};
    CHECK(nir_adam_step(&opt, w, g, 2));
    CHECK_NEAR(w[0], 0.9990000001, 1e-6);
    CHECK_NEAR(w[1], -2.00099999995, 1e-6);
    CHECK(nir_adam_step(&opt, w, g, 2));
    CHECK_NEAR(w[0], 0.9980000002, 1e-6);
    CHECK_NEAR(w[1], -2.0019999998999998, 1e-6);
    nir_adam_free(&opt);
    CHECK(nir_adam_init(
        &opt, 2, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, &error
    ));
    float w2[] = {1.0f, -2.0f};
    CHECK(nir_adam_step(&opt, w2, g, 2));
    CHECK_NEAR(w2[0], 0.9990000000909091, 1e-6);
    CHECK_NEAR(w2[1], -2.0009999999444443, 1e-6);
    nir_adam_free(&opt);
    CHECK(!nir_adam_init(
        &opt, 2, 0.001f, 1.0f, 0.999f, 1e-8f, 0.0f, &error
    ));
    CHECK(error != NULL);
    free(error);
}

static void test_stdp(void) {
    nir_stdp_t s;
    char *error = NULL;
    CHECK(nir_stdp_init(
        &s, 2, 2, 2.0f, 2.0f, 0.01f, 0.012f, 0.0f, 1.0f, &error
    ));
    CHECK(error == NULL);
    float pre1[] = {1.0f, 0.0f};
    float post1[] = {0.0f, 1.0f};
    CHECK(nir_stdp_step(&s, pre1, post1));
    CHECK_NEAR(s.weight[0], 0.0, 1e-7);
    CHECK_NEAR(s.weight[3], 0.0, 1e-7);
    float pre2[] = {0.0f, 1.0f};
    float post2[] = {1.0f, 0.0f};
    CHECK(nir_stdp_step(&s, pre2, post2));
    /* w[0][0] potentiates; w[1][1]'s raw -0.00727... clamps to 0. */
    CHECK_NEAR(s.weight[0], 0.006065306597126334, 1e-7);
    CHECK_NEAR(s.weight[1], 0.0, 1e-7);
    CHECK_NEAR(s.weight[2], 0.0, 1e-7);
    CHECK_NEAR(s.weight[3], 0.0, 1e-7);
    CHECK_NEAR(s.pre_trace[0], 0.6065306597126334, 1e-6);
    CHECK_NEAR(s.pre_trace[1], 1.0, 1e-6);
    nir_stdp_free(&s);
    /* Upper clamp: the raw update overshoots w_max. */
    CHECK(nir_stdp_init(
        &s, 1, 1, 10.0f, 10.0f, 0.1f, 0.012f, 0.0f, 0.05f, &error
    ));
    float one[] = {1.0f};
    float zero[] = {0.0f};
    CHECK(nir_stdp_step(&s, one, zero));
    CHECK(nir_stdp_step(&s, zero, one));
    CHECK_NEAR(s.weight[0], 0.05, 1e-6);
    float seed[] = {0.02f};
    CHECK(nir_stdp_set_weight(&s, seed));
    CHECK_NEAR(s.weight[0], 0.02, 1e-7);
    nir_stdp_free(&s);
    CHECK(!nir_stdp_init(
        &s, 1, 1, 0.0f, 10.0f, 0.1f, 0.012f, 0.0f, 1.0f, &error
    ));
    CHECK(error != NULL);
    free(error);
}

/* A two-layer topology: Affine(2->2) + Threshold(0.5) +
 * Affine(2->2), trained single-step against one sample. */
typedef struct {
    float w1[4];
    float b1[2];
    float w2[4];
    float b2[2];
    nir_affine_params_t l1;
    nir_affine_params_t l2;
    nir_threshold_params_t thr;
} topo_t;

static void topo_init(topo_t *net) {
    float w1[] = {0.5f, -0.5f, 0.25f, 0.75f};
    float b1[] = {0.0f, 0.1f};
    float w2[] = {1.0f, -1.0f, -0.5f, 0.5f};
    float b2[] = {0.1f, -0.1f};
    memcpy(net->w1, w1, sizeof(w1));
    memcpy(net->b1, b1, sizeof(b1));
    memcpy(net->w2, w2, sizeof(w2));
    memcpy(net->b2, b2, sizeof(b2));
    net->l1.out_features = 2;
    net->l1.in_features = 2;
    net->l1.weight = net->w1;
    net->l1.bias = net->b1;
    net->l2.out_features = 2;
    net->l2.in_features = 2;
    net->l2.weight = net->w2;
    net->l2.bias = net->b2;
    net->thr.threshold = 0.5f;
}

typedef struct {
    nir_tensor_t h_pre;
    nir_tensor_t h;
    nir_tensor_t logits;
    nir_tensor_t dlogits;
    nir_tensor_t dh;
    nir_tensor_t dh_pre;
    nir_tensor_t dw1;
    nir_tensor_t db1;
    nir_tensor_t dw2;
    nir_tensor_t db2;
} topo_grad_t;

static void topo_alloc(topo_grad_t *g) {
    size_t v[1] = {2};
    size_t m[2] = {2, 2};
    g->h_pre = nir_tensor_alloc(v, 1);
    g->h = nir_tensor_alloc(v, 1);
    g->logits = nir_tensor_alloc(v, 1);
    g->dlogits = nir_tensor_alloc(v, 1);
    g->dh = nir_tensor_alloc(v, 1);
    g->dh_pre = nir_tensor_alloc(v, 1);
    g->dw1 = nir_tensor_alloc(m, 2);
    g->db1 = nir_tensor_alloc(v, 1);
    g->dw2 = nir_tensor_alloc(m, 2);
    g->db2 = nir_tensor_alloc(v, 1);
}

static void topo_free(topo_grad_t *g) {
    nir_tensor_free(&g->h_pre);
    nir_tensor_free(&g->h);
    nir_tensor_free(&g->logits);
    nir_tensor_free(&g->dlogits);
    nir_tensor_free(&g->dh);
    nir_tensor_free(&g->dh_pre);
    nir_tensor_free(&g->dw1);
    nir_tensor_free(&g->db1);
    nir_tensor_free(&g->dw2);
    nir_tensor_free(&g->db2);
}

static float topo_loss_and_grads(
    topo_t *net, topo_grad_t *g, const nir_tensor_t *input,
    size_t target
) {
    memset(g->dw1.data, 0, 4 * sizeof(float));
    memset(g->db1.data, 0, 2 * sizeof(float));
    memset(g->dw2.data, 0, 4 * sizeof(float));
    memset(g->db2.data, 0, 2 * sizeof(float));
    CHECK(nir_op_affine_into(&net->l1, input, &g->h_pre));
    CHECK(nir_op_threshold_into(&net->thr, &g->h_pre, &g->h));
    CHECK(nir_op_affine_into(&net->l2, &g->h, &g->logits));
    float loss = 0.0f;
    CHECK(nir_cross_entropy(&g->logits, target, &loss, &g->dlogits));
    CHECK(nir_op_affine_backward(
        &net->l2, &g->h, &g->dlogits, &g->dw2, &g->db2, &g->dh
    ));
    CHECK(nir_op_threshold_backward(
        &net->thr, &g->h_pre, &g->dh, NIR_SURROGATE_FAST_SIGMOID,
        25.0f, &g->dh_pre
    ));
    size_t v[1] = {2};
    nir_tensor_t dx = nir_tensor_alloc(v, 1);
    CHECK(dx.data != NULL);
    CHECK(nir_op_affine_backward(
        &net->l1, input, &g->dh_pre, &g->dw1, &g->db1, &dx
    ));
    nir_tensor_free(&dx);
    return loss;
}

static void test_topology_step(void) {
    topo_t net;
    topo_init(&net);
    topo_grad_t g;
    topo_alloc(&g);
    float xv[] = {0.5f, -0.25f};
    nir_tensor_t input = make_vector(xv, 2);
    float loss = topo_loss_and_grads(&net, &g, &input, 1);
    CHECK_NEAR(loss, 0.7981388693815918, 1e-5);
    char *error = NULL;
    nir_adam_t o_w1, o_b1, o_w2, o_b2;
    CHECK(nir_adam_init(&o_w1, 4, 0.01f, 0.9f, 0.999f, 1e-8f, 0, &error));
    CHECK(nir_adam_init(&o_b1, 2, 0.01f, 0.9f, 0.999f, 1e-8f, 0, &error));
    CHECK(nir_adam_init(&o_w2, 4, 0.01f, 0.9f, 0.999f, 1e-8f, 0, &error));
    CHECK(nir_adam_init(&o_b2, 2, 0.01f, 0.9f, 0.999f, 1e-8f, 0, &error));
    CHECK(nir_adam_step(&o_w1, net.w1, g.dw1.data, 4));
    CHECK(nir_adam_step(&o_b1, net.b1, g.db1.data, 2));
    CHECK(nir_adam_step(&o_w2, net.w2, g.dw2.data, 4));
    CHECK(nir_adam_step(&o_b2, net.b2, g.db2.data, 2));
    double want[] = {0.4900000001650498, -0.4900000003300996,
        0.2599999984691973, 0.740000003061605, -0.009999999917475093,
        0.10999999923459859, 1.0, -1.0, -0.5, 0.5,
        0.09000000018187308, -0.09000000018187308};
    const float *bufs[] = {net.w1, net.b1, net.w2, net.b2};
    size_t sizes[] = {4, 2, 4, 2};
    size_t k = 0;
    for (size_t b = 0; b < 4; b++) {
        for (size_t i = 0; i < sizes[b]; i++, k++) {
            CHECK_NEAR(bufs[b][i], want[k], 1e-5);
        }
    }
    loss = topo_loss_and_grads(&net, &g, &input, 1);
    CHECK_NEAR(loss, 0.7871917250389795, 1e-5);
    nir_adam_free(&o_w1);
    nir_adam_free(&o_b1);
    nir_adam_free(&o_w2);
    nir_adam_free(&o_b2);
    nir_tensor_free(&input);
    topo_free(&g);
}

static void test_topology_converges(void) {
    topo_t net;
    topo_init(&net);
    topo_grad_t g;
    topo_alloc(&g);
    float xv[] = {0.5f, -0.25f};
    nir_tensor_t input = make_vector(xv, 2);
    char *error = NULL;
    nir_adam_t o_w1, o_b1, o_w2, o_b2;
    CHECK(nir_adam_init(&o_w1, 4, 0.05f, 0.9f, 0.999f, 1e-8f, 0, &error));
    CHECK(nir_adam_init(&o_b1, 2, 0.05f, 0.9f, 0.999f, 1e-8f, 0, &error));
    CHECK(nir_adam_init(&o_w2, 4, 0.05f, 0.9f, 0.999f, 1e-8f, 0, &error));
    CHECK(nir_adam_init(&o_b2, 2, 0.05f, 0.9f, 0.999f, 1e-8f, 0, &error));
    float loss = 0.0f;
    for (int step = 0; step < 300; step++) {
        loss = topo_loss_and_grads(&net, &g, &input, 1);
        CHECK(nir_adam_step(&o_w1, net.w1, g.dw1.data, 4));
        CHECK(nir_adam_step(&o_b1, net.b1, g.db1.data, 2));
        CHECK(nir_adam_step(&o_w2, net.w2, g.dw2.data, 4));
        CHECK(nir_adam_step(&o_b2, net.b2, g.db2.data, 2));
    }
    loss = topo_loss_and_grads(&net, &g, &input, 1);
    CHECK(loss < 0.01f);
    nir_adam_free(&o_w1);
    nir_adam_free(&o_b1);
    nir_adam_free(&o_w2);
    nir_adam_free(&o_b2);
    nir_tensor_free(&input);
    topo_free(&g);
}

int main(void) {
    test_surrogate_names();
    test_surrogate_values();
    test_threshold_backward();
    test_li_backward();
    test_affine_backward();
    test_linear_backward();
    test_cross_entropy();
    test_sgd();
    test_adam();
    test_stdp();
    test_topology_step();
    test_topology_converges();
    return 0;
}
