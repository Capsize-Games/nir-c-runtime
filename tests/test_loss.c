/* MSE and spike-count losses: closed-form value and
 * gradient, cross-entropy equivalence, error cases (#47). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/loss.h"
#include "nir_c_runtime/train.h"
#include "test_util.h"

static nir_tensor_t make_vector(const float *values, size_t n) {
    size_t shape[1] = {n};
    nir_tensor_t t = nir_tensor_alloc(shape, 1);
    CHECK(t.data != NULL);
    memcpy(t.data, values, n * sizeof(float));
    return t;
}

static void check_grad(
    const nir_tensor_t *t, const double *want, size_t n
) {
    CHECK(t->size == n);
    for (size_t i = 0; i < n; i++) {
        CHECK_NEAR(t->data[i], want[i], 1e-6);
    }
}

static void test_mse(void) {
    float pv[] = {1.0f, 2.0f, 3.0f};
    float tv[] = {1.0f, 0.0f, 3.0f};
    nir_tensor_t pred = make_vector(pv, 3);
    nir_tensor_t target = make_vector(tv, 3);
    size_t shape[1] = {3};
    nir_tensor_t grad = nir_tensor_alloc(shape, 1);
    CHECK(grad.data != NULL);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&pred, &target, &loss, &grad));
    CHECK_NEAR(loss, 1.3333333333333333, 1e-6);
    double want[] = {0.0, 1.3333333333333333, 0.0};
    check_grad(&grad, want, 3);
    nir_tensor_free(&pred);
    nir_tensor_free(&target);
    nir_tensor_free(&grad);
}

static void test_mse_zero(void) {
    float pv[] = {1.0f, 2.0f, 3.0f};
    nir_tensor_t pred = make_vector(pv, 3);
    size_t shape[1] = {3};
    nir_tensor_t grad = nir_tensor_alloc(shape, 1);
    CHECK(grad.data != NULL);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&pred, &pred, &loss, &grad));
    CHECK_NEAR(loss, 0.0, 1e-7);
    double zero[] = {0.0, 0.0, 0.0};
    check_grad(&grad, zero, 3);
    nir_tensor_free(&pred);
    nir_tensor_free(&grad);
}

static void test_mse_aliasing(void) {
    float pv[] = {1.0f, 2.0f, 3.0f};
    float tv[] = {1.0f, 0.0f, 3.0f};
    nir_tensor_t pred = make_vector(pv, 3);
    nir_tensor_t target = make_vector(tv, 3);
    float loss = 0.0f;
    CHECK(nir_mse_loss(&pred, &target, &loss, &pred));
    double want[] = {0.0, 1.3333333333333333, 0.0};
    check_grad(&pred, want, 3);
    nir_tensor_free(&pred);
    nir_tensor_free(&target);
}

static void test_mse_overlap(void) {
    float tv[] = {1.0f, 0.0f, 3.0f};
    nir_tensor_t target = make_vector(tv, 3);
    float *block = (float *)calloc(4, sizeof(float));
    CHECK(block != NULL);
    nir_tensor_t p;
    nir_tensor_t g;
    memset(&p, 0, sizeof(p));
    memset(&g, 0, sizeof(g));
    size_t shape[1] = {3};
    CHECK(nir_tensor_bind(&p, block, 4, shape, 1));
    CHECK(nir_tensor_bind(&g, block + 1, 3, shape, 1));
    float loss = 0.0f;
    CHECK(!nir_mse_loss(&p, &target, &loss, &g));
    free(block);
    nir_tensor_free(&target);
}

static void test_mse_nulls(void) {
    float pv[] = {1.0f, 2.0f};
    nir_tensor_t pred = make_vector(pv, 2);
    nir_tensor_t target = make_vector(pv, 2);
    size_t shape[1] = {2};
    nir_tensor_t grad = nir_tensor_alloc(shape, 1);
    CHECK(grad.data != NULL);
    float loss = 0.0f;
    CHECK(!nir_mse_loss(NULL, &target, &loss, &grad));
    CHECK(!nir_mse_loss(&pred, NULL, &loss, &grad));
    CHECK(!nir_mse_loss(&pred, &target, NULL, &grad));
    CHECK(!nir_mse_loss(&pred, &target, &loss, NULL));
    nir_tensor_free(&pred);
    nir_tensor_free(&target);
    nir_tensor_free(&grad);
}

static void test_mse_bad_data(void) {
    float pv[] = {1.0f, 2.0f};
    float tv[] = {1.0f, 2.0f, 3.0f};
    float bad[] = {1.0f, (float)NAN};
    nir_tensor_t pred = make_vector(pv, 2);
    nir_tensor_t target = make_vector(tv, 3);
    nir_tensor_t nonfinite = make_vector(bad, 2);
    size_t shape[1] = {2};
    nir_tensor_t grad = nir_tensor_alloc(shape, 1);
    CHECK(grad.data != NULL);
    float loss = 0.0f;
    CHECK(!nir_mse_loss(&pred, &target, &loss, &grad));
    CHECK(!nir_mse_loss(&nonfinite, &pred, &loss, &grad));
    CHECK(!nir_mse_loss(&pred, &nonfinite, &loss, &grad));
    nir_tensor_free(&pred);
    nir_tensor_free(&target);
    nir_tensor_free(&nonfinite);
    nir_tensor_free(&grad);
}

static void test_spike_count(void) {
    float cv[] = {2.0f, 5.0f, 1.0f};
    nir_tensor_t counts = make_vector(cv, 3);
    size_t shape[1] = {3};
    nir_tensor_t grad = nir_tensor_alloc(shape, 1);
    CHECK(grad.data != NULL);
    float loss = 0.0f;
    CHECK(nir_spike_count_loss(&counts, 1, &loss, &grad));
    CHECK_NEAR(loss, 0.06588390375742913, 1e-6);
    double want[] = {0.0466126225779739, -0.06376044812349424,
        0.01714782554552039};
    check_grad(&grad, want, 3);
    CHECK(!nir_spike_count_loss(&counts, 3, &loss, &grad));
    CHECK(!nir_spike_count_loss(NULL, 1, &loss, &grad));
    nir_tensor_free(&counts);
    nir_tensor_free(&grad);
}

static void test_spike_count_matches_ce(void) {
    float cv[] = {2.0f, 5.0f, 1.0f};
    nir_tensor_t counts = make_vector(cv, 3);
    size_t shape[1] = {3};
    nir_tensor_t grad = nir_tensor_alloc(shape, 1);
    nir_tensor_t ce_grad = nir_tensor_alloc(shape, 1);
    CHECK(grad.data != NULL && ce_grad.data != NULL);
    float loss = 0.0f;
    float ce_loss = 0.0f;
    CHECK(nir_spike_count_loss(&counts, 1, &loss, &grad));
    CHECK(nir_cross_entropy(&counts, 1, &ce_loss, &ce_grad));
    CHECK(ce_loss == loss);
    CHECK(memcmp(grad.data, ce_grad.data, 3 * sizeof(float)) == 0);
    nir_tensor_free(&counts);
    nir_tensor_free(&grad);
    nir_tensor_free(&ce_grad);
}

int main(void) {
    test_mse();
    test_mse_zero();
    test_mse_aliasing();
    test_mse_overlap();
    test_mse_nulls();
    test_mse_bad_data();
    test_spike_count();
    test_spike_count_matches_ce();
    return 0;
}
