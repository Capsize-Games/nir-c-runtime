/* Mini-batch training: accumulation over a caller loop equals the
 * sum of per-sample gradients, the batch mean matches the
 * single-sample losses, and epoch shuffling replays from its
 * seed. References are float64 closed forms; tolerances separate
 * float32 rounding from formula errors.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/data.h"
#include "nir_c_runtime/minibatch.h"
#include "nir_c_runtime/ops.h"
#include "nir_c_runtime/rng.h"
#include "nir_c_runtime/train.h"
#include "test_util.h"

static nir_tensor_t make_vector(const float *values, size_t n) {
    size_t shape[1] = {n};
    nir_tensor_t t = nir_tensor_alloc(shape, 1);
    CHECK(t.data != NULL);
    memcpy(t.data, values, n * sizeof(float));
    return t;
}

static void test_zero_accumulate_scale(void) {
    /* Dyadic fractions, so the sums below are exact in float32. */
    float g1[] = {1.5f, -2.25f, 3.5f};
    float g2[] = {0.5f, 1.25f, -1.5f};
    float g3[] = {-1.0f, 0.5f, -1.0f};
    float accum[3];
    nir_minibatch_zero(accum, 3);
    CHECK(accum[0] == 0.0f && accum[1] == 0.0f && accum[2] == 0.0f);
    CHECK(nir_minibatch_accumulate(accum, g1, 3));
    CHECK(nir_minibatch_accumulate(accum, g2, 3));
    CHECK(nir_minibatch_accumulate(accum, g3, 3));
    CHECK(accum[0] == 1.0f && accum[1] == -0.5f && accum[2] == 1.0f);
    CHECK(nir_minibatch_scale(accum, 3, 1.0f / 3.0f));
    CHECK_NEAR(accum[0], 1.0 / 3.0, 1e-7);
    CHECK_NEAR(accum[1], -1.0 / 6.0, 1e-7);
    CHECK_NEAR(accum[2], 1.0 / 3.0, 1e-7);
}

/* A batch of B=4 affine samples: the helper path (zero, then one
 * accumulate per sample) must match the sum of the per-sample
 * gradients, checked against a float64 reference and against the
 * op's own repeated-call accumulation. */
static void test_accumulate_matches_affine_loop(void) {
    float w[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {0.5f, -0.5f};
    nir_affine_params_t params = {2, 2, w, b};
    float xs[4][2] = {
        {0.5f, -0.25f}, {1.0f, 0.5f}, {-1.5f, 2.0f}, {0.125f, -1.0f},
    };
    float gs[4][2] = {
        {1.0f, -0.5f}, {0.25f, 0.75f}, {-2.0f, 1.0f}, {0.5f, 0.5f},
    };
    size_t mshape[2] = {2, 2};
    size_t vshape[1] = {2};
    nir_tensor_t sample_dw = nir_tensor_alloc(mshape, 2);
    nir_tensor_t sample_db = nir_tensor_alloc(vshape, 1);
    nir_tensor_t direct_dw = nir_tensor_alloc(mshape, 2);
    nir_tensor_t direct_db = nir_tensor_alloc(vshape, 1);
    nir_tensor_t grad_in = nir_tensor_alloc(vshape, 1);
    CHECK(sample_dw.data != NULL && sample_db.data != NULL);
    CHECK(direct_dw.data != NULL && direct_db.data != NULL);
    CHECK(grad_in.data != NULL);
    float accum_w[4];
    float accum_b[2];
    double ref_w[4] = {0.0, 0.0, 0.0, 0.0};
    double ref_b[2] = {0.0, 0.0};
    nir_minibatch_zero(accum_w, 4);
    nir_minibatch_zero(accum_b, 2);
    for (size_t k = 0; k < 4; k++) {
        nir_tensor_t input = make_vector(xs[k], 2);
        nir_tensor_t gout = make_vector(gs[k], 2);
        memset(sample_dw.data, 0, 4 * sizeof(float));
        memset(sample_db.data, 0, 2 * sizeof(float));
        CHECK(nir_op_affine_backward(
            &params, &input, &gout, &sample_dw, &sample_db,
            &grad_in
        ));
        /* The direct path accumulates across calls untouched. */
        CHECK(nir_op_affine_backward(
            &params, &input, &gout, &direct_dw, &direct_db,
            &grad_in
        ));
        CHECK(nir_minibatch_accumulate(
            accum_w, sample_dw.data, 4
        ));
        CHECK(nir_minibatch_accumulate(
            accum_b, sample_db.data, 2
        ));
        for (size_t i = 0; i < 4; i++) {
            ref_w[i] += (double)sample_dw.data[i];
        }
        for (size_t i = 0; i < 2; i++) {
            ref_b[i] += (double)sample_db.data[i];
        }
        nir_tensor_free(&input);
        nir_tensor_free(&gout);
    }
    for (size_t i = 0; i < 4; i++) {
        CHECK_NEAR(accum_w[i], ref_w[i], 1e-6);
        CHECK(accum_w[i] == direct_dw.data[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        CHECK_NEAR(accum_b[i], ref_b[i], 1e-6);
        CHECK(accum_b[i] == direct_db.data[i]);
    }
    CHECK(nir_minibatch_scale(accum_w, 4, 1.0f / 4.0f));
    CHECK(nir_minibatch_scale(accum_b, 2, 1.0f / 4.0f));
    for (size_t i = 0; i < 4; i++) {
        CHECK_NEAR(accum_w[i], ref_w[i] / 4.0, 1e-6);
    }
    for (size_t i = 0; i < 2; i++) {
        CHECK_NEAR(accum_b[i], ref_b[i] / 4.0, 1e-6);
    }
    nir_tensor_free(&sample_dw);
    nir_tensor_free(&sample_db);
    nir_tensor_free(&direct_dw);
    nir_tensor_free(&direct_db);
    nir_tensor_free(&grad_in);
}

/* A batch of B=4 cross-entropy samples: the mean loss and the
 * mean gradient match the per-sample single-sample CE values. */
static void test_mean_loss_matches_ce(void) {
    float ls[4][3] = {
        {1.0f, 2.0f, 3.0f},
        {0.5f, -1.0f, 0.25f},
        {-2.0f, 0.0f, 1.0f},
        {3.0f, 3.0f, 3.0f},
    };
    size_t targets[4] = {2, 0, 1, 2};
    size_t shape[1] = {3};
    nir_tensor_t grad = nir_tensor_alloc(shape, 1);
    CHECK(grad.data != NULL);
    float losses[4];
    float accum[3];
    double ref_loss = 0.0;
    double ref_grad[3] = {0.0, 0.0, 0.0};
    nir_minibatch_zero(accum, 3);
    for (size_t k = 0; k < 4; k++) {
        nir_tensor_t logits = make_vector(ls[k], 3);
        CHECK(nir_cross_entropy(
            &logits, targets[k], &losses[k], &grad
        ));
        CHECK(nir_minibatch_accumulate(accum, grad.data, 3));
        ref_loss += (double)losses[k];
        for (size_t i = 0; i < 3; i++) {
            ref_grad[i] += (double)grad.data[i];
        }
        nir_tensor_free(&logits);
    }
    float mean = 0.0f;
    CHECK(nir_minibatch_mean_loss(losses, 4, &mean));
    CHECK_NEAR(mean, ref_loss / 4.0, 1e-7);
    CHECK(nir_minibatch_scale(accum, 3, 1.0f / 4.0f));
    for (size_t i = 0; i < 3; i++) {
        CHECK_NEAR(accum[i], ref_grad[i] / 4.0, 1e-6);
    }
    nir_tensor_free(&grad);
}

static void check_permutation(
    const size_t *indices, size_t n
) {
    int *seen = (int *)calloc(n == 0 ? 1 : n, sizeof(int));
    CHECK(seen != NULL);
    for (size_t i = 0; i < n; i++) {
        CHECK(indices[i] < n);
        CHECK(!seen[indices[i]]);
        seen[indices[i]] = 1;
    }
    free(seen);
}

static void test_shuffle_reproducible(void) {
    size_t a[16];
    size_t b[16];
    size_t c[16];
    nir_rng_t rng;
    nir_sampler_arange(a, 16);
    nir_rng_seed(&rng, 42);
    CHECK(nir_sampler_shuffle(&rng, a, 16));
    check_permutation(a, 16);
    nir_sampler_arange(b, 16);
    nir_rng_seed(&rng, 42);
    CHECK(nir_sampler_shuffle(&rng, b, 16));
    CHECK(memcmp(a, b, sizeof(a)) == 0);
    nir_sampler_arange(c, 16);
    nir_rng_seed(&rng, 43);
    CHECK(nir_sampler_shuffle(&rng, c, 16));
    check_permutation(c, 16);
    CHECK(memcmp(a, c, sizeof(a)) != 0);
}

/* Epoch shuffling over the committed tiny-MNIST fixture: the
 * order replays from its seed and indexes the real samples. */
static void test_shuffle_fixture(void) {
    char *error = NULL;
    uint8_t *pixels = NULL;
    uint8_t *labels = NULL;
    size_t count = 0;
    size_t rows = 0;
    size_t cols = 0;
    size_t nlabels = 0;
    CHECK(nir_idx_read_images(
        "fixtures/mnist_tiny_images.idx", &pixels, &count, &rows,
        &cols, &error
    ));
    CHECK(error == NULL && count == 3);
    CHECK(nir_idx_read_labels(
        "fixtures/mnist_tiny_labels.idx", &labels, &nlabels, &error
    ));
    CHECK(error == NULL && nlabels == count);
    size_t first[3];
    size_t second[3];
    nir_rng_t rng;
    nir_sampler_arange(first, count);
    nir_rng_seed(&rng, 7);
    CHECK(nir_sampler_shuffle(&rng, first, count));
    check_permutation(first, count);
    /* Pinned order for seed 7 (Fisher-Yates over splitmix64). */
    CHECK(first[0] == 1 && first[1] == 2 && first[2] == 0);
    nir_sampler_arange(second, count);
    nir_rng_seed(&rng, 7);
    CHECK(nir_sampler_shuffle(&rng, second, count));
    CHECK(memcmp(first, second, sizeof(first)) == 0);
    /* The order indexes real samples: labels 7,3,9 read as 3,9,7. */
    CHECK(labels[0] == 7 && labels[1] == 3 && labels[2] == 9);
    CHECK(labels[first[0]] == 3 && labels[first[1]] == 9 &&
        labels[first[2]] == 7);
    nir_tensor_t frame = nir_data_frame(
        pixels, count, rows, cols, first[0]
    );
    CHECK(frame.data != NULL && frame.size == rows * cols);
    nir_tensor_free(&frame);
    free(pixels);
    free(labels);
}

static void test_shuffle_edges(void) {
    nir_rng_t rng;
    nir_rng_seed(&rng, 1);
    CHECK(nir_sampler_shuffle(&rng, NULL, 0));
    CHECK(!nir_sampler_shuffle(NULL, NULL, 0));
    size_t one[1] = {0};
    CHECK(nir_sampler_shuffle(&rng, one, 1));
    CHECK(one[0] == 0);
    size_t two[2] = {0, 1};
    CHECK(!nir_sampler_shuffle(NULL, two, 2));
    CHECK(!nir_sampler_shuffle(&rng, NULL, 2));
}

static void test_errors(void) {
    float accum[3] = {1.0f, 2.0f, 3.0f};
    float grad[3] = {0.5f, 0.5f, 0.5f};
    float bad[3] = {0.5f, (float)NAN, 0.5f};
    float inf[3] = {0.5f, (float)INFINITY, 0.5f};
    float before[3];
    memcpy(before, accum, sizeof(before));
    CHECK(!nir_minibatch_accumulate(NULL, grad, 3));
    CHECK(!nir_minibatch_accumulate(accum, NULL, 3));
    CHECK(!nir_minibatch_accumulate(accum, grad, 0));
    CHECK(!nir_minibatch_accumulate(accum, bad, 3));
    CHECK(!nir_minibatch_accumulate(accum, inf, 3));
    CHECK(!nir_minibatch_accumulate(bad, grad, 3));
    CHECK(memcmp(accum, before, sizeof(before)) == 0);
    CHECK(!nir_minibatch_scale(NULL, 3, 0.5f));
    CHECK(!nir_minibatch_scale(accum, 0, 0.5f));
    CHECK(!nir_minibatch_scale(accum, 3, (float)NAN));
    CHECK(!nir_minibatch_scale(accum, 3, (float)INFINITY));
    CHECK(!nir_minibatch_scale(bad, 3, 0.5f));
    CHECK(memcmp(accum, before, sizeof(before)) == 0);
    CHECK(nir_minibatch_scale(accum, 3, 0.0f));
    CHECK(accum[0] == 0.0f && accum[1] == 0.0f && accum[2] == 0.0f);
    float losses[2] = {1.0f, 3.0f};
    float mean = 0.0f;
    CHECK(!nir_minibatch_mean_loss(NULL, 2, &mean));
    CHECK(!nir_minibatch_mean_loss(losses, 0, &mean));
    CHECK(!nir_minibatch_mean_loss(losses, 2, NULL));
    CHECK(!nir_minibatch_mean_loss(bad, 2, &mean));
    CHECK(!nir_minibatch_mean_loss(inf, 2, &mean));
    CHECK(nir_minibatch_mean_loss(losses, 2, &mean));
    CHECK(mean == 2.0f);
}

int main(void) {
    test_zero_accumulate_scale();
    test_accumulate_matches_affine_loop();
    test_mean_loss_matches_ce();
    test_shuffle_reproducible();
    test_shuffle_fixture();
    test_shuffle_edges();
    test_errors();
    return 0;
}
