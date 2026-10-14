/* LR schedules: closed-form step/cosine/exponential values,
 * boundary steps, optimizer composition, error cases (#46). */
#include <math.h>
#include <stdlib.h>

#include "nir_c_runtime/schedule.h"
#include "nir_c_runtime/train.h"
#include "test_util.h"

/* Each case sequences the call before the read: passing `lr`
 * in the same expression that writes it is unspecified. */
static void test_step(void) {
    float lr = 0.0f;
    CHECK(nir_lr_step_decay(0.1f, 0.5f, 10, 0, &lr));
    CHECK_NEAR(lr, 0.1, 1e-6);
    CHECK(nir_lr_step_decay(0.1f, 0.5f, 10, 9, &lr));
    CHECK_NEAR(lr, 0.1, 1e-6);
    CHECK(nir_lr_step_decay(0.1f, 0.5f, 10, 10, &lr));
    CHECK_NEAR(lr, 0.05, 1e-6);
    CHECK(nir_lr_step_decay(0.1f, 0.5f, 10, 19, &lr));
    CHECK_NEAR(lr, 0.05, 1e-6);
    CHECK(nir_lr_step_decay(0.1f, 0.5f, 10, 20, &lr));
    CHECK_NEAR(lr, 0.025, 1e-6);
    CHECK(nir_lr_step_decay(0.1f, 0.5f, 10, 25, &lr));
    CHECK_NEAR(lr, 0.025, 1e-6);
}

static void test_step_flat(void) {
    float lr = 0.0f;
    CHECK(nir_lr_step_decay(0.1f, 1.0f, 3, 100, &lr));
    CHECK_NEAR(lr, 0.1, 1e-6);
    CHECK(nir_lr_step_decay(0.0f, 0.5f, 10, 25, &lr));
    CHECK_NEAR(lr, 0.0, 1e-7);
}

static void test_cosine(void) {
    float lr = 0.0f;
    CHECK(nir_lr_cosine(0.1f, 0.0f, 100, 0, &lr));
    CHECK_NEAR(lr, 0.1, 1e-6);
    CHECK(nir_lr_cosine(0.1f, 0.0f, 100, 50, &lr));
    CHECK_NEAR(lr, 0.05, 1e-6);
    CHECK(nir_lr_cosine(0.1f, 0.0f, 100, 100, &lr));
    CHECK_NEAR(lr, 0.0, 1e-6);
    /* Torch-exact past t_max: the curve rises again. */
    CHECK(nir_lr_cosine(0.1f, 0.0f, 100, 200, &lr));
    CHECK_NEAR(lr, 0.1, 1e-6);
}

static void test_cosine_floor(void) {
    float lr = 0.0f;
    CHECK(nir_lr_cosine(0.1f, 0.01f, 100, 0, &lr));
    CHECK_NEAR(lr, 0.1, 1e-6);
    CHECK(nir_lr_cosine(0.1f, 0.01f, 100, 50, &lr));
    CHECK_NEAR(lr, 0.055, 1e-6);
    CHECK(nir_lr_cosine(0.1f, 0.01f, 100, 100, &lr));
    CHECK_NEAR(lr, 0.01, 1e-6);
}

static void test_exponential(void) {
    float lr = 0.0f;
    CHECK(nir_lr_exponential(0.1f, 0.5f, 0, &lr));
    CHECK_NEAR(lr, 0.1, 1e-6);
    CHECK(nir_lr_exponential(0.1f, 0.5f, 1, &lr));
    CHECK_NEAR(lr, 0.05, 1e-6);
    CHECK(nir_lr_exponential(0.1f, 0.5f, 3, &lr));
    CHECK_NEAR(lr, 0.0125, 1e-6);
    CHECK(nir_lr_exponential(0.1f, 1.0f, 7, &lr));
    CHECK_NEAR(lr, 0.1, 1e-6);
}

static void test_errors(void) {
    float lr = 0.0f;
    CHECK(!nir_lr_step_decay(0.1f, 0.5f, 10, 0, NULL));
    CHECK(!nir_lr_step_decay(-0.1f, 0.5f, 10, 0, &lr));
    CHECK(!nir_lr_step_decay(0.1f, -0.5f, 10, 0, &lr));
    CHECK(!nir_lr_step_decay(0.1f, 0.5f, 0, 0, &lr));
    CHECK(!nir_lr_cosine(0.1f, 0.0f, 0, 0, &lr));
    CHECK(!nir_lr_cosine(0.1f, -0.01f, 10, 0, &lr));
    CHECK(!nir_lr_cosine((float)NAN, 0.0f, 10, 0, &lr));
    CHECK(!nir_lr_exponential(0.1f, -0.5f, 0, &lr));
    CHECK(!nir_lr_exponential((float)INFINITY, 0.5f, 0, &lr));
}

/* Schedules drive both optimizers through their `lr` fields,
 * keyed by the checkpoint step counter (0 for a fresh run). */
static float resumed_lr(void) {
    float lr = 0.0f;
    CHECK(nir_lr_step_decay(0.1f, 0.5f, 10, 10, &lr));
    return lr;
}

static void test_sgd_compose(void) {
    char *error = NULL;
    nir_sgd_t sgd;
    CHECK(nir_sgd_init(&sgd, 1, 0.1f, 0.0f, 0.0f, &error));
    sgd.lr = resumed_lr();
    float w[] = {1.0f};
    float g[] = {2.0f};
    CHECK(nir_sgd_step(&sgd, w, g, 1));
    CHECK_NEAR(w[0], 0.9, 1e-6);
    nir_sgd_free(&sgd);
}

static void test_adam_compose(void) {
    char *error = NULL;
    nir_adam_t adam;
    CHECK(nir_adam_init(
        &adam, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f, &error
    ));
    adam.lr = resumed_lr();
    float w[] = {1.0f};
    float g[] = {0.1f};
    CHECK(nir_adam_step(&adam, w, g, 1));
    CHECK_NEAR(w[0], 0.95, 1e-6);
    nir_adam_free(&adam);
}

int main(void) {
    test_step();
    test_step_flat();
    test_cosine();
    test_cosine_floor();
    test_exponential();
    test_errors();
    test_sgd_compose();
    test_adam_compose();
    return 0;
}
