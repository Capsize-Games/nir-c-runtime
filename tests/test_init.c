/* Weight initializers: torch formulas, fan conventions,
 * seeded reproducibility, set_weights + forward (#45). */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/init.h"
#include "test_util.h"

#define STAT_N 20000

static void mean_var(
    const float *w, size_t n, double *mean, double *var
) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) m += (double)w[i];
    m /= (double)n;
    double v = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)w[i] - m;
        v += d * d;
    }
    *mean = m;
    *var = v / (double)n;
}

static float *fresh_buffer(void) {
    float *w = (float *)malloc(STAT_N * sizeof(float));
    CHECK(w != NULL);
    return w;
}

static void check_within(
    const float *a, size_t n, double bound
) {
    for (size_t i = 0; i < n; i++) {
        CHECK(fabs(a[i]) <= bound);
    }
}

static void test_fans(void) {
    size_t in = 0;
    size_t out = 0;
    size_t linear[2] = {3, 4};
    CHECK(nir_init_fans(linear, 2, &in, &out));
    CHECK(in == 4 && out == 3);
    size_t conv[4] = {16, 3, 3, 3};
    CHECK(nir_init_fans(conv, 4, &in, &out));
    CHECK(in == 27 && out == 144);
    size_t conv1[3] = {8, 4, 5};
    CHECK(nir_init_fans(conv1, 3, &in, &out));
    CHECK(in == 20 && out == 40);
}

static void test_fan_errors(void) {
    size_t in = 0;
    size_t out = 0;
    size_t shape[2] = {3, 4};
    size_t zero[2] = {3, 0};
    size_t big[3] = {SIZE_MAX, 2, SIZE_MAX};
    CHECK(!nir_init_fans(shape, 1, &in, &out));
    CHECK(!nir_init_fans(NULL, 2, &in, &out));
    CHECK(!nir_init_fans(shape, 2, NULL, &out));
    CHECK(!nir_init_fans(zero, 2, &in, &out));
    CHECK(!nir_init_fans(big, 3, &in, &out));
}

static void test_uniform(void) {
    float *a = fresh_buffer();
    float *b = fresh_buffer();
    nir_rng_t rng;
    nir_rng_seed(&rng, 1234);
    CHECK(nir_init_uniform(&rng, a, STAT_N, -2.0f, 2.0f));
    nir_rng_seed(&rng, 1234);
    CHECK(nir_init_uniform(&rng, b, STAT_N, -2.0f, 2.0f));
    CHECK(memcmp(a, b, STAT_N * sizeof(float)) == 0);
    double mean = 0.0;
    double var = 0.0;
    mean_var(a, STAT_N, &mean, &var);
    CHECK_NEAR(mean, 0.0, 0.05);
    CHECK_NEAR(var, 1.3333333333333333, 0.08);
    free(a);
    free(b);
}

static void test_xavier_uniform(void) {
    float *a = fresh_buffer();
    float *b = fresh_buffer();
    nir_rng_t rng;
    nir_rng_seed(&rng, 7);
    CHECK(nir_init_xavier_uniform(&rng, a, STAT_N, 128, 64, 1.0f));
    nir_rng_seed(&rng, 7);
    CHECK(nir_init_xavier_uniform(&rng, b, STAT_N, 128, 64, 1.0f));
    CHECK(memcmp(a, b, STAT_N * sizeof(float)) == 0);
    check_within(a, STAT_N, sqrt(6.0 / 192.0));
    double mean = 0.0;
    double var = 0.0;
    mean_var(a, STAT_N, &mean, &var);
    CHECK_NEAR(mean, 0.0, 0.01);
    CHECK_NEAR(var, 0.010416666666666666, 0.001);
    free(a);
    free(b);
}

static void test_xavier_normal(void) {
    float *a = fresh_buffer();
    float *b = fresh_buffer();
    nir_rng_t rng;
    nir_rng_seed(&rng, 7);
    CHECK(nir_init_xavier_normal(&rng, a, STAT_N, 128, 64, 1.0f));
    nir_rng_seed(&rng, 7);
    CHECK(nir_init_xavier_normal(&rng, b, STAT_N, 128, 64, 1.0f));
    CHECK(memcmp(a, b, STAT_N * sizeof(float)) == 0);
    double mean = 0.0;
    double var = 0.0;
    mean_var(a, STAT_N, &mean, &var);
    CHECK_NEAR(mean, 0.0, 0.01);
    CHECK_NEAR(var, 0.010416666666666666, 0.0015);
    free(a);
    free(b);
}

static void test_he_uniform(void) {
    float gain = (float)sqrt(2.0);
    float *a = fresh_buffer();
    float *b = fresh_buffer();
    nir_rng_t rng;
    nir_rng_seed(&rng, 21);
    CHECK(nir_init_he_uniform(&rng, a, STAT_N, 128, gain));
    nir_rng_seed(&rng, 21);
    CHECK(nir_init_he_uniform(&rng, b, STAT_N, 128, gain));
    CHECK(memcmp(a, b, STAT_N * sizeof(float)) == 0);
    double bound = (double)gain * sqrt(3.0 / 128.0);
    check_within(a, STAT_N, bound);
    double mean = 0.0;
    double var = 0.0;
    mean_var(a, STAT_N, &mean, &var);
    CHECK_NEAR(mean, 0.0, 0.01);
    CHECK_NEAR(var, 0.015625, 0.0015);
    free(a);
    free(b);
}

static void test_he_normal(void) {
    float gain = (float)sqrt(2.0);
    float *a = fresh_buffer();
    float *b = fresh_buffer();
    nir_rng_t rng;
    nir_rng_seed(&rng, 21);
    CHECK(nir_init_he_normal(&rng, a, STAT_N, 128, gain));
    nir_rng_seed(&rng, 21);
    CHECK(nir_init_he_normal(&rng, b, STAT_N, 128, gain));
    CHECK(memcmp(a, b, STAT_N * sizeof(float)) == 0);
    double mean = 0.0;
    double var = 0.0;
    mean_var(a, STAT_N, &mean, &var);
    CHECK_NEAR(mean, 0.0, 0.01);
    CHECK_NEAR(var, 0.015625, 0.0015);
    free(a);
    free(b);
}

static void test_init_errors(void) {
    nir_rng_t rng;
    nir_rng_seed(&rng, 1);
    float w[8];
    CHECK(!nir_init_uniform(NULL, w, 8, 0.0f, 1.0f));
    CHECK(!nir_init_uniform(&rng, NULL, 8, 0.0f, 1.0f));
    CHECK(!nir_init_uniform(&rng, w, 0, 0.0f, 1.0f));
    CHECK(!nir_init_uniform(&rng, w, 8, 1.0f, 1.0f));
    CHECK(!nir_init_uniform(&rng, w, 8, 2.0f, 1.0f));
    CHECK(!nir_init_xavier_uniform(&rng, w, 8, 0, 4, 1.0f));
    CHECK(!nir_init_xavier_normal(&rng, w, 8, 4, 0, 1.0f));
    CHECK(!nir_init_xavier_uniform(&rng, w, 8, 4, 4, -1.0f));
    CHECK(!nir_init_he_uniform(&rng, w, 8, 0, 1.0f));
    CHECK(!nir_init_he_normal(&rng, w, 8, 4, (float)NAN));
}

int main(void) {
    test_fans();
    test_fan_errors();
    test_uniform();
    test_xavier_uniform();
    test_xavier_normal();
    test_he_uniform();
    test_he_normal();
    test_init_errors();
    return 0;
}
