/* Spike encoders vs committed goldens (tools/encode_golden.py). */
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/encode.h"
#include "nir_c_runtime/json.h"
#include "test_util.h"

static float *load_floats(const json_value_t *array, size_t *n) {
    size_t count = json_array_length(array);
    float *out = (float *)malloc(count * sizeof(float));
    CHECK(out != NULL);
    for (size_t i = 0; i < count; i++) {
        out[i] = (float)json_as_number(json_array_get(array, i), 0.0);
    }
    *n = count;
    return out;
}

static void check_bits(
    const json_value_t *entry, const uint8_t *got, size_t n
) {
    const json_value_t *bits = json_object_get(entry, "bits");
    CHECK(json_array_length(bits) == n);
    for (size_t i = 0; i < n; i++) {
        int want = (int)json_as_number(json_array_get(bits, i), -1);
        if (got[i] != want) {
            fprintf(
                stderr, "bit %zu: got %u want %d\n", i, got[i], want
            );
            exit(1);
        }
    }
}

static size_t num_steps_of(const json_value_t *entry) {
    return (size_t)json_as_number(json_object_get(entry, "steps"), 0);
}

int main(void) {
    char *text = read_file("fixtures/encode_golden.json");
    char *error = NULL;
    json_value_t *goldens = json_parse(text, &error);
    CHECK(error == NULL);
    CHECK(goldens != NULL);

    /* Rate: Bernoulli rule against the same RNG stream. */
    const json_value_t *rate = json_object_get(goldens, "rate");
    size_t n = 0;
    float *x = load_floats(json_object_get(rate, "x"), &n);
    size_t steps = num_steps_of(rate);
    uint8_t *out = (uint8_t *)malloc(steps * n);
    CHECK(out != NULL);
    nir_rng_t rng;
    nir_rng_seed(&rng, 12345);
    CHECK(nir_encode_rate(x, n, steps, 1.0f, &rng, out, &error));
    CHECK(error == NULL);
    check_bits(rate, out, steps * n);
    /* Deterministic: the same seed reproduces the train. */
    nir_rng_seed(&rng, 12345);
    uint8_t *again = (uint8_t *)malloc(steps * n);
    CHECK(nir_encode_rate(x, n, steps, 1.0f, &rng, again, &error));
    CHECK(memcmp(out, again, steps * n) == 0);
    free(again);
    free(x);

    /* Latency: linear, clipped, unnormalized, and log branches. */
    const char *cases[4] = {
        "latency_linear", "latency_linear_clip", "latency_linear_plain",
        "latency_log",
    };
    for (int c = 0; c < 4; c++) {
        const json_value_t *entry = json_object_get(goldens, cases[c]);
        x = load_floats(json_object_get(entry, "x"), &n);
        steps = num_steps_of(entry);
        free(out);
        out = (uint8_t *)malloc(steps * n);
        nir_latency_params_t params;
        memset(&params, 0, sizeof(params));
        params.num_steps = steps;
        params.tau = (float)json_as_number(
            json_object_get(entry, "tau"), 5.0
        );
        params.threshold = (float)json_as_number(
            json_object_get(entry, "threshold"), 0.01
        );
        const json_value_t *clip = json_object_get(entry, "clip");
        params.clip = clip != NULL && clip->type == JSON_BOOL &&
            clip->as.boolean;
        const json_value_t *norm = json_object_get(entry, "normalize");
        params.normalize = norm != NULL && norm->type == JSON_BOOL &&
            norm->as.boolean;
        const json_value_t *lin = json_object_get(entry, "linear");
        params.linear = lin != NULL && lin->type == JSON_BOOL &&
            lin->as.boolean;
        params.first_spike_time = (size_t)json_as_number(
            json_object_get(entry, "first_spike_time"), 0
        );
        params.epsilon = NIR_LATENCY_EPSILON;
        CHECK(nir_encode_latency(x, n, &params, out, &error));
        CHECK(error == NULL);
        check_bits(entry, out, steps * n);
        free(x);
    }

    /* Delta over the 0..1 ramp. */
    const json_value_t *delta = json_object_get(goldens, "delta");
    x = load_floats(json_object_get(delta, "x"), &n);
    steps = num_steps_of(delta);
    free(out);
    out = (uint8_t *)malloc(steps * n);
    float thr = (float)json_as_number(
        json_object_get(delta, "threshold"), 0
    );
    CHECK(nir_encode_delta(x, n, steps, thr, out, &error));
    CHECK(error == NULL);
    check_bits(delta, out, steps * n);
    free(x);

    /* Random noise baseline. */
    const json_value_t *random = json_object_get(goldens, "random");
    n = (size_t)json_as_number(json_object_get(random, "n"), 0);
    steps = num_steps_of(random);
    free(out);
    out = (uint8_t *)malloc(steps * n);
    nir_rng_seed(&rng, 999);
    float scale = (float)json_as_number(
        json_object_get(random, "scale"), 0
    );
    CHECK(nir_encode_random(n, steps, scale, &rng, out, &error));
    CHECK(error == NULL);
    check_bits(random, out, steps * n);
    free(out);

    /* Repeat: raw pixels across steps, no spikes involved. */
    float px[3] = {0.25f, 0.5f, 1.0f};
    float rep[12];
    CHECK(nir_encode_repeat(px, 3, 4, rep, &error));
    for (int t = 0; t < 4; t++) {
        CHECK(rep[t * 3] == 0.25f && rep[t * 3 + 1] == 0.5f &&
            rep[t * 3 + 2] == 1.0f);
    }

    /* Error paths mirror the torch side's raises. */
    uint8_t tiny[8];
    float bad[2] = {0.5f, 1.5f};
    nir_latency_params_t lp;
    memset(&lp, 0, sizeof(lp));
    lp.num_steps = 4;
    lp.tau = 5.0f;
    lp.threshold = 0.01f;
    lp.normalize = 1;
    lp.linear = 1;
    lp.epsilon = NIR_LATENCY_EPSILON;
    CHECK(!nir_encode_latency(bad, 2, &lp, tiny, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    /* Unnormalized tau=100 overflows 4 steps: spike time is 99. */
    lp.normalize = 0;
    lp.tau = 100.0f;
    float ok[1] = {0.0f};
    CHECK(!nir_encode_latency(ok, 1, &lp, tiny, &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;
    nir_rng_seed(&rng, 0);
    CHECK(!nir_encode_rate(NULL, 0, 4, 1.0f, &rng, tiny, NULL));
    CHECK(!nir_encode_delta(px, 3, 0, 0.04f, tiny, NULL));
    CHECK(!nir_encode_random(3, 4, 0.5f, NULL, tiny, NULL));
    CHECK(!nir_encode_repeat(px, 3, 0, rep, NULL));

    json_free(goldens);
    free(text);
    return 0;
}
