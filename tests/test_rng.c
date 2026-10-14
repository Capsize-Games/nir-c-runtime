/* Seeded RNG: known splitmix64 vectors plus determinism. */
#include <stdint.h>

#include "nir_c_runtime/rng.h"
#include "test_util.h"

static void check_stream(uint64_t seed, const uint64_t want[4]) {
    nir_rng_t rng;
    nir_rng_seed(&rng, seed);
    for (int i = 0; i < 4; i++) {
        CHECK(nir_rng_next_u64(&rng) == want[i]);
    }
}

int main(void) {
    /* Reference vectors (splitmix64, independently computed). */
    const uint64_t seed0[4] = {
        0xe220a8397b1dcdafULL, 0x6e789e6aa1b965f4ULL,
        0x06c45d188009454fULL, 0xf88bb8a8724c81ecULL,
    };
    const uint64_t seed1[4] = {
        0x910a2dec89025cc1ULL, 0xbeeb8da1658eec67ULL,
        0xf893a2eefb32555eULL, 0x71c18690ee42c90bULL,
    };
    const uint64_t seed_big[4] = {
        0x22118258a9d111a0ULL, 0x346edce5f713f8edULL,
        0x1e9a57bc80e6721dULL, 0x2d160e7e5c3f42caULL,
    };
    check_stream(0, seed0);
    check_stream(1, seed1);
    check_stream(12345, seed_big);

    /* Same seed, same stream, across instances and float widths. */
    nir_rng_t a;
    nir_rng_t b;
    nir_rng_seed(&a, 999);
    nir_rng_seed(&b, 999);
    for (int i = 0; i < 1000; i++) {
        CHECK(nir_rng_next_u64(&a) == nir_rng_next_u64(&b));
    }
    nir_rng_seed(&a, 7);
    nir_rng_seed(&b, 7);
    for (int i = 0; i < 1000; i++) {
        double x = nir_rng_next_f64(&a);
        double y = nir_rng_next_f64(&b);
        CHECK(x == y && x >= 0.0 && x < 1.0);
        float fx = nir_rng_next_f32(&a);
        float fy = nir_rng_next_f32(&b);
        CHECK(fx == fy && fx >= 0.0f && fx < 1.0f);
    }

    /* Distinct seeds diverge immediately. */
    nir_rng_seed(&a, 0);
    nir_rng_seed(&b, 1);
    CHECK(nir_rng_next_u64(&a) != nir_rng_next_u64(&b));
    return 0;
}
