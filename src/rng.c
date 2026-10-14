#include "nir_c_runtime/rng.h"

void nir_rng_seed(nir_rng_t *rng, uint64_t seed) {
    rng->state = seed;
}

uint64_t nir_rng_next_u64(nir_rng_t *rng) {
    uint64_t z = rng->state + UINT64_C(0x9E3779B97F4A7C15);
    rng->state = z;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

double nir_rng_next_f64(nir_rng_t *rng) {
    uint64_t bits = nir_rng_next_u64(rng) >> 11;
    return (double)bits * (1.0 / 9007199254740992.0);
}

float nir_rng_next_f32(nir_rng_t *rng) {
    uint32_t bits = (uint32_t)(nir_rng_next_u64(rng) >> 40);
    return (float)bits * (1.0f / 16777216.0f);
}
