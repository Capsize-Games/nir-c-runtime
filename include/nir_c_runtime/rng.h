/* Deterministic seeded RNG for training-side C code (#28).
 *
 * splitmix64 (Steele, Lea, O'Neill 2014): portable C11, no platform
 * entropy dependency, so embedded targets keep working. The stream
 * for a seed is fixed by the algorithm below -- identical across
 * runs and platforms -- which is what makes seeded spike encoding
 * reproducible. This is a sampler, not a CSPRNG.
 */
#ifndef NIR_C_RUNTIME_RNG_H
#define NIR_C_RUNTIME_RNG_H

#include <stdint.h>

typedef struct {
    uint64_t state;
} nir_rng_t;

/* Prime the generator with `seed`. */
void nir_rng_seed(nir_rng_t *rng, uint64_t seed);

/* Return the next 64 bits of the stream. */
uint64_t nir_rng_next_u64(nir_rng_t *rng);

/* Return a uniform double in [0, 1) from the top 53 bits of one draw. */
double nir_rng_next_f64(nir_rng_t *rng);

/* Return a uniform float in [0, 1) from the top 24 bits of one draw. */
float nir_rng_next_f32(nir_rng_t *rng);

#endif /* NIR_C_RUNTIME_RNG_H */
