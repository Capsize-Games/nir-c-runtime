#ifndef NIR_C_RUNTIME_FIXED_EVENT_H
#define NIR_C_RUNTIME_FIXED_EVENT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NIR_FIXED_EVENT_FEATURES 4u

enum nir_fixed_event_kind {
    NIR_FIXED_EVENT_NORMAL = 0,
    NIR_FIXED_EVENT_ELEVATED = 1,
    NIR_FIXED_EVENT_CRITICAL = 2,
    NIR_FIXED_EVENT_UNKNOWN = 3
};

enum nir_fixed_guard_state {
    NIR_FIXED_GUARD_OBSERVE = 0,
    NIR_FIXED_GUARD_SUSPECT = 1,
    NIR_FIXED_GUARD_CONFIRMED = 2,
    NIR_FIXED_GUARD_COOLDOWN = 3
};

enum nir_fixed_guard_action {
    NIR_FIXED_GUARD_NONE = 0,
    NIR_FIXED_GUARD_COLLECT_SAMPLE = 1,
    NIR_FIXED_GUARD_THROTTLE_RESET = 2
};

typedef struct nir_fixed_event_config {
    int16_t weights[NIR_FIXED_EVENT_FEATURES];
    int32_t bias;
    uint8_t leak_shift;
    int32_t membrane_min;
    int32_t membrane_max;
    int32_t elevated_threshold;
    int32_t critical_threshold;
} nir_fixed_event_config_t;

typedef struct nir_fixed_event_state {
    int32_t membrane;
} nir_fixed_event_state_t;

/*
 * One allocation-free, integer-only leaky-integrate step. The returned value
 * is the module's single typed event output. Invalid configuration or input
 * returns UNKNOWN without mutating state.
 */
uint8_t nir_fixed_event_step(const nir_fixed_event_config_t *config,
                             nir_fixed_event_state_t *state,
                             const uint8_t features[NIR_FIXED_EVENT_FEATURES]);

/* Deterministic authority boundary matching JOE's irq_guard_* functions. */
uint8_t nir_fixed_guard_next(uint8_t state, uint8_t event);
uint8_t nir_fixed_guard_action(uint8_t state, uint8_t event);

#ifdef __cplusplus
}
#endif

#endif
