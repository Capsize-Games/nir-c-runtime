#include "nir_c_runtime/fixed_event.h"

#include <limits.h>

static int config_valid(const nir_fixed_event_config_t *config) {
    return config != NULL && config->leak_shift > 0u && config->leak_shift < 31u &&
           config->membrane_min <= config->elevated_threshold &&
           config->elevated_threshold < config->critical_threshold &&
           config->critical_threshold <= config->membrane_max;
}

static int32_t clamp_i64(int64_t value, int32_t low, int32_t high) {
    if (value < (int64_t)low) {
        return low;
    }
    if (value > (int64_t)high) {
        return high;
    }
    return (int32_t)value;
}

static int64_t leak_term(int32_t membrane, uint8_t leak_shift) {
    return (int64_t)membrane / ((int64_t)1 << leak_shift);
}

uint8_t nir_fixed_event_step(const nir_fixed_event_config_t *config,
                             nir_fixed_event_state_t *state,
                             const uint8_t features[NIR_FIXED_EVENT_FEATURES]) {
    if (!config_valid(config) || state == NULL || features == NULL ||
        state->membrane < config->membrane_min || state->membrane > config->membrane_max) {
        return NIR_FIXED_EVENT_UNKNOWN;
    }

    int64_t accumulator = (int64_t)state->membrane -
                          leak_term(state->membrane, config->leak_shift) +
                          (int64_t)config->bias;
    for (size_t i = 0; i < NIR_FIXED_EVENT_FEATURES; ++i) {
        accumulator += (int64_t)config->weights[i] * (int64_t)features[i];
    }
    state->membrane = clamp_i64(accumulator, config->membrane_min, config->membrane_max);
    if (state->membrane >= config->critical_threshold) {
        return NIR_FIXED_EVENT_CRITICAL;
    }
    if (state->membrane >= config->elevated_threshold) {
        return NIR_FIXED_EVENT_ELEVATED;
    }
    return NIR_FIXED_EVENT_NORMAL;
}

uint8_t nir_fixed_guard_next(uint8_t state, uint8_t event) {
    if (state > NIR_FIXED_GUARD_COOLDOWN || event > NIR_FIXED_EVENT_UNKNOWN ||
        event == NIR_FIXED_EVENT_UNKNOWN) {
        return NIR_FIXED_GUARD_OBSERVE;
    }
    if (state == NIR_FIXED_GUARD_OBSERVE) {
        return event >= NIR_FIXED_EVENT_ELEVATED ?
               NIR_FIXED_GUARD_SUSPECT : NIR_FIXED_GUARD_OBSERVE;
    }
    if (state == NIR_FIXED_GUARD_SUSPECT) {
        if (event == NIR_FIXED_EVENT_CRITICAL) {
            return NIR_FIXED_GUARD_CONFIRMED;
        }
        return event == NIR_FIXED_EVENT_NORMAL ?
               NIR_FIXED_GUARD_OBSERVE : NIR_FIXED_GUARD_SUSPECT;
    }
    if (state == NIR_FIXED_GUARD_CONFIRMED) {
        return NIR_FIXED_GUARD_COOLDOWN;
    }
    return event == NIR_FIXED_EVENT_NORMAL ?
           NIR_FIXED_GUARD_OBSERVE : NIR_FIXED_GUARD_COOLDOWN;
}

uint8_t nir_fixed_guard_action(uint8_t state, uint8_t event) {
    if (state > NIR_FIXED_GUARD_COOLDOWN || event > NIR_FIXED_EVENT_UNKNOWN) {
        return NIR_FIXED_GUARD_NONE;
    }
    if (state == NIR_FIXED_GUARD_SUSPECT && event == NIR_FIXED_EVENT_CRITICAL) {
        return NIR_FIXED_GUARD_THROTTLE_RESET;
    }
    if (state == NIR_FIXED_GUARD_OBSERVE &&
        event >= NIR_FIXED_EVENT_ELEVATED && event <= NIR_FIXED_EVENT_CRITICAL) {
        return NIR_FIXED_GUARD_COLLECT_SAMPLE;
    }
    return NIR_FIXED_GUARD_NONE;
}
