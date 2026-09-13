#include "nir_c_runtime/fixed_event.h"

#include "test_util.h"

static const nir_fixed_event_config_t IRQ_CONFIG = {
    .weights = {3, 12, 2, 16},
    .bias = -192,
    .leak_shift = 3,
    .membrane_min = 0,
    .membrane_max = 4095,
    .elevated_threshold = 768,
    .critical_threshold = 2048,
};

static const nir_fixed_event_config_t LOGIC_CONFIG = {
    .weights = {2, 5, 1, 7},
    .bias = -4,
    .leak_shift = 2,
    .membrane_min = 0,
    .membrane_max = 31,
    .elevated_threshold = 16,
    .critical_threshold = 31,
};

static void test_reference_parity(void) {
    nir_fixed_event_state_t quiet_state = {.membrane = 0};
    const uint8_t quiet[4] = {10, 0, 5, 0};
    CHECK(nir_fixed_event_step(&IRQ_CONFIG, &quiet_state, quiet) ==
          NIR_FIXED_EVENT_NORMAL);
    CHECK(quiet_state.membrane == 0);

    nir_fixed_event_state_t hot_state = {.membrane = 1800};
    const uint8_t hot[4] = {255, 8, 255, 8};
    CHECK(nir_fixed_event_step(&IRQ_CONFIG, &hot_state, hot) ==
          NIR_FIXED_EVENT_CRITICAL);
    CHECK(hot_state.membrane == 2882);
}

static void test_invalid_input_fails_closed(void) {
    nir_fixed_event_config_t invalid = IRQ_CONFIG;
    invalid.leak_shift = 0;
    nir_fixed_event_state_t state = {.membrane = 100};
    const uint8_t input[4] = {1, 2, 3, 4};
    CHECK(nir_fixed_event_step(&invalid, &state, input) == NIR_FIXED_EVENT_UNKNOWN);
    CHECK(state.membrane == 100);
    CHECK(nir_fixed_event_step(&IRQ_CONFIG, NULL, input) == NIR_FIXED_EVENT_UNKNOWN);
    CHECK(nir_fixed_event_step(&IRQ_CONFIG, &state, NULL) == NIR_FIXED_EVENT_UNKNOWN);
}

static void test_negative_membrane_is_portable(void) {
    nir_fixed_event_config_t config = IRQ_CONFIG;
    config.bias = 0;
    config.membrane_min = -4095;
    config.elevated_threshold = 0;
    config.critical_threshold = 2048;
    nir_fixed_event_state_t state = {.membrane = -9};
    const uint8_t input[4] = {0, 0, 0, 0};

    CHECK(nir_fixed_event_step(&config, &state, input) ==
          NIR_FIXED_EVENT_NORMAL);
    CHECK(state.membrane == -8);
}

static void test_exhaustive_logic_truth_table(void) {
    uint64_t score_sum = 0;
    uint32_t spike_count = 0;
    for (int32_t membrane = 0; membrane < 32; ++membrane) {
        for (uint8_t irq = 0; irq < 4; ++irq) {
            for (uint8_t timeout = 0; timeout < 4; ++timeout) {
                for (uint8_t pressure = 0; pressure < 4; ++pressure) {
                    for (uint8_t errors = 0; errors < 4; ++errors) {
                        nir_fixed_event_state_t state = {.membrane = membrane};
                        const uint8_t features[4] = {irq, timeout, pressure, errors};
                        uint8_t event = nir_fixed_event_step(&LOGIC_CONFIG, &state, features);
                        score_sum += (uint32_t)state.membrane;
                        spike_count += event != NIR_FIXED_EVENT_NORMAL;
                    }
                }
            }
        }
    }
    CHECK(score_sum == 211482u);
    CHECK(spike_count == 7248u);
}

static void test_guard_truth_table(void) {
    for (uint8_t state = 0; state < 4; ++state) {
        CHECK(nir_fixed_guard_next(state, NIR_FIXED_EVENT_UNKNOWN) ==
              NIR_FIXED_GUARD_OBSERVE);
        CHECK(nir_fixed_guard_action(state, NIR_FIXED_EVENT_UNKNOWN) ==
              NIR_FIXED_GUARD_NONE);
    }
    CHECK(nir_fixed_guard_next(NIR_FIXED_GUARD_OBSERVE,
                               NIR_FIXED_EVENT_ELEVATED) == NIR_FIXED_GUARD_SUSPECT);
    CHECK(nir_fixed_guard_action(NIR_FIXED_GUARD_OBSERVE,
                                 NIR_FIXED_EVENT_ELEVATED) == NIR_FIXED_GUARD_COLLECT_SAMPLE);
    CHECK(nir_fixed_guard_next(NIR_FIXED_GUARD_SUSPECT,
                               NIR_FIXED_EVENT_CRITICAL) == NIR_FIXED_GUARD_CONFIRMED);
    CHECK(nir_fixed_guard_action(NIR_FIXED_GUARD_SUSPECT,
                                 NIR_FIXED_EVENT_CRITICAL) == NIR_FIXED_GUARD_THROTTLE_RESET);
    CHECK(nir_fixed_guard_next(NIR_FIXED_GUARD_CONFIRMED,
                               NIR_FIXED_EVENT_CRITICAL) == NIR_FIXED_GUARD_COOLDOWN);
    CHECK(nir_fixed_guard_next(NIR_FIXED_GUARD_COOLDOWN,
                               NIR_FIXED_EVENT_NORMAL) == NIR_FIXED_GUARD_OBSERVE);
    CHECK(nir_fixed_guard_next(255, NIR_FIXED_EVENT_CRITICAL) ==
          NIR_FIXED_GUARD_OBSERVE);
    CHECK(nir_fixed_guard_action(255, NIR_FIXED_EVENT_CRITICAL) ==
          NIR_FIXED_GUARD_NONE);
}

int main(void) {
    test_reference_parity();
    test_invalid_input_fails_closed();
    test_negative_membrane_is_portable();
    test_exhaustive_logic_truth_table();
    test_guard_truth_table();
    return 0;
}
