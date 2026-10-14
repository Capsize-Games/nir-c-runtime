/* End-to-end C training driver (#44).
 *
 * Wires the tested training pieces into one loop with zero
 * Python at train time: IDX loading (data.h), spike encoding
 * (encode.h), BPTT tape forward/backward (tape.h), minibatch
 * mean reduction (minibatch.h), SGD/Adam steps (train.h), LR
 * schedules (schedule.h), and checkpoint save/load with
 * optimizer resume (checkpoint.h).
 *
 * The graph is borrowed and mutated in place (optimizer steps
 * write its weight/bias buffers). The tape owns retained
 * activations; the driver owns per-node accumulators and
 * optimizer instances. Temporal state never carries across
 * samples: every sample pairs one fresh tape forward with one
 * backward, matching the tape's unprimed convention.
 *
 * Batch reduction follows the minibatch.h contract: per-sample
 * gradients accumulate, scale by 1/B, then step. The LR
 * schedule counter is the driver's completed-step count,
 * persisted as the checkpoint meta step so resume continues
 * the curve exactly.
 */
#ifndef NIR_C_RUNTIME_DRIVER_H
#define NIR_C_RUNTIME_DRIVER_H

#include <stddef.h>
#include <stdint.h>

#include "nir_c_runtime/ops.h"
#include "nir_c_runtime/rng.h"
#include "nir_c_runtime/tape.h"
#include "nir_c_runtime/train.h"

typedef enum {
    NIR_DRIVER_OPT_SGD,
    NIR_DRIVER_OPT_ADAM
} nir_driver_opt_kind_t;

typedef enum {
    NIR_DRIVER_ENCODE_REPEAT,
    NIR_DRIVER_ENCODE_RATE
} nir_driver_encode_t;

typedef enum {
    NIR_DRIVER_SCHED_NONE,
    NIR_DRIVER_SCHED_STEP,
    NIR_DRIVER_SCHED_COSINE,
    NIR_DRIVER_SCHED_EXP
} nir_driver_sched_t;

typedef struct {
    size_t num_steps;
    nir_surrogate_kind_t surrogate;
    float slope;
    nir_driver_encode_t encode;
    float encode_gain;
    size_t batch_size;
    uint64_t seed;
    nir_driver_opt_kind_t opt_kind;
    float lr;
    float momentum;
    float weight_decay;
    float beta1;
    float beta2;
    float eps;
    nir_driver_sched_t sched;
    float sched_gamma;
    unsigned long sched_step_size;
    float sched_eta_min;
    unsigned long sched_t_max;
} nir_driver_config_t;

typedef struct {
    float *accum_w;
    float *accum_b;
    size_t w_count;
    size_t b_count;
    nir_sgd_t sgd_w;
    nir_sgd_t sgd_b;
    nir_adam_t adam_w;
    nir_adam_t adam_b;
    int has_w;
    int has_b;
    int sgd_w_live;
    int sgd_b_live;
    int adam_w_live;
    int adam_b_live;
} nir_driver_node_t;

typedef struct {
    nir_graph_t *graph;
    nir_tape_t tape;
    nir_driver_node_t *nodes;
    size_t node_count;
    nir_driver_config_t config;
    nir_rng_t shuffle_rng;
    nir_rng_t encode_rng;
    unsigned long step;
    int live;
} nir_driver_t;

typedef struct {
    float loss;
    float accuracy;
} nir_driver_stats_t;

/* Bind `graph` for training: validate `config`, init the tape,
 * allocate one accumulator plus one SGD/Adam instance per
 * weight/bias buffer, and seed the shuffle/encode streams.
 * `graph` must outlive `driver`. Returns 0 with *error set. */
int nir_driver_init(
    nir_driver_t *driver, nir_graph_t *graph,
    const nir_driver_config_t *config, char **error
);

/* Free tape, accumulators, and optimizer state. Safe on a
 * zeroed or already-freed driver. */
void nir_driver_free(nir_driver_t *driver);

/* One shuffled epoch over IDX buffers: encode each sample,
 * tape forward, cross-entropy, tape backward, accumulate,
 * mean-scale per batch, schedule the LR, step. Writes the
 * epoch mean loss and argmax accuracy. Fails on shape,
 * label-range, or non-finite errors with *error set. */
int nir_driver_train_epoch(
    nir_driver_t *driver, const uint8_t *pixels,
    const uint8_t *labels, size_t count, size_t rows,
    size_t cols, nir_driver_stats_t *stats, char **error
);

/* Forward-only pass over the same buffers: no shuffle, no
 * backward, no weight updates. Rate encoding still consumes
 * the encode stream. */
int nir_driver_eval(
    nir_driver_t *driver, const uint8_t *pixels,
    const uint8_t *labels, size_t count, size_t rows,
    size_t cols, nir_driver_stats_t *stats, char **error
);

/* Save weights plus optimizer state and the step counter.
 * `topology` names the run (NULL omits it). */
int nir_driver_save(
    nir_driver_t *driver, const char *path,
    const char *topology, char **error
);

/* Load weights plus optimizer state and the step counter.
 * Optimizer hyperparameters must already match the saver. */
int nir_driver_load(
    nir_driver_t *driver, const char *path, char **error
);

/* Completed optimizer steps (the schedule counter). */
unsigned long nir_driver_step(const nir_driver_t *driver);

#endif /* NIR_C_RUNTIME_DRIVER_H */
