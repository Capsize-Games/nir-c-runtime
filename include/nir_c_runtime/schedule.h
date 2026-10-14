/* Learning-rate schedules for the SGD/Adam optimizers (#46).
 *
 * Stateless functions computing lr from a step counter, applied
 * by writing the optimizer's `lr` field before stepping:
 *
 *   float lr = 0.0f;
 *   nir_lr_step_decay(0.1f, 0.5f, 10, meta.step, &lr);
 *   sgd.lr = lr;
 *   nir_sgd_step(&sgd, weights, grad, count);
 *
 * The same write works for `nir_adam_t.lr`. There is no
 * scheduler struct to persist: the counter IS the checkpoint's
 * `step` (completed optimizer steps, 0 for a fresh run), so a
 * resumed run passes the loaded `meta.step` and continues the
 * curve exactly where it stopped.
 *
 * Formulas mirror torch.optim.lr_scheduler with last_epoch set
 * to `step` (step 0 always yields base_lr):
 *
 *   step_decay:  base_lr * gamma^(step / step_size)
 *   cosine:      eta_min + (base_lr - eta_min) *
 *                (1 + cos(pi * step / t_max)) / 2
 *   exponential: base_lr * gamma^step
 *
 * Cosine follows torch past t_max too: beyond the minimum the
 * curve rises again rather than clamping, so callers that want
 * a floor should stop the run (or the schedule) at t_max.
 */
#ifndef NIR_C_RUNTIME_SCHEDULE_H
#define NIR_C_RUNTIME_SCHEDULE_H

/* Step decay (torch StepLR). Needs base_lr and gamma finite
 * and >= 0, step_size >= 1. */
int nir_lr_step_decay(
    float base_lr, float gamma, unsigned long step_size,
    unsigned long step, float *lr
);

/* Cosine annealing (torch CosineAnnealingLR). Needs base_lr
 * and eta_min finite and >= 0, t_max >= 1. */
int nir_lr_cosine(
    float base_lr, float eta_min, unsigned long t_max,
    unsigned long step, float *lr
);

/* Exponential decay (torch ExponentialLR). Needs base_lr and
 * gamma finite and >= 0. */
int nir_lr_exponential(
    float base_lr, float gamma, unsigned long step, float *lr
);

#endif /* NIR_C_RUNTIME_SCHEDULE_H */
