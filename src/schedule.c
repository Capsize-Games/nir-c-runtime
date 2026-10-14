#include "nir_c_runtime/schedule.h"

#include <math.h>
#include <stddef.h>

static int check_base(float base_lr, const float *lr) {
    return lr != NULL && isfinite(base_lr) && base_lr >= 0.0f;
}

int nir_lr_step_decay(
    float base_lr, float gamma, unsigned long step_size,
    unsigned long step, float *lr
) {
    if (!check_base(base_lr, lr) || !isfinite(gamma) ||
        gamma < 0.0f || step_size == 0) {
        return 0;
    }
    double factor = pow(
        (double)gamma, (double)(step / step_size)
    );
    *lr = (float)((double)base_lr * factor);
    return 1;
}

int nir_lr_cosine(
    float base_lr, float eta_min, unsigned long t_max,
    unsigned long step, float *lr
) {
    if (!check_base(base_lr, lr) || !isfinite(eta_min) ||
        eta_min < 0.0f || t_max == 0) {
        return 0;
    }
    double pi = acos(-1.0);
    double half = (1.0 + cos(pi * (double)step / (double)t_max));
    half /= 2.0;
    *lr = (float)((double)eta_min +
        ((double)base_lr - (double)eta_min) * half);
    return 1;
}

int nir_lr_exponential(
    float base_lr, float gamma, unsigned long step, float *lr
) {
    if (!check_base(base_lr, lr) || !isfinite(gamma) ||
        gamma < 0.0f) {
        return 0;
    }
    *lr = (float)((double)base_lr * pow((double)gamma,
        (double)step));
    return 1;
}
