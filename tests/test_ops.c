#include <math.h>

#include "nir_c_runtime/ops.h"
#include "test_util.h"

static void test_affine(void) {
    /* y = [[1,2],[3,4]] @ [1,1] + [0,1] = [3,8] */
    float weight[4] = {1, 2, 3, 4};
    float bias[2] = {0, 1};
    nir_affine_params_t params = {2, 2, weight, bias};
    size_t shape[1] = {2};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    input.data[0] = 1;
    input.data[1] = 1;

    nir_tensor_t output = nir_op_affine(&params, &input);
    CHECK_NEAR(output.data[0], 3.0, 1e-6);
    CHECK_NEAR(output.data[1], 8.0, 1e-6);

    nir_tensor_free(&input);
    nir_tensor_free(&output);
}

static void test_threshold_is_strict(void) {
    nir_threshold_params_t params = {1.0f};
    size_t shape[1] = {3};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    input.data[0] = 0.5f;
    input.data[1] = 1.0f;  /* equal to threshold: must NOT fire */
    input.data[2] = 1.5f;

    nir_tensor_t output = nir_op_threshold(&params, &input);
    CHECK_NEAR(output.data[0], 0.0, 1e-9);
    CHECK_NEAR(output.data[1], 0.0, 1e-9);
    CHECK_NEAR(output.data[2], 1.0, 1e-9);

    nir_tensor_free(&input);
    nir_tensor_free(&output);
}

static void test_scale(void) {
    nir_scale_params_t params = {-2.0f};
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    input.data[0] = 3.0f;
    nir_tensor_t output = nir_op_scale(&params, &input);
    CHECK_NEAR(output.data[0], -6.0, 1e-9);
    nir_tensor_free(&input);
    nir_tensor_free(&output);
}

static void test_li_zero_order_hold(void) {
    /* v_leak=0, r=1, tau=1: decay = exp(-1); v = decay*v_prev + (1-decay)*I */
    nir_li_params_t params = {1.0f, 1.0f, 0.0f};
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    input.data[0] = 1.0f;
    nir_tensor_t v_prev = nir_tensor_alloc(shape, 1);
    v_prev.data[0] = 0.5f;

    nir_tensor_t output = nir_op_li(&params, &input, &v_prev);
    double decay = exp(-1.0);
    double expected = decay * 0.5 + (1.0 - decay) * 1.0;
    CHECK_NEAR(output.data[0], expected, 1e-6);

    /* NULL v_prev behaves as zero */
    nir_tensor_t first_step = nir_op_li(&params, &input, NULL);
    CHECK_NEAR(first_step.data[0], (1.0 - decay) * 1.0, 1e-6);

    nir_tensor_free(&input);
    nir_tensor_free(&v_prev);
    nir_tensor_free(&output);
    nir_tensor_free(&first_step);
}

static void test_conv2d_identity_kernel(void) {
    /* A single 1x1 identity-weight kernel over a 1-channel 2x2 image
     * must reproduce the input exactly. */
    float weight[1] = {1.0f};
    nir_conv2d_params_t params = {
        1, 1, 1, 1, weight, NULL, 1, 1, 0, 0, 1, 1
    };
    size_t shape[3] = {1, 2, 2};
    nir_tensor_t input = nir_tensor_alloc(shape, 3);
    for (size_t i = 0; i < 4; i++) input.data[i] = (float)(i + 1);

    nir_tensor_t output = nir_op_conv2d(&params, &input);
    CHECK(output.shape[0] == 1 && output.shape[1] == 2 && output.shape[2] == 2);
    for (size_t i = 0; i < 4; i++) {
        CHECK_NEAR(output.data[i], input.data[i], 1e-9);
    }

    nir_tensor_free(&input);
    nir_tensor_free(&output);
}

static void test_avgpool_and_sumpool_relationship(void) {
    /* A 4x4 all-ones image, 2x2 pool, stride 2: avg=1 everywhere,
     * sum=4 everywhere (2*2 window area). */
    nir_pool2d_params_t params = {2, 2, 2, 2, 0, 0};
    size_t shape[3] = {1, 4, 4};
    nir_tensor_t input = nir_tensor_alloc(shape, 3);
    for (size_t i = 0; i < 16; i++) input.data[i] = 1.0f;

    nir_tensor_t avg = nir_op_avgpool2d(&params, &input);
    nir_tensor_t sum = nir_op_sumpool2d(&params, &input);
    CHECK(avg.size == 4 && sum.size == 4);
    for (size_t i = 0; i < avg.size; i++) {
        CHECK_NEAR(avg.data[i], 1.0, 1e-6);
        CHECK_NEAR(sum.data[i], 4.0, 1e-6);
    }

    nir_tensor_free(&input);
    nir_tensor_free(&avg);
    nir_tensor_free(&sum);
}

static void test_flatten_preserves_data_order(void) {
    size_t shape[3] = {2, 1, 2};
    nir_tensor_t input = nir_tensor_alloc(shape, 3);
    for (size_t i = 0; i < 4; i++) input.data[i] = (float)i;
    nir_tensor_t output = nir_op_flatten(&input);
    CHECK(output.ndim == 1 && output.size == 4);
    for (size_t i = 0; i < 4; i++) CHECK_NEAR(output.data[i], (double)i, 1e-9);
    nir_tensor_free(&input);
    nir_tensor_free(&output);
}

int main(void) {
    test_affine();
    test_threshold_is_strict();
    test_scale();
    test_li_zero_order_hold();
    test_conv2d_identity_kernel();
    test_avgpool_and_sumpool_relationship();
    test_flatten_preserves_data_order();
    return 0;
}
