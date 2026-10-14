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

static void test_linear_has_no_bias(void) {
    /* y = [[1,2],[3,4]] @ [1,1] = [3,7] */
    float weight[4] = {1, 2, 3, 4};
    nir_linear_params_t params = {2, 2, weight};
    size_t shape[1] = {2};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    input.data[0] = 1;
    input.data[1] = 1;

    nir_tensor_t output = nir_op_linear(&params, &input);
    CHECK(output.ndim == 1 && output.size == 2);
    CHECK_NEAR(output.data[0], 3.0, 1e-6);
    CHECK_NEAR(output.data[1], 7.0, 1e-6);

    nir_tensor_free(&input);
    nir_tensor_free(&output);
}

static void test_conv1d_matches_hand_computation(void) {
    /* x=[1,2,3,4], w=[1,0,-1], pad=1: out[i]=x[i-1]-x[i+1]. */
    float weight[3] = {1.0f, 0.0f, -1.0f};
    float bias[1] = {0.5f};
    nir_conv1d_params_t params = {1, 1, 3, weight, bias, 1, 1, 1};
    size_t shape[2] = {1, 4};
    nir_tensor_t input = nir_tensor_alloc(shape, 2);
    for (size_t i = 0; i < 4; i++) input.data[i] = (float)(i + 1);

    nir_tensor_t output = nir_op_conv1d(&params, &input);
    CHECK(output.ndim == 2 && output.shape[0] == 1 && output.shape[1] == 4);
    double want[4] = {-1.5, -1.5, -1.5, 3.5};
    for (size_t i = 0; i < 4; i++) {
        CHECK_NEAR(output.data[i], want[i], 1e-6);
    }
    nir_tensor_free(&output);

    /* Stride 2, no padding: x=[1,2,3,4], w=[1,1] -> [3,7]. */
    float stride_weight[2] = {1.0f, 1.0f};
    nir_conv1d_params_t strided = {1, 1, 2, stride_weight, NULL, 2, 0, 1};
    output = nir_op_conv1d(&strided, &input);
    CHECK(output.shape[0] == 1 && output.shape[1] == 2);
    CHECK_NEAR(output.data[0], 3.0, 1e-6);
    CHECK_NEAR(output.data[1], 7.0, 1e-6);

    nir_tensor_free(&input);
    nir_tensor_free(&output);
}

static void test_lif_integrates_then_resets(void) {
    nir_lif_params_t params = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    nir_tensor_t spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t stored = nir_tensor_alloc(shape, 1);
    double decay = exp(-1.0);

    /* First step from rest: membrane (1-decay) stays below threshold. */
    input.data[0] = 1.0f;
    CHECK(nir_op_lif_into(&params, &input, NULL, &spike, &stored));
    double membrane = 1.0 - decay;
    CHECK_NEAR(spike.data[0], 0.0, 1e-9);
    CHECK_NEAR(stored.data[0], membrane, 1e-6);

    /* Stronger drive crosses threshold: spike out, reset stored. */
    input.data[0] = 2.0f;
    CHECK(nir_op_lif_into(&params, &input, &stored, &spike, &stored));
    membrane = decay * membrane + (1.0 - decay) * 2.0;
    CHECK(membrane > 1.0);
    CHECK_NEAR(spike.data[0], 1.0, 1e-9);
    CHECK_NEAR(stored.data[0], 0.0, 1e-9);

    nir_tensor_free(&input);
    nir_tensor_free(&spike);
    nir_tensor_free(&stored);
}

static void test_if_has_no_leak(void) {
    nir_if_params_t params = {2.0f, 2.0f, 0.5f};
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    nir_tensor_t spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t stored = nir_tensor_alloc(shape, 1);

    /* 0 + 2*1 = 2 equals the threshold: strict, so no spike. */
    input.data[0] = 1.0f;
    CHECK(nir_op_if_into(&params, &input, NULL, &spike, &stored));
    CHECK_NEAR(spike.data[0], 0.0, 1e-9);
    CHECK_NEAR(stored.data[0], 2.0, 1e-9);

    /* 2 + 2*0.5 = 3 crosses: spike out, v_reset stored. */
    input.data[0] = 0.5f;
    CHECK(nir_op_if_into(&params, &input, &stored, &spike, &stored));
    CHECK_NEAR(spike.data[0], 1.0, 1e-9);
    CHECK_NEAR(stored.data[0], 0.5, 1e-9);

    nir_tensor_free(&input);
    nir_tensor_free(&spike);
    nir_tensor_free(&stored);
}

static void test_cuba_integrates_two_states(void) {
    nir_cuba_params_t params = {1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    size_t shape[1] = {1};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    nir_tensor_t spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t syn = nir_tensor_alloc(shape, 1);
    nir_tensor_t stored = nir_tensor_alloc(shape, 1);
    double decay = exp(-1.0);

    /* From rest with x=1: syn=(1-decay), mem=(1-decay)*syn. */
    input.data[0] = 1.0f;
    CHECK(nir_op_cuba_into(
        &params, &input, NULL, NULL, &spike, &syn, &stored
    ));
    double syn_want = 1.0 - decay;
    double mem_want = (1.0 - decay) * syn_want;
    CHECK_NEAR(syn.data[0], syn_want, 1e-6);
    CHECK_NEAR(spike.data[0], 0.0, 1e-9);
    CHECK_NEAR(stored.data[0], mem_want, 1e-6);

    /* With x=0 both states decay without spiking. */
    input.data[0] = 0.0f;
    CHECK(nir_op_cuba_into(
        &params, &input, &syn, &stored, &spike, &syn, &stored
    ));
    syn_want = decay * syn_want;
    mem_want = decay * mem_want + (1.0 - decay) * syn_want;
    CHECK_NEAR(syn.data[0], syn_want, 1e-6);
    CHECK_NEAR(spike.data[0], 0.0, 1e-9);
    CHECK_NEAR(stored.data[0], mem_want, 1e-6);

    nir_tensor_free(&input);
    nir_tensor_free(&spike);
    nir_tensor_free(&syn);
    nir_tensor_free(&stored);
}

static void test_spiking_ops_reject_bad_geometry(void) {
    size_t shape[1] = {2};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    nir_tensor_t spike = nir_tensor_alloc(shape, 1);
    nir_tensor_t stored = nir_tensor_alloc(shape, 1);
    nir_lif_params_t lif = {1.0f, 1.0f, 0.0f, 1.0f, 0.0f};

    /* The two outputs must not alias each other, even exactly. */
    CHECK(!nir_op_lif_into(&lif, &input, NULL, &spike, &spike));
    nir_tensor_t aliased = {0};
    CHECK(nir_tensor_bind(&aliased, spike.data, 2, shape, 1));
    CHECK(!nir_op_lif_into(&lif, &input, NULL, &spike, &aliased));

    /* A nonpositive tau is rejected. */
    nir_lif_params_t bad_tau = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    CHECK(!nir_op_lif_into(&bad_tau, &input, NULL, &spike, &stored));

    /* Mismatched previous state is rejected. */
    size_t one[1] = {1};
    nir_tensor_t short_prev = nir_tensor_alloc(one, 1);
    CHECK(!nir_op_lif_into(&lif, &input, &short_prev, &spike, &stored));

    /* NULL inputs fail cleanly. */
    CHECK(!nir_op_lif_into(&lif, NULL, NULL, &spike, &stored));
    nir_if_params_t if_ = {1.0f, 1.0f, 0.0f};
    CHECK(!nir_op_if_into(&if_, NULL, NULL, &spike, &stored));
    nir_tensor_t syn = nir_tensor_alloc(shape, 1);
    nir_cuba_params_t cuba = {1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    CHECK(!nir_op_cuba_into(
        &cuba, NULL, NULL, NULL, &spike, &syn, &stored
    ));
    /* Split synaptic/membrane state is rejected: they prime together. */
    CHECK(!nir_op_cuba_into(
        &cuba, &input, NULL, &stored, &spike, &syn, &stored
    ));
    CHECK(!nir_op_cuba_into(
        &cuba, &input, &syn, NULL, &spike, &syn, &stored
    ));

    nir_tensor_free(&short_prev);
    nir_tensor_free(&aliased);
    nir_tensor_free(&syn);
    nir_tensor_free(&input);
    nir_tensor_free(&spike);
    nir_tensor_free(&stored);
}

static void test_ops_reject_malformed_shapes_and_geometry(void) {
    float data[4] = {1, 2, 3, 4};
    size_t wrong_shape[2] = {2, 2};
    nir_tensor_t wrong_rank = {0};
    CHECK(nir_tensor_bind(&wrong_rank, data, 4, wrong_shape, 2));
    float weight[1] = {1};
    nir_conv2d_params_t conv = {1, 1, 1, 1, weight, NULL, 1, 1, 0, 0, 1, 1};
    nir_tensor_t result = nir_op_conv2d(&conv, &wrong_rank);
    CHECK(result.data == NULL);
    nir_pool2d_params_t pool = {2, 2, 0, 1, 0, 0};
    result = nir_op_avgpool2d(&pool, &wrong_rank);
    CHECK(result.data == NULL);
    nir_tensor_free(&wrong_rank);
}

static void test_ops_reject_aliasing_and_mismatched_state(void) {
    size_t shape[1] = {2};
    nir_tensor_t input = nir_tensor_alloc(shape, 1);
    input.data[0] = 2;
    input.data[1] = 3;
    nir_scale_params_t scale = {2};
    nir_tensor_t output = {0};
    CHECK(nir_tensor_bind(&output, input.data, input.capacity, shape, 1));
    CHECK(nir_op_scale_into(&scale, &input, &output));
    CHECK(input.data[0] == 4 && input.data[1] == 6);
    float shifted[3] = {2, 3, 0};
    CHECK(nir_tensor_bind(&output, shifted + 1, 2, shape, 1));
    nir_tensor_t shifted_input = {0};
    CHECK(nir_tensor_bind(&shifted_input, shifted, 3, shape, 1));
    CHECK(!nir_op_scale_into(&scale, &shifted_input, &output));
    CHECK(shifted[0] == 2 && shifted[1] == 3);

    nir_li_params_t li = {1, 1, 0};
    size_t wrong_shape[1] = {1};
    nir_tensor_t prev = nir_tensor_alloc(wrong_shape, 1);
    CHECK(!nir_op_li_into(&li, &input, &prev, &output));
    nir_tensor_free(&prev);
    nir_tensor_free(&shifted_input);
    nir_tensor_free(&output);
    nir_tensor_free(&input);
}

static void test_convenience_ops_accept_null_inputs_safely(void) {
    nir_tensor_t out = nir_op_affine(NULL, NULL);
    CHECK(out.data == NULL);
    out = nir_op_linear(NULL, NULL);
    CHECK(out.data == NULL);
    out = nir_op_conv1d(NULL, NULL);
    CHECK(out.data == NULL);
    out = nir_op_flatten(NULL);
    CHECK(out.data == NULL);
    out = nir_op_li(NULL, NULL, NULL);
    CHECK(out.data == NULL);
    out = nir_op_threshold(NULL, NULL);
    CHECK(out.data == NULL);
    out = nir_op_scale(NULL, NULL);
    CHECK(out.data == NULL);
}

static void test_into_ops_accept_same_tensor_alias(void) {
    size_t shape[2] = {1, 2};
    nir_tensor_t tensor = nir_tensor_alloc(shape, 2);
    tensor.data[0] = 2;
    tensor.data[1] = 3;
    nir_scale_params_t scale = {2};
    CHECK(nir_op_scale_into(&scale, &tensor, &tensor));
    CHECK(tensor.data[0] == 4 && tensor.data[1] == 6);

    nir_threshold_params_t threshold = {4};
    CHECK(nir_op_threshold_into(&threshold, &tensor, &tensor));
    CHECK(tensor.data[0] == 0 && tensor.data[1] == 1);

    nir_li_params_t li = {1, 1, 0};
    CHECK(nir_op_li_into(&li, &tensor, NULL, &tensor));
    CHECK(tensor.data[0] == 0 && tensor.data[1] > 0);
    nir_tensor_free(&tensor);

    size_t spatial_shape[2] = {2, 2};
    tensor = nir_tensor_alloc(spatial_shape, 2);
    for (size_t i = 0; i < 4; i++) tensor.data[i] = (float)(i + 1);
    CHECK(nir_op_flatten_into(&tensor, &tensor));
    CHECK(tensor.ndim == 1 && tensor.shape[0] == 4 && tensor.size == 4);
    CHECK(tensor.data[0] == 1 && tensor.data[3] == 4);
    nir_tensor_free(&tensor);
}

static void test_affine_alias_failure_preserves_tensor(void) {
    size_t shape[1] = {2};
    nir_tensor_t tensor = nir_tensor_alloc(shape, 1);
    tensor.data[0] = 5;
    tensor.data[1] = 7;
    float weight[2] = {1, 1};
    float bias[1] = {0};
    nir_affine_params_t params = {1, 2, weight, bias};
    CHECK(!nir_op_affine_into(&params, &tensor, &tensor));
    CHECK(tensor.ndim == 1 && tensor.shape[0] == 2 && tensor.size == 2 &&
        tensor.data[0] == 5 && tensor.data[1] == 7);

    nir_tensor_t overlapping_output = {0};
    CHECK(nir_tensor_bind(&overlapping_output, tensor.data, 2, shape, 1));
    CHECK(!nir_op_affine_into(&params, &tensor, &overlapping_output));
    CHECK(overlapping_output.ndim == 1 && overlapping_output.shape[0] == 2 &&
        overlapping_output.size == 2 && tensor.data[0] == 5 &&
        tensor.data[1] == 7);
    nir_tensor_free(&overlapping_output);
    nir_tensor_free(&tensor);
}

int main(void) {
    test_affine();
    test_threshold_is_strict();
    test_scale();
    test_li_zero_order_hold();
    test_conv2d_identity_kernel();
    test_avgpool_and_sumpool_relationship();
    test_flatten_preserves_data_order();
    test_linear_has_no_bias();
    test_conv1d_matches_hand_computation();
    test_lif_integrates_then_resets();
    test_if_has_no_leak();
    test_cuba_integrates_two_states();
    test_spiking_ops_reject_bad_geometry();
    test_ops_reject_malformed_shapes_and_geometry();
    test_ops_reject_aliasing_and_mismatched_state();
    test_convenience_ops_accept_null_inputs_safely();
    test_into_ops_accept_same_tensor_alias();
    test_affine_alias_failure_preserves_tensor();
    return 0;
}
