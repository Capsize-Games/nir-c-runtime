#include "nir_c_runtime/ops.h"
#include "test_util.h"

static void test_bound_spatial_input_runs_identity_convolution(void) {
    float data[6] = {1, 2, 3, 4, 5, 6};
    size_t shape[3] = {1, 2, 3};
    nir_tensor_t input = {0};
    CHECK(nir_tensor_bind(&input, data, 6, shape, 3));
    float weight[1] = {1};
    nir_conv2d_params_t params = {
        1, 1, 1, 1, weight, NULL, 1, 1, 0, 0, 1, 1
    };
    nir_tensor_t output = nir_op_conv2d(&params, &input);
    CHECK(output.data != NULL && output.size == 6);
    for (size_t i = 0; i < 3; i++) CHECK(output.shape[i] == shape[i]);
    for (size_t i = 0; i < 6; i++) CHECK_NEAR(output.data[i], data[i], 1e-9);
    nir_tensor_free(&output);
    nir_tensor_free(&input);
    data[0] = 7; /* Bound storage remains caller-owned. */
    CHECK_NEAR(data[0], 7, 1e-9);
}

static void test_bound_shape_metadata(void) {
    float data[6] = {0};
    size_t shape[2] = {2, 3};
    nir_tensor_t tensor = {0};
    CHECK(nir_tensor_bind(&tensor, data, 6, shape, 2));
    CHECK(tensor.ndim == 2 && tensor.size == 6 && tensor.capacity == 6);
    CHECK(tensor.shape[0] == 2 && tensor.shape[1] == 3);
    CHECK(tensor.shape[2] == 0 && tensor.owns_data == 0);
    nir_tensor_free(&tensor);
}

static void test_rebinding_shape_may_alias_tensor(void) {
    size_t shape[2] = {2, 3};
    nir_tensor_t tensor = nir_tensor_alloc(shape, 2);
    CHECK(tensor.data != NULL && tensor.owns_data);
    float data[6] = {0};
    CHECK(nir_tensor_bind(&tensor, data, 6, tensor.shape, tensor.ndim));
    CHECK(tensor.shape[0] == 2 && tensor.shape[1] == 3);
    CHECK(tensor.data == data && tensor.owns_data == 0);
    nir_tensor_free(&tensor);
}

static void test_failed_bind_preserves_tensor(void) {
    size_t shape[2] = {2, 3};
    nir_tensor_t tensor = nir_tensor_alloc(shape, 2);
    float *original = tensor.data;
    float data[5] = {0};
    CHECK(!nir_tensor_bind(&tensor, data, 5, shape, 2));
    CHECK(tensor.data == original && tensor.owns_data);
    CHECK(tensor.shape[0] == 2 && tensor.shape[1] == 3);
    CHECK(tensor.ndim == 2 && tensor.size == 6);
    nir_tensor_free(&tensor);
}

static void test_bound_scalar_has_no_axes(void) {
    float data = 4;
    size_t unused[1] = {99};
    nir_tensor_t tensor = {0};
    CHECK(nir_tensor_bind(&tensor, &data, 1, unused, 0));
    CHECK(tensor.ndim == 0 && tensor.size == 1);
    for (size_t i = 0; i < NIR_TENSOR_MAX_DIMS; i++) {
        CHECK(tensor.shape[i] == 0);
    }
    nir_tensor_free(&tensor);
    CHECK_NEAR(data, 4, 1e-9);
}

int main(void) {
    test_bound_spatial_input_runs_identity_convolution();
    test_bound_shape_metadata();
    test_rebinding_shape_may_alias_tensor();
    test_failed_bind_preserves_tensor();
    test_bound_scalar_has_no_axes();
    return 0;
}
