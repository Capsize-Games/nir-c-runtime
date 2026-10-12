#include "nir_c_runtime/ops.h"
#include "test_util.h"

#include <stdint.h>

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

static void test_owned_rebind_preserves_ownership_and_rejects_interior(void) {
    size_t shape[1] = {3};
    nir_tensor_t tensor = nir_tensor_alloc(shape, 1);
    CHECK(tensor.data != NULL && tensor.owns_data);
    float *storage = tensor.data;
    CHECK(!nir_tensor_bind(&tensor, storage + 1, 3, shape, 1));
    CHECK(tensor.data == storage && tensor.owns_data && tensor.size == 3);
    CHECK(nir_tensor_bind(&tensor, storage, 3, shape, 1));
    CHECK(tensor.data == storage && tensor.owns_data);
    nir_tensor_free(&tensor);
    CHECK(tensor.data == NULL);
}

static void test_owned_rebind_cannot_claim_more_capacity(void) {
    size_t one_shape[1] = {1};
    nir_tensor_t tensor = nir_tensor_alloc(one_shape, 1);
    CHECK(tensor.data != NULL && tensor.capacity == 1 && tensor.owns_data);
    float *storage = tensor.data;
    storage[0] = 17;
    size_t two_shape[1] = {2};
    CHECK(!nir_tensor_bind(&tensor, storage, 2, two_shape, 1));
    CHECK(tensor.data == storage && tensor.capacity == 1 && tensor.size == 1 &&
        tensor.owns_data && tensor.data[0] == 17);
    CHECK(nir_tensor_bind(&tensor, storage, 1, one_shape, 1));
    CHECK(tensor.data == storage && tensor.capacity == 1 && tensor.owns_data &&
        tensor.data[0] == 17);
    nir_tensor_free(&tensor);
}

static void test_bind_rejects_product_overflow_and_preserves_destination(void) {
    size_t original_shape[1] = {2};
    nir_tensor_t tensor = nir_tensor_alloc(original_shape, 1);
    float *original = tensor.data;
    size_t overflow_shape[2] = {SIZE_MAX, SIZE_MAX};
    float external[1] = {9};
    CHECK(!nir_tensor_bind(&tensor, external, 1, overflow_shape, 2));
    CHECK(tensor.data == original && tensor.owns_data && tensor.size == 2);
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

static void test_alloc_rejects_null_shape_and_overflow(void) {
    nir_tensor_t missing = nir_tensor_alloc(NULL, 1);
    CHECK(missing.data == NULL && missing.ndim == 0);
    size_t overflow_shape[2] = {SIZE_MAX, 2};
    nir_tensor_t overflow = nir_tensor_alloc(overflow_shape, 2);
    CHECK(overflow.data == NULL && overflow.ndim == 0);
}

static void test_copy_handles_overlap_and_rejects_bad_metadata(void) {
    float data[5] = {1, 2, 3, 4, 5};
    size_t shape[1] = {4};
    nir_tensor_t src = {0}, dst = {0};
    CHECK(nir_tensor_bind(&src, data, 5, shape, 1));
    size_t dst_shape[1] = {4};
    CHECK(nir_tensor_bind(&dst, data + 1, 4, dst_shape, 1));
    CHECK(nir_tensor_copy_into(&dst, &src));
    CHECK(data[1] == 1 && data[2] == 2 && data[3] == 3 && data[4] == 4);
    src.size = SIZE_MAX;
    CHECK(!nir_tensor_copy_into(&dst, &src));
    nir_tensor_free(&src);
    nir_tensor_free(&dst);

    float output_data[4] = {0};
    nir_tensor_t raw_output = {0};
    raw_output.data = output_data;
    raw_output.capacity = 4;
    size_t source_shape[2] = {2, 2};
    float source_data[4] = {5, 6, 7, 8};
    nir_tensor_t source = {0};
    CHECK(nir_tensor_bind(&source, source_data, 4, source_shape, 2));
    CHECK(nir_tensor_copy_into(&raw_output, &source));
    CHECK(raw_output.ndim == 2 && raw_output.shape[0] == 2 &&
        raw_output.shape[1] == 2 && output_data[3] == 8);
    nir_tensor_free(&source);
}

static void test_add_rejects_partial_overlap_and_snapshots_scalar(void) {
    float overlap_data[5] = {1, 2, 3, 4, 5};
    size_t shape[1] = {4};
    nir_tensor_t lhs = {0}, rhs = {0};
    CHECK(nir_tensor_bind(&lhs, overlap_data, 5, shape, 1));
    CHECK(nir_tensor_bind(&rhs, overlap_data + 1, 4, shape, 1));
    CHECK(!nir_tensor_add_inplace(&lhs, &rhs));
    CHECK(overlap_data[0] == 1 && overlap_data[1] == 2);

    size_t scalar_shape[1] = {1};
    nir_tensor_t scalar = {0};
    CHECK(nir_tensor_bind(&scalar, overlap_data + 1, 1, scalar_shape, 1));
    CHECK(nir_tensor_add_inplace(&lhs, &scalar));
    CHECK(overlap_data[0] == 3 && overlap_data[1] == 4 &&
        overlap_data[2] == 5 && overlap_data[3] == 6);
    nir_tensor_free(&lhs);
    nir_tensor_free(&rhs);
    nir_tensor_free(&scalar);
}

int main(void) {
    test_bound_spatial_input_runs_identity_convolution();
    test_bound_shape_metadata();
    test_rebinding_shape_may_alias_tensor();
    test_failed_bind_preserves_tensor();
    test_owned_rebind_preserves_ownership_and_rejects_interior();
    test_owned_rebind_cannot_claim_more_capacity();
    test_bind_rejects_product_overflow_and_preserves_destination();
    test_bound_scalar_has_no_axes();
    test_alloc_rejects_null_shape_and_overflow();
    test_copy_handles_overlap_and_rejects_bad_metadata();
    test_add_rejects_partial_overlap_and_snapshots_scalar();
    return 0;
}
