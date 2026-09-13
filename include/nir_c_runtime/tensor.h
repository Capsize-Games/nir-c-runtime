/* A dense float tensor, up to 4 dimensions.
 *
 * Batch size 1 only -- this runtime targets single-sample edge
 * inference, so no batch axis is tracked. Shapes below batch are
 * (C, H, W) for spatial data, (F,) for flat feature data.
 */
#ifndef NIR_C_RUNTIME_TENSOR_H
#define NIR_C_RUNTIME_TENSOR_H

#include <stddef.h>

#define NIR_TENSOR_MAX_DIMS 3

typedef struct {
    float *data;
    size_t ndim;
    size_t shape[NIR_TENSOR_MAX_DIMS];
    size_t size; /* product of shape[0..ndim) */
    size_t capacity; /* number of writable float elements in `data` */
    int owns_data; /* nonzero when `nir_tensor_free` owns `data` */
} nir_tensor_t;

/* Allocate a zero-filled tensor of the given shape. `ndim` must be
 * <= NIR_TENSOR_MAX_DIMS. Returns a tensor with data == NULL on
 * allocation failure. */
nir_tensor_t nir_tensor_alloc(const size_t *shape, size_t ndim);

/* Return a deep copy of `src`. */
nir_tensor_t nir_tensor_clone(const nir_tensor_t *src);

/* Bind caller-owned storage to `tensor` with a logical shape. The storage
 * must remain valid until the tensor is rebound or freed. */
int nir_tensor_bind(
    nir_tensor_t *tensor, float *data, size_t capacity,
    const size_t *shape, size_t ndim
);

/* Copy a tensor into caller-provided storage without allocating. */
int nir_tensor_copy_into(nir_tensor_t *out, const nir_tensor_t *src);

/* Free `tensor`'s data and zero it out. Safe to call on an
 * already-freed or never-allocated tensor. */
void nir_tensor_free(nir_tensor_t *tensor);

/* Add `rhs` into `lhs` elementwise, in place. Equal-size tensors add
 * pointwise; a single-element `rhs` broadcasts to every element of
 * `lhs` (matching a 0-d scalar's broadcast against a real tensor,
 * which is exactly what an unprimed Delay node's output is -- see
 * runtime.c's sum_incoming). Growing a single-element `lhs` against a
 * larger `rhs` is not supported (unneeded by any graph this runtime
 * targets) and returns 0, as does any other size mismatch. */
int nir_tensor_add_inplace(nir_tensor_t *lhs, const nir_tensor_t *rhs);

#endif /* NIR_C_RUNTIME_TENSOR_H */
