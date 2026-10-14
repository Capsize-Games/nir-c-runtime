/* Torch-free dataset loading (#27).
 *
 * Minimal IDX-format MNIST reader (no new dependencies): the magic
 * numbers, big-endian headers, and row-major u8 pixels of the
 * original MNIST files. Normalization matches spikeforge's torch
 * transform exactly (`datasets.transform`: Grayscale, Resize,
 * ToTensor, Normalize((0,), (1,))) -- ToTensor scales bytes by
 * 1/255 and the Normalize is an identity -- so C decodes each
 * pixel as v / 255.0f. Multi-byte samples land in `nir_tensor_t`
 * frames (batch-1, one tensor per sample); batching stays the
 * caller's loop until the tensor ABI grows a batch axis.
 */
#ifndef NIR_C_RUNTIME_DATA_H
#define NIR_C_RUNTIME_DATA_H

#include <stddef.h>
#include <stdint.h>

#include "nir_c_runtime/tensor.h"

/* Decode `n` IDX pixels into normalized floats (v / 255). */
void nir_data_normalize(const uint8_t *pixels, float *out, size_t n);

/* Read an IDX image file (magic 2051) into a malloc'd row-major
 * buffer of `*count * *rows * *cols` bytes. Dimensions are capped
 * (count <= 2^24, rows/cols <= 1024) and the file must hold exactly
 * the declared payload. Returns 0 and sets *error on failure. */
int nir_idx_read_images(
    const char *path, uint8_t **pixels, size_t *count, size_t *rows,
    size_t *cols, char **error
);

/* Read an IDX label file (magic 2049) into a malloc'd buffer of
 * `*count` bytes. Returns 0 and sets *error on failure. */
int nir_idx_read_labels(
    const char *path, uint8_t **labels, size_t *count, char **error
);

/* Normalize sample `index` of an image buffer into a fresh (C=1, H,
 * W) tensor. Returns a tensor with data == NULL on bad sizes or
 * allocation failure. */
nir_tensor_t nir_data_frame(
    const uint8_t *pixels, size_t count, size_t rows, size_t cols,
    size_t index
);

#endif /* NIR_C_RUNTIME_DATA_H */
