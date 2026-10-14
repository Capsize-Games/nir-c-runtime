#include "nir_c_runtime/data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IDX_MAGIC_IMAGES 2051u
#define IDX_MAGIC_LABELS 2049u
#define IDX_MAX_COUNT ((size_t)1 << 24)
#define IDX_MAX_DIM 1024u

void nir_data_normalize(const uint8_t *pixels, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = (float)pixels[i] / 255.0f;
    }
}

static void set_error(char **error, const char *message) {
    if (error == NULL || *error != NULL) {
        return;
    }
    size_t length = strlen(message) + 1;
    *error = (char *)malloc(length);
    if (*error != NULL) {
        memcpy(*error, message, length);
    }
}

static int read_u32be(FILE *f, uint32_t *out) {
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, f) != 4) {
        return 0;
    }
    *out = ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
        ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
    return 1;
}

/* Read the whole payload after the header; the file must end
 * exactly at the declared size (no truncation, no trailing junk). */
static uint8_t *read_payload(FILE *f, size_t size) {
    uint8_t *buffer = (uint8_t *)malloc(size == 0 ? 1 : size);
    if (buffer == NULL) {
        return NULL;
    }
    if (size > 0 && fread(buffer, 1, size, f) != size) {
        free(buffer);
        return NULL;
    }
    if (fgetc(f) != EOF) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

int nir_idx_read_images(
    const char *path, uint8_t **pixels, size_t *count, size_t *rows,
    size_t *cols, char **error
) {
    if (path == NULL || pixels == NULL || count == NULL || rows == NULL ||
        cols == NULL) {
        set_error(error, "IDX image read needs out-parameters");
        return 0;
    }
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        set_error(error, "cannot open IDX image file");
        return 0;
    }
    uint32_t magic = 0, n = 0, h = 0, w = 0;
    int ok = read_u32be(f, &magic) && read_u32be(f, &n) &&
        read_u32be(f, &h) && read_u32be(f, &w);
    if (!ok || magic != IDX_MAGIC_IMAGES || n == 0 || n > IDX_MAX_COUNT ||
        h == 0 || h > IDX_MAX_DIM || w == 0 || w > IDX_MAX_DIM) {
        fclose(f);
        set_error(error, "bad IDX image header");
        return 0;
    }
    size_t total = (size_t)n * (size_t)h * (size_t)w;
    uint8_t *buffer = read_payload(f, total);
    fclose(f);
    if (buffer == NULL) {
        set_error(error, "IDX image payload is truncated or ragged");
        return 0;
    }
    *pixels = buffer;
    *count = n;
    *rows = h;
    *cols = w;
    return 1;
}

int nir_idx_read_labels(
    const char *path, uint8_t **labels, size_t *count, char **error
) {
    if (path == NULL || labels == NULL || count == NULL) {
        set_error(error, "IDX label read needs out-parameters");
        return 0;
    }
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        set_error(error, "cannot open IDX label file");
        return 0;
    }
    uint32_t magic = 0, n = 0;
    int ok = read_u32be(f, &magic) && read_u32be(f, &n);
    if (!ok || magic != IDX_MAGIC_LABELS || n == 0 ||
        n > IDX_MAX_COUNT) {
        fclose(f);
        set_error(error, "bad IDX label header");
        return 0;
    }
    uint8_t *buffer = read_payload(f, n);
    fclose(f);
    if (buffer == NULL) {
        set_error(error, "IDX label payload is truncated or ragged");
        return 0;
    }
    *labels = buffer;
    *count = n;
    return 1;
}

nir_tensor_t nir_data_frame(
    const uint8_t *pixels, size_t count, size_t rows, size_t cols,
    size_t index
) {
    nir_tensor_t empty;
    memset(&empty, 0, sizeof(empty));
    if (pixels == NULL || count == 0 || rows == 0 || cols == 0 ||
        index >= count) {
        return empty;
    }
    size_t shape[3] = {1, rows, cols};
    nir_tensor_t frame = nir_tensor_alloc(shape, 3);
    if (frame.data == NULL) {
        return frame;
    }
    nir_data_normalize(pixels + index * rows * cols, frame.data, frame.size);
    return frame;
}
