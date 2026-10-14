/* Torch-free IDX loading: exact bytes, torch-matching normalize. */
#include <stdlib.h>
#include <string.h>

#include "nir_c_runtime/data.h"
#include "test_util.h"

static void write_temp(
    const char *name, const unsigned char *bytes, size_t n, char *path
) {
    const char *tmp = getenv("TMPDIR");
    if (tmp == NULL) {
        tmp = "/tmp";
    }
    sprintf(path, "%s/nir-data-test-%s", tmp, name);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(bytes, 1, n, f) == n);
    CHECK(fclose(f) == 0);
}

int main(void) {
    char *error = NULL;
    uint8_t *pixels = NULL;
    uint8_t *labels = NULL;
    size_t count = 0;
    size_t rows = 0;
    size_t cols = 0;
    size_t nlabels = 0;

    CHECK(nir_idx_read_images(
        "fixtures/mnist_tiny_images.idx", &pixels, &count, &rows, &cols,
        &error
    ));
    CHECK(error == NULL);
    CHECK(count == 3 && rows == 2 && cols == 2);
    /* tools/make_mnist_fixture.py wrote these exact bytes. */
    const uint8_t want_px[12] = {
        0, 85, 170, 255, 255, 170, 85, 0, 1, 2, 3, 4,
    };
    CHECK(memcmp(pixels, want_px, sizeof(want_px)) == 0);

    CHECK(nir_idx_read_labels(
        "fixtures/mnist_tiny_labels.idx", &labels, &nlabels, &error
    ));
    CHECK(error == NULL);
    CHECK(nlabels == 3);
    CHECK(labels[0] == 7 && labels[1] == 3 && labels[2] == 9);

    /* Normalize matches ToTensor+Normalize((0,),(1,)): v / 255. */
    float norm[12];
    nir_data_normalize(pixels, norm, 12);
    CHECK(norm[0] == 0.0f);
    CHECK_NEAR(norm[1], 85.0 / 255.0, 1e-7);
    CHECK_NEAR(norm[2], 170.0 / 255.0, 1e-7);
    CHECK(norm[3] == 1.0f);

    /* One sample lands in a (1, H, W) tensor frame. */
    nir_tensor_t frame = nir_data_frame(pixels, count, rows, cols, 2);
    CHECK(frame.data != NULL);
    CHECK(frame.ndim == 3);
    CHECK(frame.shape[0] == 1 && frame.shape[1] == 2 &&
        frame.shape[2] == 2);
    CHECK_NEAR(frame.data[0], 1.0 / 255.0, 1e-7);
    CHECK_NEAR(frame.data[3], 4.0 / 255.0, 1e-7);
    nir_tensor_free(&frame);
    nir_tensor_t bad = nir_data_frame(pixels, count, rows, cols, 3);
    CHECK(bad.data == NULL);
    free(pixels);
    free(labels);

    /* Malformed inputs fail with named errors, not crashes. */
    char path[512];
    const unsigned char bad_magic[16] = {
        0, 0, 8, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    };
    write_temp("bad-magic.idx", bad_magic, sizeof(bad_magic), path);
    pixels = NULL;
    CHECK(!nir_idx_read_images(path, &pixels, &count, &rows, &cols,
        &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;

    /* Declares one 1x1 image but the payload is empty. */
    const unsigned char truncated[16] = {
        0, 0, 8, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    };
    write_temp("truncated.idx", truncated, sizeof(truncated), path);
    CHECK(!nir_idx_read_images(path, &pixels, &count, &rows, &cols,
        &error));
    CHECK(error != NULL);
    free(error);
    error = NULL;

    /* Trailing junk after the declared payload is rejected. */
    const unsigned char ragged[18] = {
        0, 0, 8, 1, 0, 0, 0, 1, 9, 9,
    };
    write_temp("ragged.idx", ragged, 10, path);
    labels = NULL;
    CHECK(!nir_idx_read_labels(path, &labels, &nlabels, &error));
    CHECK(error != NULL);
    free(error);

    CHECK(!nir_idx_read_images(
        "fixtures/does-not-exist.idx", &pixels, &count, &rows, &cols,
        NULL
    ));
    return 0;
}
