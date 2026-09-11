/* Minimal test helpers shared across test executables: no framework,
 * just a fail-loud assert and a whole-file reader for fixtures. Each
 * test is its own executable; CTest checks the exit code.
 */
#ifndef NIR_C_RUNTIME_TEST_UTIL_H
#define NIR_C_RUNTIME_TEST_UTIL_H

#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf( \
                stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #cond \
            ); \
            exit(1); \
        } \
    } while (0)

#define CHECK_NEAR(actual, expected, tol) \
    do { \
        double _a = (actual), _e = (expected), _t = (tol); \
        double _d = _a > _e ? _a - _e : _e - _a; \
        if (_d > _t) { \
            fprintf( \
                stderr, \
                "CHECK_NEAR failed at %s:%d: %s (%g) vs %s (%g), diff %g > %g\n", \
                __FILE__, __LINE__, #actual, _a, #expected, _e, _d, _t \
            ); \
            exit(1); \
        } \
    } while (0)

/* Read a whole file into a NUL-terminated malloc'd buffer. Exits the
 * process on failure -- test fixtures are expected to exist. */
static inline char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "cannot open fixture: %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buffer = (char *)malloc((size_t)size + 1);
    if (buffer == NULL || fread(buffer, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "cannot read fixture: %s\n", path);
        exit(1);
    }
    buffer[size] = '\0';
    fclose(f);
    return buffer;
}

#endif /* NIR_C_RUNTIME_TEST_UTIL_H */
