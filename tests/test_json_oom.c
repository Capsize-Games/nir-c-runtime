#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void *pointer;
    size_t size;
} allocation_t;

static allocation_t allocations[8192];
static size_t allocation_count;
static size_t calls;
static size_t fail_at;
static int fail_paired_realloc;
static int paired_realloc_triggered;
static int previous_realloc_nonnull;
static size_t previous_realloc_size;

static int should_fail(void) {
    calls++;
    return fail_at != 0 && calls == fail_at;
}

static void remember(void *pointer, size_t size) {
    if (pointer == NULL) return;
    for (size_t i = 0; i < allocation_count; i++) {
        if (allocations[i].pointer == NULL) {
            allocations[i] = (allocation_t){pointer, size};
            return;
        }
    }
    if (allocation_count < sizeof(allocations) / sizeof(allocations[0]))
        allocations[allocation_count++] = (allocation_t){pointer, size};
}

static size_t forget(void *pointer) {
    for (size_t i = 0; i < allocation_count; i++) {
        if (allocations[i].pointer == pointer) {
            size_t size = allocations[i].size;
            allocations[i].pointer = NULL;
            return size;
        }
    }
    return 0;
}

static void *test_malloc(size_t size) {
    if (should_fail()) return NULL;
    void *pointer = malloc(size);
    remember(pointer, size);
    return pointer;
}

static void *test_calloc(size_t count, size_t size) {
    if (should_fail()) return NULL;
    void *pointer = calloc(count, size);
    remember(pointer, count * size);
    return pointer;
}

static void *test_realloc(void *pointer, size_t size) {
    int fail = should_fail();
    if (!fail && fail_paired_realloc && pointer != NULL &&
        previous_realloc_nonnull && previous_realloc_size == size) {
        fail = 1;
        fail_paired_realloc = 0;
        paired_realloc_triggered = 1;
    }
    if (fail) {
        previous_realloc_nonnull = 0;
        return NULL;
    }
    if (pointer == NULL) {
        previous_realloc_nonnull = 0;
        void *fresh = test_malloc(size);
        previous_realloc_size = size;
        return fresh;
    }
    size_t old_size = forget(pointer);
    void *grown = malloc(size);
    if (grown == NULL) {
        remember(pointer, old_size);
        previous_realloc_nonnull = 0;
        return NULL;
    }
    memcpy(grown, pointer, old_size < size ? old_size : size);
    free(pointer);
    remember(grown, size);
    previous_realloc_nonnull = 1;
    previous_realloc_size = size;
    return grown;
}

static void test_free(void *pointer) {
    if (pointer != NULL) forget(pointer);
    free(pointer);
}

#define malloc test_malloc
#define calloc test_calloc
#define realloc test_realloc
#define free test_free
#include "../src/json.c"
#undef malloc
#undef calloc
#undef realloc
#undef free

static size_t live_allocations(void) {
    size_t live = 0;
    for (size_t i = 0; i < allocation_count; i++)
        if (allocations[i].pointer != NULL) live++;
    return live;
}

int main(void) {
    const char *input =
        "{\"first\":[1,2,3],\"second\":{\"nested\":true},\"third\":null}";
    char *error = NULL;

    calls = 0;
    fail_at = 0;
    json_value_t *value = json_parse(input, &error);
    if (value == NULL || error != NULL) return 1;
    size_t total_calls = calls;
    json_free(value);
    if (live_allocations() != 0) return 2;

    calls = 0;
    fail_at = 0;
    fail_paired_realloc = 1;
    paired_realloc_triggered = 0;
    previous_realloc_nonnull = 0;
    value = json_parse(input, &error);
    if (value != NULL || !paired_realloc_triggered) return 5;
    test_free(error);
    if (live_allocations() != 0) return 6;

    for (size_t failure = 1; failure <= total_calls; failure++) {
        calls = 0;
        fail_at = failure;
        previous_realloc_nonnull = 0;
        value = json_parse(input, &error);
        if (value != NULL) {
            json_free(value);
            test_free(error);
            return 3;
        }
        test_free(error);
        if (live_allocations() != 0) return 4;
    }
    printf("swept %zu allocation failure points with no leaks\n", total_calls);
    return 0;
}
