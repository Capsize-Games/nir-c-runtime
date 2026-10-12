/* Minimal recursive-descent JSON parser.
 *
 * Scoped to what a NIR graph export actually contains: objects,
 * arrays (arbitrarily nested, for weight tensors), numbers, strings,
 * booleans, and null. No streaming, no comments, no trailing commas --
 * this reads a whole file into memory and parses it once at graph
 * load time, not on any hot path.
 */
#ifndef NIR_C_RUNTIME_JSON_H
#define NIR_C_RUNTIME_JSON_H

#include <stddef.h>

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type_t;

typedef struct json_value json_value_t;

struct json_value {
    json_type_t type;
    union {
        int boolean;
        double number;
        char *string;
        struct {
            json_value_t **items;
            size_t count;
        } array;
        struct {
            char **keys;
            json_value_t **values;
            size_t count;
        } object;
    } as;
};

/* Parse `text` (NUL-terminated) using JSON number and string grammar.
 * Unicode escapes are decoded to UTF-8; unpaired surrogates and embedded
 * NUL escapes are rejected because strings are exposed as C strings.
 * Nesting deeper than 256 containers is rejected. Returns NULL and sets
 * *error to a malloc'd message on failure; the caller owns and must free it. */
json_value_t *json_parse(const char *text, char **error);

/* Recursively free a value returned by json_parse. */
void json_free(json_value_t *value);

/* Return the member `key` of an object, or NULL if absent or `value`
 * is not an object. */
const json_value_t *json_object_get(
    const json_value_t *value, const char *key
);

/* Return element `index` of an array, or NULL if out of range or
 * `value` is not an array. */
const json_value_t *json_array_get(const json_value_t *value, size_t index);

/* Return the array length, or 0 if `value` is not an array. */
size_t json_array_length(const json_value_t *value);

/* Return the numeric value, or `fallback` if `value` is NULL or not a
 * number. */
double json_as_number(const json_value_t *value, double fallback);

/* Return the string value, or NULL if `value` is NULL or not a
 * string. Owned by the json_value_t; do not free separately. */
const char *json_as_string(const json_value_t *value);

#endif /* NIR_C_RUNTIME_JSON_H */
