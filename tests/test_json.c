#include <string.h>

#include "nir_c_runtime/json.h"
#include "test_util.h"

static void test_scalars(void) {
    char *error = NULL;
    json_value_t *n = json_parse("42.5", &error);
    CHECK(error == NULL);
    CHECK(n->type == JSON_NUMBER);
    CHECK_NEAR(json_as_number(n, 0.0), 42.5, 1e-9);
    json_free(n);

    json_value_t *s = json_parse("\"hi\\n\\\"there\\\"\"", &error);
    CHECK(error == NULL);
    CHECK(s->type == JSON_STRING);
    CHECK(strcmp(json_as_string(s), "hi\n\"there\"") == 0);
    json_free(s);
}

static void test_array_and_object(void) {
    char *error = NULL;
    json_value_t *v = json_parse(
        "{\"a\": [1, 2, [3, 4]], \"b\": true, \"c\": null}", &error
    );
    CHECK(error == NULL);
    CHECK(v->type == JSON_OBJECT);
    const json_value_t *a = json_object_get(v, "a");
    CHECK(json_array_length(a) == 3);
    CHECK_NEAR(json_as_number(json_array_get(a, 0), -1), 1.0, 1e-9);
    const json_value_t *nested = json_array_get(a, 2);
    CHECK(json_array_length(nested) == 2);
    CHECK_NEAR(json_as_number(json_array_get(nested, 1), -1), 4.0, 1e-9);
    const json_value_t *b = json_object_get(v, "b");
    CHECK(b->type == JSON_BOOL && b->as.boolean == 1);
    const json_value_t *c = json_object_get(v, "c");
    CHECK(c->type == JSON_NULL);
    json_free(v);
}

static void test_malformed_input_is_rejected(void) {
    char *error = NULL;
    json_value_t *v = json_parse("{\"a\": }", &error);
    CHECK(v == NULL);
    CHECK(error != NULL);
    free(error);
}

int main(void) {
    test_scalars();
    test_array_and_object();
    test_malformed_input_is_rejected();
    return 0;
}
