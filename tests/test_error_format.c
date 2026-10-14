/* Error-message hardening: dynamic names are joined into messages
 * as data, never interpreted as printf conversions. These pin the
 * exact bytes, including '%' characters in names.
 */
#include <string.h>

#include "nir_c_runtime/graph.h"
#include "nir_c_runtime/json.h"
#include "nir_c_runtime/runtime.h"
#include "test_util.h"

static char *load_error(const char *text) {
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(json != NULL && error == NULL);
    nir_graph_t graph;
    CHECK(!nir_graph_load(json, &graph, &error));
    CHECK(error != NULL);
    json_free(json);
    return error;
}

static void test_unsupported_kind_keeps_format_chars_literal(void) {
    char *error = load_error(
        "{\"nodes\":[{\"name\":\"m\",\"kind\":\"Myst%sry\",\"params\":{}}],"
        "\"edges\":[]}");
    CHECK(strcmp(error, "unsupported node kind: Myst%sry") == 0);
    free(error);
}

static void test_duplicate_name_keeps_format_chars_literal(void) {
    char *error = load_error(
        "{\"nodes\":["
        "{\"name\":\"n%nm\",\"kind\":\"Scale\",\"params\":{\"scale\":1}},"
        "{\"name\":\"n%nm\",\"kind\":\"Scale\",\"params\":{\"scale\":1}}],"
        "\"edges\":[]}");
    CHECK(strcmp(error, "duplicate node name: n%nm") == 0);
    free(error);
}

static void test_runtime_no_inputs_keeps_format_chars_literal(void) {
    const char *text =
        "{\"nodes\":["
        "{\"name\":\"input\",\"kind\":\"Input\"},"
        "{\"name\":\"output\",\"kind\":\"Output\"},"
        "{\"name\":\"s%s\",\"kind\":\"Scale\",\"params\":{\"scale\":2}}],"
        "\"edges\":[{\"source\":\"input\",\"target\":\"output\"}]}";
    char *error = NULL;
    json_value_t *json = json_parse(text, &error);
    CHECK(json != NULL && error == NULL);
    nir_graph_t graph;
    CHECK(nir_graph_load(json, &graph, &error));
    nir_runtime_t runtime;
    CHECK(nir_runtime_init(&runtime, &graph, &error));
    size_t shape[1] = {2};
    nir_tensor_t frame = nir_tensor_alloc(shape, 1);
    frame.data[0] = 1.0f;
    frame.data[1] = 2.0f;
    nir_tensor_t readout;
    CHECK(!nir_runtime_run(&runtime, &frame, 1, &readout, &error));
    CHECK(error != NULL);
    CHECK(strcmp(error, "node s%s received no inputs") == 0);
    nir_tensor_free(&frame);
    nir_runtime_free(&runtime);
    nir_graph_free(&graph);
    json_free(json);
    free(error);
}

int main(void) {
    test_unsupported_kind_keeps_format_chars_literal();
    test_duplicate_name_keeps_format_chars_literal();
    test_runtime_no_inputs_keeps_format_chars_literal();
    return 0;
}
