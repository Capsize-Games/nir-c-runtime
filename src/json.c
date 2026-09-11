#include "nir_c_runtime/json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *cursor;
    char *error;
} parser_t;

static json_value_t *parse_value(parser_t *p);

static void skip_whitespace(parser_t *p) {
    while (*p->cursor == ' ' || *p->cursor == '\t' || *p->cursor == '\n' ||
           *p->cursor == '\r') {
        p->cursor++;
    }
}

static void set_error(parser_t *p, const char *message) {
    if (p->error != NULL) {
        return; /* keep the first error */
    }
    size_t length = strlen(message) + 32;
    p->error = (char *)malloc(length);
    if (p->error != NULL) {
        snprintf(p->error, length, "%s near: %.16s", message, p->cursor);
    }
}

static json_value_t *alloc_value(json_type_t type) {
    json_value_t *value = (json_value_t *)calloc(1, sizeof(json_value_t));
    if (value != NULL) {
        value->type = type;
    }
    return value;
}

static int append_char(char **buffer, size_t *len, size_t *cap, char c) {
    if (*len + 1 >= *cap) {
        size_t new_cap = (*cap == 0) ? 16 : (*cap * 2);
        char *grown = (char *)realloc(*buffer, new_cap);
        if (grown == NULL) {
            return 0;
        }
        *buffer = grown;
        *cap = new_cap;
    }
    (*buffer)[(*len)++] = c;
    return 1;
}

/* Encodes a BMP code point as UTF-8; surrogate pairs are not
 * combined, since NIR's own strings (node names/kinds) never need
 * them -- a lone \uXXXX escape is all this format actually emits. */
static void append_utf8(char **buffer, size_t *len, size_t *cap,
                         unsigned int code_point) {
    if (code_point < 0x80) {
        append_char(buffer, len, cap, (char)code_point);
    } else if (code_point < 0x800) {
        append_char(buffer, len, cap, (char)(0xC0 | (code_point >> 6)));
        append_char(buffer, len, cap, (char)(0x80 | (code_point & 0x3F)));
    } else {
        append_char(buffer, len, cap, (char)(0xE0 | (code_point >> 12)));
        append_char(
            buffer, len, cap, (char)(0x80 | ((code_point >> 6) & 0x3F))
        );
        append_char(buffer, len, cap, (char)(0x80 | (code_point & 0x3F)));
    }
}

static char *parse_raw_string(parser_t *p) {
    if (*p->cursor != '"') {
        set_error(p, "expected string");
        return NULL;
    }
    p->cursor++;
    char *buffer = NULL;
    size_t len = 0, cap = 0;
    while (*p->cursor != '"') {
        if (*p->cursor == '\0') {
            set_error(p, "unterminated string");
            free(buffer);
            return NULL;
        }
        if (*p->cursor == '\\') {
            p->cursor++;
            switch (*p->cursor) {
                case '"': append_char(&buffer, &len, &cap, '"'); break;
                case '\\': append_char(&buffer, &len, &cap, '\\'); break;
                case '/': append_char(&buffer, &len, &cap, '/'); break;
                case 'b': append_char(&buffer, &len, &cap, '\b'); break;
                case 'f': append_char(&buffer, &len, &cap, '\f'); break;
                case 'n': append_char(&buffer, &len, &cap, '\n'); break;
                case 'r': append_char(&buffer, &len, &cap, '\r'); break;
                case 't': append_char(&buffer, &len, &cap, '\t'); break;
                case 'u': {
                    unsigned int code_point = 0;
                    sscanf(p->cursor + 1, "%4x", &code_point);
                    append_utf8(&buffer, &len, &cap, code_point);
                    p->cursor += 4;
                    break;
                }
                default:
                    set_error(p, "invalid escape");
                    free(buffer);
                    return NULL;
            }
            p->cursor++;
        } else {
            append_char(&buffer, &len, &cap, *p->cursor);
            p->cursor++;
        }
    }
    p->cursor++;
    append_char(&buffer, &len, &cap, '\0');
    if (buffer == NULL) {
        buffer = (char *)calloc(1, 1);
    }
    return buffer;
}

static json_value_t *parse_string(parser_t *p) {
    char *text = parse_raw_string(p);
    if (text == NULL) {
        return NULL;
    }
    json_value_t *value = alloc_value(JSON_STRING);
    if (value == NULL) {
        free(text);
        return NULL;
    }
    value->as.string = text;
    return value;
}

static json_value_t *parse_number(parser_t *p) {
    char *end = NULL;
    double number = strtod(p->cursor, &end);
    if (end == p->cursor) {
        set_error(p, "invalid number");
        return NULL;
    }
    p->cursor = end;
    json_value_t *value = alloc_value(JSON_NUMBER);
    if (value != NULL) {
        value->as.number = number;
    }
    return value;
}

static int array_push(json_value_t *array, json_value_t *item) {
    size_t count = array->as.array.count;
    json_value_t **grown = (json_value_t **)realloc(
        array->as.array.items, (count + 1) * sizeof(json_value_t *)
    );
    if (grown == NULL) {
        return 0;
    }
    grown[count] = item;
    array->as.array.items = grown;
    array->as.array.count = count + 1;
    return 1;
}

static json_value_t *parse_array(parser_t *p) {
    p->cursor++; /* '[' */
    json_value_t *array = alloc_value(JSON_ARRAY);
    skip_whitespace(p);
    if (*p->cursor == ']') {
        p->cursor++;
        return array;
    }
    while (1) {
        skip_whitespace(p);
        json_value_t *item = parse_value(p);
        if (item == NULL || !array_push(array, item)) {
            json_free(array);
            return NULL;
        }
        skip_whitespace(p);
        if (*p->cursor == ',') {
            p->cursor++;
            continue;
        }
        if (*p->cursor == ']') {
            p->cursor++;
            return array;
        }
        set_error(p, "expected ',' or ']'");
        json_free(array);
        return NULL;
    }
}

static int object_push(json_value_t *object, char *key, json_value_t *val) {
    size_t count = object->as.object.count;
    char **keys = (char **)realloc(
        object->as.object.keys, (count + 1) * sizeof(char *)
    );
    json_value_t **values = (json_value_t **)realloc(
        object->as.object.values, (count + 1) * sizeof(json_value_t *)
    );
    if (keys == NULL || values == NULL) {
        return 0;
    }
    keys[count] = key;
    values[count] = val;
    object->as.object.keys = keys;
    object->as.object.values = values;
    object->as.object.count = count + 1;
    return 1;
}

static json_value_t *parse_object(parser_t *p) {
    p->cursor++; /* '{' */
    json_value_t *object = alloc_value(JSON_OBJECT);
    skip_whitespace(p);
    if (*p->cursor == '}') {
        p->cursor++;
        return object;
    }
    while (1) {
        skip_whitespace(p);
        char *key = parse_raw_string(p);
        if (key == NULL) {
            json_free(object);
            return NULL;
        }
        skip_whitespace(p);
        if (*p->cursor != ':') {
            set_error(p, "expected ':'");
            free(key);
            json_free(object);
            return NULL;
        }
        p->cursor++;
        skip_whitespace(p);
        json_value_t *val = parse_value(p);
        if (val == NULL || !object_push(object, key, val)) {
            free(key);
            json_free(object);
            return NULL;
        }
        skip_whitespace(p);
        if (*p->cursor == ',') {
            p->cursor++;
            continue;
        }
        if (*p->cursor == '}') {
            p->cursor++;
            return object;
        }
        set_error(p, "expected ',' or '}'");
        json_free(object);
        return NULL;
    }
}

static int starts_with(const char *text, const char *literal) {
    return strncmp(text, literal, strlen(literal)) == 0;
}

static json_value_t *parse_value(parser_t *p) {
    skip_whitespace(p);
    switch (*p->cursor) {
        case '"': return parse_string(p);
        case '{': return parse_object(p);
        case '[': return parse_array(p);
        case 't':
            if (starts_with(p->cursor, "true")) {
                p->cursor += 4;
                json_value_t *value = alloc_value(JSON_BOOL);
                if (value != NULL) value->as.boolean = 1;
                return value;
            }
            break;
        case 'f':
            if (starts_with(p->cursor, "false")) {
                p->cursor += 5;
                json_value_t *value = alloc_value(JSON_BOOL);
                if (value != NULL) value->as.boolean = 0;
                return value;
            }
            break;
        case 'n':
            if (starts_with(p->cursor, "null")) {
                p->cursor += 4;
                return alloc_value(JSON_NULL);
            }
            break;
        default:
            if (*p->cursor == '-' || isdigit((unsigned char)*p->cursor)) {
                return parse_number(p);
            }
            break;
    }
    set_error(p, "unexpected token");
    return NULL;
}

json_value_t *json_parse(const char *text, char **error) {
    parser_t p = {text, NULL};
    json_value_t *value = parse_value(&p);
    if (value != NULL) {
        skip_whitespace(&p);
        if (*p.cursor != '\0') {
            set_error(&p, "trailing content after JSON value");
            json_free(value);
            value = NULL;
        }
    }
    if (error != NULL) {
        *error = p.error;
    } else {
        free(p.error);
    }
    return value;
}

void json_free(json_value_t *value) {
    if (value == NULL) {
        return;
    }
    switch (value->type) {
        case JSON_STRING:
            free(value->as.string);
            break;
        case JSON_ARRAY:
            for (size_t i = 0; i < value->as.array.count; i++) {
                json_free(value->as.array.items[i]);
            }
            free(value->as.array.items);
            break;
        case JSON_OBJECT:
            for (size_t i = 0; i < value->as.object.count; i++) {
                free(value->as.object.keys[i]);
                json_free(value->as.object.values[i]);
            }
            free(value->as.object.keys);
            free(value->as.object.values);
            break;
        default:
            break;
    }
    free(value);
}

const json_value_t *json_object_get(
    const json_value_t *value, const char *key
) {
    if (value == NULL || value->type != JSON_OBJECT) {
        return NULL;
    }
    for (size_t i = 0; i < value->as.object.count; i++) {
        if (strcmp(value->as.object.keys[i], key) == 0) {
            return value->as.object.values[i];
        }
    }
    return NULL;
}

const json_value_t *json_array_get(const json_value_t *value, size_t index) {
    if (value == NULL || value->type != JSON_ARRAY) {
        return NULL;
    }
    if (index >= value->as.array.count) {
        return NULL;
    }
    return value->as.array.items[index];
}

size_t json_array_length(const json_value_t *value) {
    if (value == NULL || value->type != JSON_ARRAY) {
        return 0;
    }
    return value->as.array.count;
}

double json_as_number(const json_value_t *value, double fallback) {
    if (value == NULL || value->type != JSON_NUMBER) {
        return fallback;
    }
    return value->as.number;
}

const char *json_as_string(const json_value_t *value) {
    if (value == NULL || value->type != JSON_STRING) {
        return NULL;
    }
    return value->as.string;
}
