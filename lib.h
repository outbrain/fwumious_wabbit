#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define FFM_MAX_K 128

#define NUM_TAPES 8

#define TRANSFORM_NAMESPACE_MARK (1 << 31)

#define FASTMATH_LR_LUT_BITS 11

#define FASTMATH_LR_LUT_SIZE (1 << FASTMATH_LR_LUT_BITS)

#define HEADER_LEN 3

#define NAMESPACE_DESC_LEN 1

#define LABEL_OFFSET 1

#define EXAMPLE_IMPORTANCE_OFFSET 2

#define IS_NOT_SINGLE_MASK (1 << 31)

#define NO_FEATURES IS_NOT_SINGLE_MASK

#define NO_LABEL 255

#define FLOAT32_ONE 1065353216

#define FFM_CONTRA_BUF_LEN 16384

typedef struct FfiPredictor {

} FfiPredictor;

struct FfiPredictor *new_fw_predictor_prototype(const char *command);

struct FfiPredictor *clone_lite(struct FfiPredictor *prototype);

float fw_predict(struct FfiPredictor *ptr, const char *input_buffer);

float fw_predict_with_cache(struct FfiPredictor *ptr, const char *input_buffer);

float fw_setup_cache(struct FfiPredictor *ptr, const char *input_buffer);

void free_predictor(struct FfiPredictor *ptr);
