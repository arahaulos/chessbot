#pragma once

#include <stdint.h>

struct nnue_compressor
{
    static void encode(uint8_t *input, int32_t input_size, uint8_t *&output, int32_t &output_size);
    static void decode(uint8_t *input, int32_t input_size, uint8_t *&output, int32_t &output_size);
};
