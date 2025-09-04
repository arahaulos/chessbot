#pragma once

#include <stdint.h>

struct huffman_coder
{
    static void encode(uint8_t *input, int input_size, uint8_t *&output, int &output_size);
    static void decode(uint8_t *input, int input_size, uint8_t *&output, int &output_size);
};
