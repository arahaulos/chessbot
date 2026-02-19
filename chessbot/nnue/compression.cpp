#include "compression.hpp"
#include <array>
#include <vector>

#include "nnue_defs.hpp"

#include <iostream>

void transform_first_layer(uint8_t *input, int32_t input_size, uint8_t *&output, bool reverse, int neurons, int buckets, int pieces, int squares)
{
    output = new uint8_t[input_size];
    for (int i = 0; i < input_size; i++) {
        output[i] = input[i];
    }

    if (input_size < pieces*squares*neurons*buckets*2) {
        return;
    }

    int16_t *input_i16 = (int16_t*)input;
    int16_t *output_i16 = (int16_t*)output;

    for (int b = 0; b < buckets; b++) {
        for (int n = 0; n < neurons; n++) {
            for (int p = 0; p < pieces; p++) {
                for (int s = 0; s < squares; s++) {
                    int idx0 = (b*pieces*squares + s*pieces + p)*neurons + n;
                    int idx1 = (n*pieces*squares + p*squares + s)*buckets + b;

                    if (reverse) {
                        output_i16[idx0] = input_i16[idx1];
                    } else {
                        output_i16[idx1] = input_i16[idx0];
                    }
                }
            }
        }
    }
}

void variable_bitwidth_decode(uint8_t *input, int32_t input_size, uint8_t *&output, int32_t &output_size, int32_t block_size)
{
    std::vector<uint8_t> decoded;
    decoded.reserve(input_size*2);

    int32_t read_index = 0;
    auto read_bits = [&](int num_of_bits) -> uint32_t {
        if (read_index+32 < input_size*8 && num_of_bits < 17) {
            uint32_t byte_index = read_index >> 3;
            uint32_t bit_index = read_index & 0x7;
            uint32_t data = ((input[byte_index + 0]) | (input[byte_index + 1] << 8) | (input[byte_index + 2] << 16)) >> bit_index;

            read_index += num_of_bits;

            return data & ((1 << num_of_bits)-1);
        } else {
            uint32_t data = 0;
            for (int i = 0; i < num_of_bits; i++) {
                uint8_t byte = input[read_index >> 3];
                if ((byte >> (read_index & 0x7)) & 0x1) {
                    data |= (1 << i);
                }
                read_index += 1;
            }
            return data;
        }
    };

    while (read_index < input_size*8) {
        int16_t min_val = read_bits(16);
        int bits = read_bits(5);

        for (int i = 0; i < block_size && read_index+bits < input_size*8; i++) {
            int16_t value = min_val + read_bits(bits);

            decoded.push_back( value       & 0xFF);
            decoded.push_back((value >> 8) & 0xFF);
        }
    }

    output_size = decoded.size();
    output = new uint8_t[output_size];
    for (int32_t i = 0; i < output_size; i++) {
        output[i] = decoded[i];
    }
}


void variable_bitwidth_encode(uint8_t *input, int32_t input_size, uint8_t *&output, int32_t &output_size, int32_t block_size)
{
    std::vector<uint8_t> encoded;
    encoded.reserve(input_size);

    uint8_t bit_index = 0;
    uint8_t byte = 0;
    auto write_bits = [&](uint32_t data, int num_of_bits) {
        for (int i = 0; i < num_of_bits; i++) {
            uint8_t bit = (data & 0x1);
            data >>= 1;

            byte |= (bit << bit_index);

            bit_index++;
            if (bit_index > 7) {
                encoded.push_back(byte);
                byte = 0;
                bit_index = 0;
            }
        }
    };

    int16_t *data_i16 = (int16_t*)input;
    int32_t len_i16 = input_size/2;

    for (int32_t i = 0; i < len_i16; i += block_size) {
        int32_t min_val = data_i16[i];
        int32_t max_val = data_i16[i];
        for (int32_t j = i; j < i+block_size && j < len_i16; j++) {
            int32_t d = data_i16[j];

            min_val = std::min(min_val, d);
            max_val = std::max(max_val, d);
        }

        int32_t range = max_val - min_val;
        int32_t max_range = 1;
        int bits = 1;

        while ((max_range <<= 1) <= range) { bits++; }

        if (max_val == min_val && i+block_size < len_i16) {
            bits = 0;
        }

        write_bits(min_val, 16);
        write_bits(bits, 5);

        for (int j = i; j < i+block_size && j < len_i16; j++) {
            int32_t d = data_i16[j] - min_val;
            write_bits(d, bits);
        }
    }
    if (bit_index > 0) {
        encoded.push_back(byte);
    }

    output_size = encoded.size();
    output = new uint8_t[output_size];
    for (int32_t i = 0; i < output_size; i++) {
        output[i] = encoded[i];
    }
}

void nnue_compressor::encode(uint8_t *input, int32_t input_size, uint8_t *&output, int32_t &output_size)
{
    uint8_t *tmp;
    transform_first_layer(input, input_size, tmp, false, num_perspective_neurons+num_perspective_psqt+quantized_perspective_pad, num_of_king_buckets, 12, 64);
    variable_bitwidth_encode(tmp, input_size, output, output_size, 64);
    delete [] tmp;
}

void nnue_compressor::decode(uint8_t *input, int32_t input_size, uint8_t *&output, int32_t &output_size)
{
    uint8_t *tmp;
    variable_bitwidth_decode(input, input_size, tmp, output_size, 64);
    transform_first_layer(tmp, output_size, output, true, num_perspective_neurons+num_perspective_psqt+quantized_perspective_pad, num_of_king_buckets, 12, 64);
    delete [] tmp;
}

