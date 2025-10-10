#pragma once

#include "../bitboard.hpp"

#if defined(__AVX2__)
    #define USE_AVX2 1
#else
    #define USE_AVX2 0
#endif

constexpr size_t inputs_per_bucket = 64*12;
constexpr size_t num_of_king_buckets = 16;

constexpr size_t num_perspective_inputs = inputs_per_bucket*num_of_king_buckets;
constexpr size_t factorizer_inputs = inputs_per_bucket;

constexpr size_t num_perspective_neurons = 512;

constexpr size_t layer_stack_size = 8;
constexpr size_t layer1_neurons = 16;
constexpr size_t layer2_neurons = 32;

constexpr size_t output_buckets = 8;

constexpr int output_quantization_fractions = 2048;
constexpr float output_quantization_clamp_max = 10000.0f/output_quantization_fractions;
constexpr float output_quantization_clamp_min = -output_quantization_clamp_max;

constexpr int layer_quantization_fractions = 2048;
constexpr float layer_quantization_clamp_max = 8000.0f/layer_quantization_fractions;
constexpr float layer_quantization_clamp_min = -layer_quantization_clamp_max;

constexpr int halfkp_quantization_fractions = 1024;
constexpr float halfkp_quantization_clamp_max = 4000.0f/halfkp_quantization_fractions;
constexpr float halfkp_quantization_clamp_min = -halfkp_quantization_clamp_max;

constexpr int output_quantization_shift = 11;
constexpr int layer_quantization_shift = 11;

inline int get_king_bucket(int king_sq)
{
    //return 0;

    static const int buckets[64] = {
    15, 15, 14, 14, 14, 14, 15, 15,
    15, 15, 14, 14, 14, 14, 15, 15,
    13, 13, 12, 12, 12, 12, 13, 13,
    13, 13, 12, 12, 12, 12, 13, 13,
    11, 11, 10, 10, 10, 10, 11, 11,
    9,  9,  8,  8,  8,  8,  9,  9,
    7,  6,  5,  4,  4,  5,  6,  7,
    3,  2,  1,  0,  0,  1,  2,  3
    };

    return buckets[king_sq];
}

inline int encode_input_with_buckets(int type, int color, int sq_index, int king_sq)
{
    int king_bucket = get_king_bucket(king_sq);

    if ((king_sq & 0x4) != 0) {
        sq_index ^= 0x7;
    }

    int index = sq_index*12;
    if (color == 0x8) {
        index += 1;
    }
    index += (type-1)*2;
    return index + king_bucket*inputs_per_bucket;
}

inline int encode_factorizer_input(int type, int color, int sq_index, int king_sq)
{
    int king_bucket = 16;

    if ((king_sq & 0x4) != 0) {
        sq_index ^= 0x7;
    }

    int index = sq_index*12;
    if (color == 0x8) {
        index += 1;
    }
    index += (type-1)*2;
    return index + king_bucket*inputs_per_bucket;
}

inline int encode_output_bucket(uint64_t non_pawn_pieces)
{
    int b = pop_count(non_pawn_pieces) / 2;
    if (b < 0) {
        b = 0;
    } else if (b >= output_buckets) {
        b = output_buckets-1;
    }
    return b;
}






