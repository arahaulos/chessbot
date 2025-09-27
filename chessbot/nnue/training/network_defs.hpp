#pragma once


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


/*constexpr int output_quantization_fractions = 1024;
constexpr float output_quantization_clamp_max = 10000.0f/output_quantization_fractions;
constexpr float output_quantization_clamp_min = -output_quantization_clamp_max;

constexpr int layer_quantization_fractions = 512;
constexpr float layer_quantization_clamp_max = 8000.0f/layer_quantization_fractions;
constexpr float layer_quantization_clamp_min = -layer_quantization_clamp_max;

constexpr int halfkp_quantization_fractions = 128;
constexpr float halfkp_quantization_clamp_max = 4000.0f/halfkp_quantization_fractions;
constexpr float halfkp_quantization_clamp_min = -halfkp_quantization_clamp_max;

constexpr int output_quantization_shift = 10;
constexpr int layer_quantization_shift = 9;*/


