#pragma once

#include <cmath>

#include <x86gprintrin.h>
#include <x86intrin.h>

#include "training/training_layer.hpp"



inline int32_t activation_func_i32(int32_t x)
{
    if (x < 0) {
        return 0;
    } else if (x > layer_quantization_fractions) {
        return layer_quantization_fractions;
    }
    return x;
}


template <int INPUTS, int NEURONS>
struct nnue_layer_weights
{
    nnue_layer_weights() {
        weights_buffer = new int16_t[INPUTS*NEURONS+64];
        biases_buffer = new int16_t[NEURONS+64];

        weights = align_ptr(weights_buffer);
        biases = align_ptr(biases_buffer);
    }

    ~nnue_layer_weights() {
        delete [] weights_buffer;
        delete [] biases_buffer;
    }


    void load(std::istream &stream) {
        stream.read((char*)weights, (NEURONS*INPUTS)*sizeof(int16_t));
        stream.read((char*)biases, NEURONS*sizeof(int16_t));
    }

    void load(int16_t *data, size_t &index)
    {
        for (size_t i = 0; i < NEURONS*INPUTS; i++) {
            weights[i] = data[index++];
        }
        for (size_t i = 0; i < NEURONS; i++) {
            biases[i] = data[index++];
        }
    }

    int num_of_biases() const {
        return NEURONS;
    }

    int num_of_weights() const {
        return NEURONS*INPUTS;
    }

    int16_t *weights;
    int16_t *biases;

    int16_t *weights_buffer;
    int16_t *biases_buffer;
};


template <int IN, int OUT>
struct nnue_layer
{
    nnue_layer(nnue_layer_weights<IN, OUT> *w) {
        weights = w;
        neurons_buffer = new int16_t[OUT+64];
        neurons = align_ptr(neurons_buffer);
    }

    ~nnue_layer() {
        delete [] neurons_buffer;
    }


    void reset() {
        int16_t *neuron = (int16_t*)__builtin_assume_aligned(neurons, 64);

        for (int i = 0; i < OUT; i++) {
            neuron[i] = 0;
        }
    }

    void update(int16_t *prev_layer) {
        int16_t *neuron = (int16_t*)__builtin_assume_aligned(neurons, 64);
        int16_t *input = (int16_t*)__builtin_assume_aligned(prev_layer, 64);

        int16_t *bias = (int16_t*)__builtin_assume_aligned(weights->biases, 64);


        for (int i = 0; i < OUT; i++) {
            int16_t *weight = (int16_t*)__builtin_assume_aligned(&weights->weights[i*IN], 64);

            __m256i tmp = _mm256_setzero_si256();

            for (int j = 0; j < IN; j += 16) {
                __m256i a = _mm256_load_si256((__m256i*)&input[j]);
                __m256i b = _mm256_load_si256((__m256i*)&weight[j]);

                __m256i m = _mm256_madd_epi16(a, b);

                tmp = _mm256_add_epi32(m, tmp);
            }
            __m256i hadd = _mm256_hadd_epi32(tmp, tmp);

            __m128i low = _mm256_castsi256_si128(hadd);
            __m128i high = _mm256_extracti128_si256(hadd, 1);

            __m128i sum128 = _mm_add_epi32(low, high);

            __m128i shuffled = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1));
            __m128i final_sum = _mm_add_epi32(sum128, shuffled);

            if (OUT == 1) {
                int32_t n = _mm_cvtsi128_si32(final_sum) / output_quantization_fractions + bias[i];

                out = n;
                neuron[i] = n;
            } else {
                int32_t n = _mm_cvtsi128_si32(final_sum) / layer_quantization_fractions + bias[i];
                neuron[i] = activation_func_i32(n);
            }
        }
    }

    void update(int16_t *prev_layer0, int16_t *prev_layer1) {

        int16_t *neuron = (int16_t*)__builtin_assume_aligned(neurons, 64);
        int16_t *input0 = (int16_t*)__builtin_assume_aligned(prev_layer0, 64);
        int16_t *input1 = (int16_t*)__builtin_assume_aligned(prev_layer1, 64);

        int16_t *bias = (int16_t*)__builtin_assume_aligned(weights->biases, 64);

        for (int i = 0; i < OUT; i++) {
            int16_t *weight0 = (int16_t*)__builtin_assume_aligned(&weights->weights[i*IN], 64);
            int16_t *weight1 = (int16_t*)__builtin_assume_aligned(&weights->weights[(i*IN) + (IN/2)], 64);

            __m256i tmp = _mm256_setzero_si256();

            for (int j = 0; j < (IN/2); j += 32) {
                __m256i a0 = _mm256_load_si256((__m256i*)&input0[j]);
                __m256i b0 = _mm256_load_si256((__m256i*)&weight0[j]);

                __m256i a1 = _mm256_load_si256((__m256i*)&input1[j]);
                __m256i b1 = _mm256_load_si256((__m256i*)&weight1[j]);

                __m256i a2 = _mm256_load_si256((__m256i*)&input0[j+16]);
                __m256i b2 = _mm256_load_si256((__m256i*)&weight0[j+16]);

                __m256i a3 = _mm256_load_si256((__m256i*)&input1[j+16]);
                __m256i b3 = _mm256_load_si256((__m256i*)&weight1[j+16]);

                __m256i m0 = _mm256_madd_epi16(a0, b0);
                __m256i m1 = _mm256_madd_epi16(a1, b1);
                __m256i m2 = _mm256_madd_epi16(a2, b2);
                __m256i m3 = _mm256_madd_epi16(a3, b3);

                m0 = _mm256_add_epi32(m0, m1);
                m1 = _mm256_add_epi32(m2, m3);

                tmp = _mm256_add_epi32(_mm256_add_epi32(m0, m1), tmp);
            }

            __m256i hadd = _mm256_hadd_epi32(tmp, tmp);

            __m128i low = _mm256_castsi256_si128(hadd);
            __m128i high = _mm256_extracti128_si256(hadd, 1);

            __m128i sum128 = _mm_add_epi32(low, high);

            __m128i shuffled = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1));
            __m128i final_sum = _mm_add_epi32(sum128, shuffled);

            if (OUT == 1) {
                int32_t n = _mm_cvtsi128_si32(final_sum) / output_quantization_fractions + bias[i];
                out = n;
                neuron[i] = n;
            } else {
                int32_t n = _mm_cvtsi128_si32(final_sum) / layer_quantization_fractions + bias[i];
                neuron[i] = activation_func_i32(n);
            }
        }
    }

    int32_t out;
    int16_t *neurons;
    int16_t *neurons_buffer;

    nnue_layer_weights<IN, OUT> *weights;
};





