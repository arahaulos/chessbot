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


template <int INPUTS, int NEURONS, int STACK_SIZE>
struct nnue_layer_weights
{
    nnue_layer_weights() {
        weights_buffer = new int16_t[INPUTS*NEURONS*STACK_SIZE+64];
        biases_buffer = new int16_t[NEURONS*STACK_SIZE+64];

        transposed_weights_buffer = new int16_t[INPUTS*NEURONS*STACK_SIZE+128];

        weights = align_ptr(weights_buffer);
        biases = align_ptr(biases_buffer);

        transposed_weights = align_ptr(transposed_weights_buffer);
    }

    ~nnue_layer_weights() {
        delete [] weights_buffer;
        delete [] biases_buffer;

        delete [] transposed_weights_buffer;
    }

    void load(int16_t *data, size_t &index)
    {
        for (size_t i = 0; i < NEURONS*INPUTS*STACK_SIZE; i++) {
            weights[i] = data[index++];
        }
        for (size_t i = 0; i < NEURONS*STACK_SIZE; i++) {
            biases[i] = data[index++];
        }


        for (int i = 0; i < STACK_SIZE; i++) {
            for (int j = 0; j < NEURONS; j++) {
                for (int k = 0; k < INPUTS; k++) {
                    transposed_weights[i*NEURONS*INPUTS + k*NEURONS + j] = weights[i*NEURONS*INPUTS + j*INPUTS + k];
                }
            }
        }


    }

    int num_of_biases() const {
        return NEURONS*STACK_SIZE;
    }

    int num_of_weights() const {
        return NEURONS*INPUTS*STACK_SIZE;
    }

    int16_t *weights;
    int16_t *biases;

    int16_t *weights_buffer;
    int16_t *biases_buffer;

    int16_t *transposed_weights;
    int16_t *transposed_weights_buffer;
};


template <int IN, int OUT, int STACK_SIZE, bool IS_OUTPUT_LAYER>
struct nnue_layer
{
    nnue_layer(nnue_layer_weights<IN, OUT, STACK_SIZE> *w) {
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

    void update(int i, int bucket, int16_t *prev_layer) {
        int16_t *neuron = (int16_t*)__builtin_assume_aligned(neurons, 64);
        int16_t *input = (int16_t*)__builtin_assume_aligned(prev_layer, 64);
        int16_t *bias = (int16_t*)__builtin_assume_aligned(&weights->biases[OUT*bucket], 64);
        int16_t *weight = (int16_t*)__builtin_assume_aligned(&weights->weights[IN*OUT*bucket + i*IN], 64);

        __m256i tmp = _mm256_setzero_si256();

        if (IN < 32) {
            for (int j = 0; j < IN; j += 16) {
                __m256i a = _mm256_load_si256((__m256i*)&input[j]);
                __m256i b = _mm256_load_si256((__m256i*)&weight[j]);

                __m256i m = _mm256_madd_epi16(a, b);

                tmp = _mm256_add_epi32(m, tmp);
            }
        } else {
            for (int j = 0; j < IN; j += 32) {
                __m256i a0 = _mm256_load_si256((__m256i*)&input[j]);
                __m256i a1 = _mm256_load_si256((__m256i*)&input[j+16]);

                __m256i b0 = _mm256_load_si256((__m256i*)&weight[j]);
                __m256i b1 = _mm256_load_si256((__m256i*)&weight[j+16]);

                __m256i m0 = _mm256_madd_epi16(a0, b0);
                __m256i m1 = _mm256_madd_epi16(a1, b1);

                tmp = _mm256_add_epi32(m0, tmp);
                tmp = _mm256_add_epi32(m1, tmp);
            }
        }

        constexpr int shift = (IS_OUTPUT_LAYER ? output_quantization_shift : layer_quantization_shift);

        tmp = _mm256_srai_epi32(tmp, shift);

        __m256i hadd = _mm256_hadd_epi32(tmp, tmp);

        __m128i low = _mm256_castsi256_si128(hadd);
        __m128i high = _mm256_extracti128_si256(hadd, 1);

        __m128i sum128 = _mm_add_epi32(low, high);

        __m128i shuffled = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1));
        __m128i final_sum = _mm_add_epi32(sum128, shuffled);

        if (IS_OUTPUT_LAYER) {
            int32_t n = _mm_cvtsi128_si32(final_sum) + bias[i];

            out = n;
            neuron[i] = n;
        } else {
            int32_t n = _mm_cvtsi128_si32(final_sum) + bias[i];
            neuron[i] = activation_func_i32(n);
        }
    }

    void update(int i, int bucket, int16_t *prev_layer0, int16_t *prev_layer1) {

        int16_t *neuron = (int16_t*)__builtin_assume_aligned(neurons, 64);
        int16_t *input0 = (int16_t*)__builtin_assume_aligned(prev_layer0, 64);
        int16_t *input1 = (int16_t*)__builtin_assume_aligned(prev_layer1, 64);

        int16_t *bias = (int16_t*)__builtin_assume_aligned(&weights->biases[OUT*bucket], 64);

        int16_t *weight0 = (int16_t*)__builtin_assume_aligned(&weights->weights[IN*OUT*bucket + i*IN], 64);
        int16_t *weight1 = (int16_t*)__builtin_assume_aligned(&weights->weights[IN*OUT*bucket + (i*IN) + (IN/2)], 64);

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

        constexpr int shift = (IS_OUTPUT_LAYER ? output_quantization_shift : layer_quantization_shift);

        tmp = _mm256_srai_epi32(tmp, shift);

        __m256i hadd = _mm256_hadd_epi32(tmp, tmp);

        __m128i low = _mm256_castsi256_si128(hadd);
        __m128i high = _mm256_extracti128_si256(hadd, 1);

        __m128i sum128 = _mm_add_epi32(low, high);

        __m128i shuffled = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1));
        __m128i final_sum = _mm_add_epi32(sum128, shuffled);

        if (IS_OUTPUT_LAYER) {
            int32_t n = _mm_cvtsi128_si32(final_sum) + bias[i];
            out = n;
            neuron[i] = n;
        } else {
            int32_t n = _mm_cvtsi128_si32(final_sum) + bias[i];
            neuron[i] = activation_func_i32(n);
        }
    }

    void update(int bucket, int16_t *prev_layer) {
        if (IS_OUTPUT_LAYER) {
            for (int i = 0; i < OUT; i++) {
                update(i, bucket, prev_layer);
            }
            return;
        }

        int16_t *weights_ptr = (int16_t*)__builtin_assume_aligned(&weights->transposed_weights[bucket*IN*OUT], 32);
        int16_t *biases_ptr = (int16_t*)__builtin_assume_aligned(&weights->biases[bucket*OUT], 32);
        int16_t *p_layer = (int16_t*)__builtin_assume_aligned(prev_layer, 32);

        __m256i acc[OUT / 8];
        for (int i = 0; i < OUT / 8; i++) {
            acc[i] = _mm256_set1_epi32(0);
        }

        for (int i = 0; i < IN; i += 4) {
            __m256i input0 = _mm256_set1_epi32(*(int32_t*)&p_layer[i+0]);
            __m256i input1 = _mm256_set1_epi32(*(int32_t*)&p_layer[i+2]);

            for (int j = 0; j < OUT / 8; j += 2) {
                __m256i w16_0 = _mm256_load_si256((__m256i*)&weights_ptr[(i+0)*OUT + j*8]);
                __m256i w16_1 = _mm256_load_si256((__m256i*)&weights_ptr[(i+1)*OUT + j*8]);

                __m256i w16_2 = _mm256_load_si256((__m256i*)&weights_ptr[(i+2)*OUT + j*8]);
                __m256i w16_3 = _mm256_load_si256((__m256i*)&weights_ptr[(i+3)*OUT + j*8]);

                __m256i w0_lo = _mm256_unpacklo_epi16(w16_0, w16_1);
                __m256i w0_hi = _mm256_unpackhi_epi16(w16_0, w16_1);

                __m256i w1_lo = _mm256_unpacklo_epi16(w16_2, w16_3);
                __m256i w1_hi = _mm256_unpackhi_epi16(w16_2, w16_3);

                acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(input0, w0_lo));
                acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(input1, w1_lo));

                acc[j+1] = _mm256_add_epi32(acc[j+1], _mm256_madd_epi16(input0, w0_hi));
                acc[j+1] = _mm256_add_epi32(acc[j+1], _mm256_madd_epi16(input1, w1_hi));
            }
        }

        //Lets fix interleaving after everything is summed
        for (int j = 0; j < OUT / 8; j += 2) {
            __m256i lo = acc[j];
            __m256i hi = acc[j+1];

            acc[j]   = _mm256_permute2x128_si256(lo, hi, 0x20);
            acc[j+1] = _mm256_permute2x128_si256(lo, hi, 0x31);
        }

        __m256i minv = _mm256_set1_epi32(0);
        __m256i maxv = _mm256_set1_epi32(layer_quantization_fractions);
        for (int i = 0; i < OUT / 8; i++) {
            __m128i bias16 = _mm_load_si128((__m128i*)&biases_ptr[i*8]);
            __m256i bias = _mm256_cvtepi16_epi32(bias16);

            acc[i] = _mm256_add_epi32(_mm256_srai_epi32(acc[i], layer_quantization_shift), bias);

            acc[i] = _mm256_max_epi32(acc[i], minv);
            acc[i] = _mm256_min_epi32(acc[i], maxv);

            __m128i lo = _mm256_castsi256_si128(acc[i]);
            __m128i hi = _mm256_extracti128_si256(acc[i], 1);
            __m128i v = _mm_packs_epi32(lo, hi);

            _mm_store_si128((__m128i*)&neurons[i*8], v);
        }
    }

    void update(int bucket, int16_t *prev_layer0, int16_t *prev_layer1) {
        for (int i = 0; i < OUT; i++) {
            update(i, bucket, prev_layer0, prev_layer1);
        }
    }

    void update(int bucket, int16_t *prev_layer0, int16_t *prev_layer0_idx, int ni0, int16_t *prev_layer1, int16_t *prev_layer1_idx, int ni1) {
        constexpr int shift = (IS_OUTPUT_LAYER ? output_quantization_shift : layer_quantization_shift);

        int16_t *weights0_ptr = (int16_t*)__builtin_assume_aligned(&weights->transposed_weights[bucket*IN*OUT], 32);
        int16_t *weights1_ptr = (int16_t*)__builtin_assume_aligned(&weights->transposed_weights[bucket*IN*OUT + (IN/2)*OUT], 32);
        int16_t *biases_ptr = (int16_t*)__builtin_assume_aligned(&weights->biases[bucket*OUT], 32);

        int16_t *p_layer0 = (int16_t*)__builtin_assume_aligned(prev_layer0, 32);
        int16_t *p_layer1 = (int16_t*)__builtin_assume_aligned(prev_layer1, 32);

        int16_t *p_layer0_idx = (int16_t*)__builtin_assume_aligned(prev_layer0_idx, 32);
        int16_t *p_layer1_idx = (int16_t*)__builtin_assume_aligned(prev_layer1_idx, 32);

        __m256i acc[OUT / 8];

        for (int i = 0; i < OUT / 8; i++) {
            acc[i] = _mm256_set1_epi32(0);
        }

        for (int i = 0; i < ni0; i += 4) {
            int16_t idx0 = p_layer0_idx[i];
            int16_t idx1 = p_layer0_idx[i+1];
            int16_t idx2 = p_layer0_idx[i+2];
            int16_t idx3 = p_layer0_idx[i+3];

            __m256i input0 = _mm256_set1_epi32((int32_t)p_layer0[idx0] | ((int32_t)p_layer0[idx1] << 16));
            __m256i input1 = _mm256_set1_epi32((int32_t)p_layer0[idx2] | ((int32_t)p_layer0[idx3] << 16));

            for (int j = 0; j < OUT / 8; j += 2) {
                __m256i w16_0 = _mm256_load_si256((__m256i*)&weights0_ptr[idx0*OUT + j*8]);
                __m256i w16_1 = _mm256_load_si256((__m256i*)&weights0_ptr[idx1*OUT + j*8]);

                __m256i w16_2 = _mm256_load_si256((__m256i*)&weights0_ptr[idx2*OUT + j*8]);
                __m256i w16_3 = _mm256_load_si256((__m256i*)&weights0_ptr[idx3*OUT + j*8]);

                __m256i w0_lo = _mm256_unpacklo_epi16(w16_0, w16_1);
                __m256i w1_lo = _mm256_unpacklo_epi16(w16_2, w16_3);

                __m256i w0_hi = _mm256_unpackhi_epi16(w16_0, w16_1);
                __m256i w1_hi = _mm256_unpackhi_epi16(w16_2, w16_3);

                acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(input0, w0_lo));
                acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(input1, w1_lo));
                acc[j+1] = _mm256_add_epi32(acc[j+1], _mm256_madd_epi16(input0, w0_hi));
                acc[j+1] = _mm256_add_epi32(acc[j+1], _mm256_madd_epi16(input1, w1_hi));
            }
        }

        for (int i = 0; i < ni1; i += 4) {
            int16_t idx0 = p_layer1_idx[i];
            int16_t idx1 = p_layer1_idx[i+1];
            int16_t idx2 = p_layer1_idx[i+2];
            int16_t idx3 = p_layer1_idx[i+3];

            __m256i input0 = _mm256_set1_epi32((int32_t)p_layer1[idx0] | ((int32_t)p_layer1[idx1] << 16));
            __m256i input1 = _mm256_set1_epi32((int32_t)p_layer1[idx2] | ((int32_t)p_layer1[idx3] << 16));

            for (int j = 0; j < OUT / 8; j += 2) {
                __m256i w16_0 = _mm256_load_si256((__m256i*)&weights1_ptr[idx0*OUT + j*8]);
                __m256i w16_1 = _mm256_load_si256((__m256i*)&weights1_ptr[idx1*OUT + j*8]);

                __m256i w16_2 = _mm256_load_si256((__m256i*)&weights1_ptr[idx2*OUT + j*8]);
                __m256i w16_3 = _mm256_load_si256((__m256i*)&weights1_ptr[idx3*OUT + j*8]);

                __m256i w0_lo = _mm256_unpacklo_epi16(w16_0, w16_1);
                __m256i w1_lo = _mm256_unpacklo_epi16(w16_2, w16_3);

                __m256i w0_hi = _mm256_unpackhi_epi16(w16_0, w16_1);
                __m256i w1_hi = _mm256_unpackhi_epi16(w16_2, w16_3);

                acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(input0, w0_lo));
                acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(input1, w1_lo));
                acc[j+1] = _mm256_add_epi32(acc[j+1], _mm256_madd_epi16(input0, w0_hi));
                acc[j+1] = _mm256_add_epi32(acc[j+1], _mm256_madd_epi16(input1, w1_hi));
            }
        }

        //Lets fix interleaving after everything is summed
        for (int j = 0; j < OUT / 8; j += 2) {
            __m256i lo = acc[j];
            __m256i hi = acc[j+1];

            acc[j]   = _mm256_permute2x128_si256(lo, hi, 0x20);
            acc[j+1] = _mm256_permute2x128_si256(lo, hi, 0x31);
        }

        //Lets add bias and calculate activation
        __m256i minv = _mm256_set1_epi32(0);
        __m256i maxv = _mm256_set1_epi32(layer_quantization_fractions);
        for (int i = 0; i < OUT / 8; i++) {
            __m128i bias16 = _mm_load_si128((__m128i*)&biases_ptr[i*8]);
            __m256i bias = _mm256_cvtepi16_epi32(bias16);

            acc[i] = _mm256_add_epi32(_mm256_srai_epi32(acc[i], shift), bias);

            if (!IS_OUTPUT_LAYER) {
                acc[i] = _mm256_max_epi32(acc[i], minv);
                acc[i] = _mm256_min_epi32(acc[i], maxv);
            }


            __m128i lo = _mm256_castsi256_si128(acc[i]);
            __m128i hi = _mm256_extracti128_si256(acc[i], 1);
            __m128i v = _mm_packs_epi32(lo, hi);

            _mm_store_si128((__m128i*)&neurons[i*8], v);
        }
    }


    int32_t out;
    int16_t *neurons;
    int16_t *neurons_buffer;


    nnue_layer_weights<IN, OUT, STACK_SIZE> *weights;
};





