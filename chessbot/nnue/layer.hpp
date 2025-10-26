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

    void save(int16_t *data, size_t &index)
    {
        for (size_t i = 0; i < NEURONS*INPUTS*STACK_SIZE; i++) {
            data[index++] = weights[i];
        }
        for (size_t i = 0; i < NEURONS*STACK_SIZE; i++) {
            data[index++] = biases[i];
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

        #if USE_AXV2

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

        #else


        __m128i tmp0 = _mm_setzero_si128();
        __m128i tmp1 = _mm_setzero_si128();

        for (int j = 0; j < IN; j += 16) {
            __m128i a0 = _mm_load_si128((__m128i*)&input[j]);
            __m128i a1 = _mm_load_si128((__m128i*)&input[j+8]);

            __m128i b0 = _mm_load_si128((__m128i*)&weight[j]);
            __m128i b1 = _mm_load_si128((__m128i*)&weight[j+8]);

            __m128i m0 = _mm_madd_epi16(a0, b0);
            __m128i m1 = _mm_madd_epi16(a1, b1);

            tmp0 = _mm_add_epi32(m0, tmp0);
            tmp1 = _mm_add_epi32(m1, tmp1);
        }

        constexpr int shift = (IS_OUTPUT_LAYER ? output_quantization_shift : layer_quantization_shift);

        tmp0 = _mm_srai_epi32(tmp0, shift);
        tmp1 = _mm_srai_epi32(tmp1, shift);

        __m128i hadd0 = _mm_hadd_epi32(tmp0, tmp0);
        __m128i hadd1 = _mm_hadd_epi32(tmp1, tmp1);

        __m128i sum128 = _mm_add_epi32(hadd0, hadd1);

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

        #endif
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


        #if USE_AVX2

        constexpr int vec_out = OUT / 16;

        __m256i acc_lo[vec_out];
        __m256i acc_hi[vec_out];

        for (int i = 0; i < vec_out; i++) {
            acc_lo[i] = _mm256_set1_epi32(0);
            acc_hi[i] = _mm256_set1_epi32(0);
        }

        for (int i = 0; i < IN; i += 4) {
            __m256i input0 = _mm256_set1_epi32(*(int32_t*)&p_layer[i+0]);
            __m256i input1 = _mm256_set1_epi32(*(int32_t*)&p_layer[i+2]);

            const int16_t* __restrict w0 = &weights_ptr[(i+0)*OUT];
            const int16_t* __restrict w1 = &weights_ptr[(i+1)*OUT];
            const int16_t* __restrict w2 = &weights_ptr[(i+2)*OUT];
            const int16_t* __restrict w3 = &weights_ptr[(i+3)*OUT];

            for (int j = 0; j < vec_out; j++) {
                __m256i w16_0 = _mm256_load_si256((__m256i*)&w0[j*16]);
                __m256i w16_1 = _mm256_load_si256((__m256i*)&w1[j*16]);

                __m256i w16_2 = _mm256_load_si256((__m256i*)&w2[j*16]);
                __m256i w16_3 = _mm256_load_si256((__m256i*)&w3[j*16]);

                __m256i w0_lo = _mm256_unpacklo_epi16(w16_0, w16_1);
                __m256i w0_hi = _mm256_unpackhi_epi16(w16_0, w16_1);

                __m256i w1_lo = _mm256_unpacklo_epi16(w16_2, w16_3);
                __m256i w1_hi = _mm256_unpackhi_epi16(w16_2, w16_3);

                acc_lo[j] = _mm256_add_epi32(acc_lo[j], _mm256_madd_epi16(input0, w0_lo));
                acc_hi[j] = _mm256_add_epi32(acc_hi[j], _mm256_madd_epi16(input0, w0_hi));

                acc_lo[j] = _mm256_add_epi32(acc_lo[j], _mm256_madd_epi16(input1, w1_lo));
                acc_hi[j] = _mm256_add_epi32(acc_hi[j], _mm256_madd_epi16(input1, w1_hi));
            }
        }

        __m256i acc[OUT / 8];

        //Lets fix interleaving after everything is summed
        for (int i = 0; i < vec_out; i++) {
            acc[i*2+0] = _mm256_permute2x128_si256(acc_lo[i], acc_hi[i], 0x20);
            acc[i*2+1] = _mm256_permute2x128_si256(acc_lo[i], acc_hi[i], 0x31);
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


        #else


        __m128i acc[OUT / 4];

        for (int i = 0; i < OUT / 4; i++) {
            acc[i] = _mm_set1_epi32(0);
        }

        for (int i = 0; i < IN; i += 4) {
            __m128i input0 = _mm_set1_epi32(*(int32_t*)&p_layer[i+0]);
            __m128i input1 = _mm_set1_epi32(*(int32_t*)&p_layer[i+2]);

            for (int j = 0; j < OUT / 4; j += 2) {
                __m128i w16_0 = _mm_load_si128((__m128i*)&weights_ptr[(i+0)*OUT + j*4]);
                __m128i w16_1 = _mm_load_si128((__m128i*)&weights_ptr[(i+1)*OUT + j*4]);

                __m128i w16_2 = _mm_load_si128((__m128i*)&weights_ptr[(i+2)*OUT + j*4]);
                __m128i w16_3 = _mm_load_si128((__m128i*)&weights_ptr[(i+3)*OUT + j*4]);

                __m128i w0_lo = _mm_unpacklo_epi16(w16_0, w16_1);
                __m128i w0_hi = _mm_unpackhi_epi16(w16_0, w16_1);

                __m128i w1_lo = _mm_unpacklo_epi16(w16_2, w16_3);
                __m128i w1_hi = _mm_unpackhi_epi16(w16_2, w16_3);

                acc[j] = _mm_add_epi32(acc[j], _mm_madd_epi16(input0, w0_lo));
                acc[j+1] = _mm_add_epi32(acc[j+1], _mm_madd_epi16(input0, w0_hi));

                acc[j] = _mm_add_epi32(acc[j], _mm_madd_epi16(input1, w1_lo));
                acc[j+1] = _mm_add_epi32(acc[j+1], _mm_madd_epi16(input1, w1_hi));
            }
        }

        __m128i minv = _mm_set1_epi32(0);
        __m128i maxv = _mm_set1_epi32(layer_quantization_fractions);
        for (int i = 0; i < OUT / 4; i += 2) {
            __m128i bias16_0 = _mm_load_si128((__m128i*)&biases_ptr[i*4]);
            __m128i bias16_1 = _mm_load_si128((__m128i*)&biases_ptr[i*4+4]);

            __m128i bias0 = _mm_cvtepi16_epi32(bias16_0);
            __m128i bias1 = _mm_cvtepi16_epi32(bias16_1);

            acc[i+0] = _mm_add_epi32(_mm_srai_epi32(acc[i+0], layer_quantization_shift), bias0);
            acc[i+1] = _mm_add_epi32(_mm_srai_epi32(acc[i+1], layer_quantization_shift), bias1);

            acc[i+0] = _mm_max_epi32(acc[i+0], minv);
            acc[i+1] = _mm_max_epi32(acc[i+1], minv);

            acc[i+0] = _mm_min_epi32(acc[i+0], maxv);
            acc[i+1] = _mm_min_epi32(acc[i+1], maxv);

            _mm_store_si128((__m128i*)&neurons[i*4], _mm_packs_epi32(acc[i+0], acc[i+1]));
        }

        #endif // USE_AVX2

    }

    void update(int bucket, int16_t *prev_layer0, int16_t *prev_layer_active_outputs0, int num_of_active_inputs0, int16_t *prev_layer1, int16_t *prev_layer_active_outputs1, int num_of_active_inputs1) {
        constexpr int shift = (IS_OUTPUT_LAYER ? output_quantization_shift : layer_quantization_shift);

        int16_t *weights0_ptr = (int16_t*)__builtin_assume_aligned(&weights->transposed_weights[bucket*IN*OUT], 32);
        int16_t *weights1_ptr = (int16_t*)__builtin_assume_aligned(&weights->transposed_weights[bucket*IN*OUT + (IN/2)*OUT], 32);
        int16_t *biases_ptr = (int16_t*)__builtin_assume_aligned(&weights->biases[bucket*OUT], 32);

        int16_t *p_layer0 = (int16_t*)__builtin_assume_aligned(prev_layer0, 32);
        int16_t *p_layer1 = (int16_t*)__builtin_assume_aligned(prev_layer1, 32);

        int16_t *p_layer0_idx = (int16_t*)__builtin_assume_aligned(prev_layer_active_outputs0, 32);
        int16_t *p_layer1_idx = (int16_t*)__builtin_assume_aligned(prev_layer_active_outputs1, 32);


        #if USE_AVX2

        constexpr int vec_out = OUT / 16;

        __m256i acc_lo[vec_out];
        __m256i acc_hi[vec_out];

        for (int i = 0; i < vec_out; i++) {
            acc_lo[i] = _mm256_set1_epi32(0);
            acc_hi[i] = _mm256_set1_epi32(0);
        }

        for (int i = 0; i < num_of_active_inputs0; i += 4) {
            int16_t idx0 = p_layer0_idx[i];
            int16_t idx1 = p_layer0_idx[i+1];
            int16_t idx2 = p_layer0_idx[i+2];
            int16_t idx3 = p_layer0_idx[i+3];

            __m256i input0 = _mm256_unpacklo_epi16(_mm256_set1_epi16(p_layer0[idx0]), _mm256_set1_epi16(p_layer0[idx1]));
            __m256i input1 = _mm256_unpacklo_epi16(_mm256_set1_epi16(p_layer0[idx2]), _mm256_set1_epi16(p_layer0[idx3]));

            for (int j = 0; j < vec_out; j++) {
                __m256i w16_0 = _mm256_load_si256((__m256i*)&weights0_ptr[idx0*OUT + j*16]);
                __m256i w16_1 = _mm256_load_si256((__m256i*)&weights0_ptr[idx1*OUT + j*16]);

                __m256i w16_2 = _mm256_load_si256((__m256i*)&weights0_ptr[idx2*OUT + j*16]);
                __m256i w16_3 = _mm256_load_si256((__m256i*)&weights0_ptr[idx3*OUT + j*16]);

                __m256i w0_lo = _mm256_unpacklo_epi16(w16_0, w16_1);
                __m256i w0_hi = _mm256_unpackhi_epi16(w16_0, w16_1);

                __m256i w1_lo = _mm256_unpacklo_epi16(w16_2, w16_3);
                __m256i w1_hi = _mm256_unpackhi_epi16(w16_2, w16_3);

                acc_lo[j] = _mm256_add_epi32(acc_lo[j], _mm256_madd_epi16(input0, w0_lo));
                acc_hi[j] = _mm256_add_epi32(acc_hi[j], _mm256_madd_epi16(input0, w0_hi));

                acc_lo[j] = _mm256_add_epi32(acc_lo[j], _mm256_madd_epi16(input1, w1_lo));
                acc_hi[j] = _mm256_add_epi32(acc_hi[j], _mm256_madd_epi16(input1, w1_hi));
            }
        }
        for (int i = 0; i < num_of_active_inputs1; i += 4) {
            int16_t idx0 = p_layer1_idx[i];
            int16_t idx1 = p_layer1_idx[i+1];
            int16_t idx2 = p_layer1_idx[i+2];
            int16_t idx3 = p_layer1_idx[i+3];

            __m256i input0 = _mm256_unpacklo_epi16(_mm256_set1_epi16(p_layer1[idx0]), _mm256_set1_epi16(p_layer1[idx1]));
            __m256i input1 = _mm256_unpacklo_epi16(_mm256_set1_epi16(p_layer1[idx2]), _mm256_set1_epi16(p_layer1[idx3]));

            for (int j = 0; j < vec_out; j++) {
                __m256i w16_0 = _mm256_load_si256((__m256i*)&weights1_ptr[idx0*OUT + j*16]);
                __m256i w16_1 = _mm256_load_si256((__m256i*)&weights1_ptr[idx1*OUT + j*16]);

                __m256i w16_2 = _mm256_load_si256((__m256i*)&weights1_ptr[idx2*OUT + j*16]);
                __m256i w16_3 = _mm256_load_si256((__m256i*)&weights1_ptr[idx3*OUT + j*16]);

                __m256i w0_lo = _mm256_unpacklo_epi16(w16_0, w16_1);
                __m256i w0_hi = _mm256_unpackhi_epi16(w16_0, w16_1);

                __m256i w1_lo = _mm256_unpacklo_epi16(w16_2, w16_3);
                __m256i w1_hi = _mm256_unpackhi_epi16(w16_2, w16_3);

                acc_lo[j] = _mm256_add_epi32(acc_lo[j], _mm256_madd_epi16(input0, w0_lo));
                acc_hi[j] = _mm256_add_epi32(acc_hi[j], _mm256_madd_epi16(input0, w0_hi));

                acc_lo[j] = _mm256_add_epi32(acc_lo[j], _mm256_madd_epi16(input1, w1_lo));
                acc_hi[j] = _mm256_add_epi32(acc_hi[j], _mm256_madd_epi16(input1, w1_hi));
            }
        }

        __m256i acc[OUT / 8];

        //Lets fix interleaving after everything is summed
        for (int i = 0; i < vec_out; i++) {
            acc[i*2+0] = _mm256_permute2x128_si256(acc_lo[i], acc_hi[i], 0x20);
            acc[i*2+1] = _mm256_permute2x128_si256(acc_lo[i], acc_hi[i], 0x31);
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

        #else

        __m128i acc[OUT / 4];

        for (int i = 0; i < OUT / 4; i++) {
            acc[i] = _mm_set1_epi32(0);
        }

        for (int i = 0; i < num_of_active_inputs0; i += 4) {
            int16_t idx0 = p_layer0_idx[i];
            int16_t idx1 = p_layer0_idx[i+1];
            int16_t idx2 = p_layer0_idx[i+2];
            int16_t idx3 = p_layer0_idx[i+3];

            __m128i input0 = _mm_set1_epi32((int32_t)p_layer0[idx0] | ((int32_t)p_layer0[idx1] << 16));
            __m128i input1 = _mm_set1_epi32((int32_t)p_layer0[idx2] | ((int32_t)p_layer0[idx3] << 16));

            for (int j = 0; j < OUT / 4; j += 2) {
                __m128i w16_0 = _mm_load_si128((__m128i*)&weights0_ptr[idx0*OUT + j*4]);
                __m128i w16_1 = _mm_load_si128((__m128i*)&weights0_ptr[idx1*OUT + j*4]);

                __m128i w16_2 = _mm_load_si128((__m128i*)&weights0_ptr[idx2*OUT + j*4]);
                __m128i w16_3 = _mm_load_si128((__m128i*)&weights0_ptr[idx3*OUT + j*4]);

                __m128i w0_lo = _mm_unpacklo_epi16(w16_0, w16_1);
                __m128i w0_hi = _mm_unpackhi_epi16(w16_0, w16_1);

                __m128i w1_lo = _mm_unpacklo_epi16(w16_2, w16_3);
                __m128i w1_hi = _mm_unpackhi_epi16(w16_2, w16_3);

                acc[j] = _mm_add_epi32(acc[j], _mm_madd_epi16(input0, w0_lo));
                acc[j+1] = _mm_add_epi32(acc[j+1], _mm_madd_epi16(input0, w0_hi));

                acc[j] = _mm_add_epi32(acc[j], _mm_madd_epi16(input1, w1_lo));
                acc[j+1] = _mm_add_epi32(acc[j+1], _mm_madd_epi16(input1, w1_hi));
            }
        }

        for (int i = 0; i < num_of_active_inputs1; i += 4) {
            int16_t idx0 = p_layer1_idx[i];
            int16_t idx1 = p_layer1_idx[i+1];
            int16_t idx2 = p_layer1_idx[i+2];
            int16_t idx3 = p_layer1_idx[i+3];

            __m128i input0 = _mm_set1_epi32((int32_t)p_layer1[idx0] | ((int32_t)p_layer1[idx1] << 16));
            __m128i input1 = _mm_set1_epi32((int32_t)p_layer1[idx2] | ((int32_t)p_layer1[idx3] << 16));

            for (int j = 0; j < OUT / 4; j += 2) {
                __m128i w16_0 = _mm_load_si128((__m128i*)&weights1_ptr[idx0*OUT + j*4]);
                __m128i w16_1 = _mm_load_si128((__m128i*)&weights1_ptr[idx1*OUT + j*4]);

                __m128i w16_2 = _mm_load_si128((__m128i*)&weights1_ptr[idx2*OUT + j*4]);
                __m128i w16_3 = _mm_load_si128((__m128i*)&weights1_ptr[idx3*OUT + j*4]);

                __m128i w0_lo = _mm_unpacklo_epi16(w16_0, w16_1);
                __m128i w0_hi = _mm_unpackhi_epi16(w16_0, w16_1);

                __m128i w1_lo = _mm_unpacklo_epi16(w16_2, w16_3);
                __m128i w1_hi = _mm_unpackhi_epi16(w16_2, w16_3);

                acc[j] = _mm_add_epi32(acc[j], _mm_madd_epi16(input0, w0_lo));
                acc[j+1] = _mm_add_epi32(acc[j+1], _mm_madd_epi16(input0, w0_hi));

                acc[j] = _mm_add_epi32(acc[j], _mm_madd_epi16(input1, w1_lo));
                acc[j+1] = _mm_add_epi32(acc[j+1], _mm_madd_epi16(input1, w1_hi));
            }
        }
        //Lets add bias and calculate activation
        __m128i minv = _mm_set1_epi32(0);
        __m128i maxv = _mm_set1_epi32(layer_quantization_fractions);
        for (int i = 0; i < OUT / 4; i += 2) {
            __m128i bias16_0 = _mm_load_si128((__m128i*)&biases_ptr[i*4]);
            __m128i bias16_1 = _mm_load_si128((__m128i*)&biases_ptr[i*4+4]);

            __m128i bias0 = _mm_cvtepi16_epi32(bias16_0);
            __m128i bias1 = _mm_cvtepi16_epi32(bias16_1);

            acc[i+0] = _mm_add_epi32(_mm_srai_epi32(acc[i+0], shift), bias0);
            acc[i+1] = _mm_add_epi32(_mm_srai_epi32(acc[i+1], shift), bias1);

            if (!IS_OUTPUT_LAYER) {
                acc[i+0] = _mm_max_epi32(acc[i+0], minv);
                acc[i+1] = _mm_max_epi32(acc[i+1], minv);

                acc[i+0] = _mm_min_epi32(acc[i+0], maxv);
                acc[i+1] = _mm_min_epi32(acc[i+1], maxv);
            }

            _mm_store_si128((__m128i*)&neurons[i*4], _mm_packs_epi32(acc[i+0], acc[i+1]));
        }

        #endif
    }


    int32_t out;
    int16_t *neurons;
    int16_t *neurons_buffer;


    nnue_layer_weights<IN, OUT, STACK_SIZE> *weights;
};





