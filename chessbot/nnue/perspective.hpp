#pragma once

#include "layer.hpp"


template <int INPUTS, int NEURONS>
struct nnue_perspective_weights
{
    nnue_perspective_weights() {
        weights_buffer = new int16_t[INPUTS*NEURONS+64];
        biases_buffer = new int16_t[INPUTS+64];

        biases = align_ptr(biases_buffer);
        weights = align_ptr(weights_buffer);
    }

    ~nnue_perspective_weights() {
        delete [] weights_buffer;
        delete [] biases_buffer;
    }

    void load(std::istream &stream, bool big_net) {
        if (big_net) {
            stream.read((char*)weights, (NEURONS*INPUTS)*sizeof(int16_t));
            stream.read((char*)biases, NEURONS*sizeof(int16_t));
        } else {
            stream.read((char*)weights, (NEURONS*inputs_per_bucket)*sizeof(int16_t));
            stream.read((char*)biases, NEURONS*sizeof(int16_t));
        }
    }

    void load(int16_t *data, size_t &index, bool big_net)
    {
        if (big_net) {
            for (size_t i = 0; i < NEURONS*INPUTS; i++) {
                weights[i] = data[index++];
            }
            for (size_t i = 0; i < NEURONS; i++) {
                biases[i] = data[index++];
            }
        } else {
            for (size_t i = 0; i < NEURONS*inputs_per_bucket; i++) {
                weights[i] = data[index++];
            }
            for (size_t i = 0; i < NEURONS; i++) {
                biases[i] = data[index++];
            }
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


template <int INPUTS, int NEURONS>
struct nnue_perspective
{
    nnue_perspective(nnue_perspective_weights<INPUTS, NEURONS> *w) {
        weights = w;
        neurons_buffer = new int16_t[NEURONS+64];
        acculumator_buffer = new int16_t[NEURONS*128+64];

        neurons = align_ptr(neurons_buffer);
        acculumator = align_ptr(acculumator_buffer);

        incremental_updates = false;
    }

    ~nnue_perspective() {
        delete [] neurons_buffer;
        delete [] acculumator_buffer;
    }


    void reset() {
        int16_t *acc = (int16_t*)__builtin_assume_aligned(acculumator, 64);
        int16_t *bias = (int16_t*)__builtin_assume_aligned(weights->biases, 64);

        for (int i = 0; i < NEURONS; i += 32) {
            _mm256_store_si256((__m256i*)&acc[i], _mm256_load_si256((__m256i*)&bias[i]));
            _mm256_store_si256((__m256i*)&acc[i+16], _mm256_load_si256((__m256i*)&bias[i+16]));
        }

    }

    void set_input(int index) {
        if (!incremental_updates) {
            return;
        }

        int16_t *acc = acculumator;
        int16_t *weight = &weights->weights[index*NEURONS];

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&acc[i]);
            __m256i b0 = _mm256_load_si256((__m256i*)&weight[i]);

            __m256i a1 = _mm256_load_si256((__m256i*)&acc[i+16]);
            __m256i b1 = _mm256_load_si256((__m256i*)&weight[i+16]);


            a0 = _mm256_add_epi16(a0, b0);
            a1 = _mm256_add_epi16(a1, b1);

            _mm256_store_si256((__m256i*)&acc[i], a0);
            _mm256_store_si256((__m256i*)&acc[i+16], a1);
        }
    }

    void unset_input(int index) {
        if (!incremental_updates) {
            return;
        }

        int16_t *acc = acculumator;
        int16_t *weight = &weights->weights[index*NEURONS];

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&acc[i]);
            __m256i b0 = _mm256_load_si256((__m256i*)&weight[i]);

            __m256i a1 = _mm256_load_si256((__m256i*)&acc[i+16]);
            __m256i b1 = _mm256_load_si256((__m256i*)&weight[i+16]);


            a0 = _mm256_sub_epi16(a0, b0);
            a1 = _mm256_sub_epi16(a1, b1);

            _mm256_store_si256((__m256i*)&acc[i], a0);
            _mm256_store_si256((__m256i*)&acc[i+16], a1);
        }

    }

    void update() {

        int16_t *acc = (int16_t*)__builtin_assume_aligned(acculumator, 64);
        int16_t *neuron = (int16_t*)__builtin_assume_aligned(neurons, 64);

        __m256i minv = _mm256_set1_epi16(0);
        __m256i maxv = _mm256_set1_epi16(halfkp_quantization_fractions);

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&acc[i]);
            __m256i a1 = _mm256_load_si256((__m256i*)&acc[i+16]);

            a0 = _mm256_max_epi16(a0, minv);
            a1 = _mm256_max_epi16(a1, minv);
            a0 = _mm256_min_epi16(a0, maxv);
            a1 = _mm256_min_epi16(a1, maxv);

            _mm256_store_si256((__m256i*)&neuron[i], a0);
            _mm256_store_si256((__m256i*)&neuron[i+16], a1);
        }
    }

    void push_acculumator() {
        int16_t *new_acculumator = &acculumator[NEURONS];
        for (int i = 0; i < NEURONS; i += 32) {
            _mm256_store_si256((__m256i*)&new_acculumator[i], _mm256_load_si256((__m256i*)&acculumator[i]));
            _mm256_store_si256((__m256i*)&new_acculumator[i+16], _mm256_load_si256((__m256i*)&acculumator[i+16]));
        }
        acculumator = new_acculumator;
    }

    void pop_acculumator() {
        acculumator -= NEURONS;
    }

    int16_t *neurons;
    int16_t *acculumator;

    int16_t *neurons_buffer;
    int16_t *acculumator_buffer;

    bool incremental_updates;

    nnue_perspective_weights<INPUTS, NEURONS> *weights;
};



