#pragma once

#include "training_layer.hpp"

#include "../nnue_defs.hpp"


template <int INPUTS, int NEURONS>
struct training_perspective_weights
{
    training_perspective_weights() {
        weights_buffer = new float[INPUTS*NEURONS + 64];
        biases_buffer = new float[NEURONS + 64];

        weights = align_ptr(weights_buffer);
        biases = align_ptr(biases_buffer);
    }

    ~training_perspective_weights() {
        delete [] weights_buffer;
        delete [] biases_buffer;
    }

    void save(std::ostream &stream) {
        stream.write((char*)weights, (NEURONS*INPUTS)*sizeof(float));
        stream.write((char*)biases, NEURONS*sizeof(float));
    }

    void save_quantized(int16_t *data, size_t &index) {

        float clamp_val = 32000.0f / halfkp_quantization_fractions;

        for (size_t j = 0; j < num_perspective_inputs; j++) {
            for (size_t i = 0; i < NEURONS; i++) {
                float w = weights[j*NEURONS + i];

                int fi = num_of_king_buckets*inputs_per_bucket + (j % inputs_per_bucket);

                float fw = weights[fi*NEURONS + i];

                float val = std::clamp(w+fw, -clamp_val, clamp_val);
                int16_t qval = round(val*halfkp_quantization_fractions);
                data[index++] = qval;
            }
        }
        for (int i = 0; i < NEURONS; i++) {
            float val = std::clamp(biases[i], -clamp_val, clamp_val);
            int16_t qval = round(val*halfkp_quantization_fractions);
            data[index++] = qval;
        }
    }

    void load(std::istream &stream) {
        stream.read((char*)weights, (NEURONS*INPUTS)*sizeof(float));
        stream.read((char*)biases, NEURONS*sizeof(float));
    }

    void copy_from(const training_perspective_weights<INPUTS, NEURONS> &other, int tid, int tc) {
        copy_vectorized<NEURONS>(biases, other.biases, tid, tc);
        copy_vectorized<NEURONS*INPUTS>(weights, other.weights, tid, tc);
    }

    void zero(int tid, int tc)
    {
        set_vectorized<NEURONS>(biases, 0.0f, tid, tc);
        set_vectorized<NEURONS*INPUTS>(weights, 0.0f, tid, tc);
    }

    void add(const training_perspective_weights<INPUTS, NEURONS> &other, int tid, int tc) {
        add_vectorized<NEURONS>(biases, biases, other.biases, tid, tc);
        add_vectorized<NEURONS*INPUTS>(weights, weights, other.weights, tid, tc);
    }

    void mult(float val, int tid, int tc) {
        mult_vectorized<NEURONS>(biases, biases, val, tid, tc);
        mult_vectorized<NEURONS*INPUTS>(weights, weights, val, tid, tc);
    }

    void mult(const training_perspective_weights<INPUTS, NEURONS> &other, float val, int tid, int tc) {
        mult_vectorized<NEURONS>(biases, other.biases, val, tid, tc);
        mult_vectorized<NEURONS*INPUTS>(weights, other.weights, val, tid, tc);
    }

    void squared(const training_perspective_weights<INPUTS, NEURONS> &other, int tid, int tc) {
        mult_vectorized<NEURONS>(biases, other.biases, other.biases, tid, tc);
        mult_vectorized<NEURONS*INPUTS>(weights, other.weights, other.weights, tid, tc);
    }

    void exponential_smoothing(const training_perspective_weights<INPUTS, NEURONS> &other, float factor, int tid, int tc) {
        ema_vectorized<NEURONS>(biases, other.biases, factor, tid, tc);
        ema_vectorized<NEURONS*INPUTS>(weights, other.weights, factor, tid, tc);
    }

    int num_of_biases() const {
        return NEURONS;
    }

    int num_of_weights() const {
        return NEURONS*INPUTS;
    }

    int num_of_quantized_params() const {
        return NEURONS + (num_perspective_inputs*NEURONS);
    }

    void rmsprop(training_perspective_weights<INPUTS, NEURONS> *grad, training_perspective_weights<INPUTS, NEURONS> *pg, float learning_rate, int tid, int tc)
    {
        float epsilon = 0.0000001f;

        #if USE_AVX2

        __m256 cmin = _mm256_set1_ps(halfkp_quantization_clamp_min);
        __m256 cmax = _mm256_set1_ps(halfkp_quantization_clamp_max);

        __m256 lr = _mm256_set1_ps(learning_rate);

        int per_thread = INPUTS*NEURONS / tc;
        int start = per_thread*tid;

        for (int i = start; i < start + per_thread; i += 8) {
            __m256 w = _mm256_load_ps(&weights[i]);
            __m256 g = _mm256_load_ps(&grad->weights[i]);

            __m256 v = _mm256_load_ps(&pg->weights[i]);

            __m256 mul = _mm256_mul_ps(lr, inv_sqrt_plus_eps(v, epsilon));

            __m256 nw = _mm256_fnmadd_ps(g, mul, w);

            nw = _mm256_min_ps(nw, cmax);
            nw = _mm256_max_ps(nw, cmin);

            _mm256_store_ps(&weights[i], nw);
        }

        if (tid == 0) {
            for (int i = 0; i < NEURONS; i += 8) {

                __m256 b = _mm256_load_ps(&biases[i]);
                __m256 g = _mm256_load_ps(&grad->biases[i]);

                __m256 v = _mm256_load_ps(&pg->biases[i]);

                __m256 mul = _mm256_mul_ps(lr, inv_sqrt_plus_eps(v, epsilon));

                __m256 nb = _mm256_fnmadd_ps(g, mul, b);

                nb = _mm256_min_ps(nb, cmax);
                nb = _mm256_max_ps(nb, cmin);

                _mm256_store_ps(&biases[i], nb);
            }
        }

        #else

        int per_thread = INPUTS*NEURONS / tc;
        int start = per_thread*tid;

        for (int i = start; i < start + per_thread; i++) {
            weights[i] -= grad->weights[i]*learning_rate / sqrt(pg->weights[i] + epsilon);
        }
        if (tid == 0) {
            for (int i = 0; i < NEURONS; i++) {
                biases[i] -= grad->biases[i]*learning_rate / sqrt(pg->biases[i] + epsilon);
            }
        }

        #endif
    }



    void coalesce_factorizer_weights() {
        for (size_t i = 0; i < num_of_king_buckets; i++) {
            for (size_t j = 0; j < inputs_per_bucket; j++) {
                for (size_t k = 0; k < NEURONS; k++) {
                    size_t input_index = i*inputs_per_bucket + j;
                    size_t factorizer_input_index = num_of_king_buckets*inputs_per_bucket + j;

                    weights[input_index*NEURONS + k] += weights[factorizer_input_index*NEURONS + k];
                }
            }
        }
        for (size_t i = 0; i < inputs_per_bucket; i++) {
            for (size_t j = 0; j < NEURONS; j++) {
                size_t factorizer_input_index = num_of_king_buckets*inputs_per_bucket + i;

                weights[factorizer_input_index*NEURONS + j] = 0;
            }
        }
    }


    float *weights;
    float *biases;

    float *weights_buffer;
    float *biases_buffer;
};



template <int INPUTS, int NEURONS>
struct training_perspective
{
    training_perspective(training_perspective_weights<INPUTS, NEURONS> *w) {
        weights = w;
        neurons_buffer = new float[NEURONS + 64];
        acculumator_buffer = new float[NEURONS + 64];
        grads_buffer = new float[NEURONS + 64];
        active_inputs_buffer = new int[INPUTS  + 64];

        neurons = align_ptr(neurons_buffer);
        acculumator = align_ptr(acculumator_buffer);
        grads = align_ptr(grads_buffer);
        active_inputs = align_ptr(active_inputs_buffer);

        king_bucket = -1;
    }

    ~training_perspective() {
        delete [] neurons_buffer;
        delete [] acculumator_buffer;
        delete [] grads_buffer;
        delete [] active_inputs_buffer;
    }


    void reset() {
        num_of_active_inputs = 0;

        #if USE_AVX2

        for (int i = 0; i < NEURONS; i += 8) {
            __m256 tmp = _mm256_load_ps(&weights->biases[i]);

            _mm256_store_ps(&acculumator[i], tmp);
        }

        #else

        for (int i = 0; i < NEURONS; i++) {
            acculumator[i] = weights->biases[i];
        }

        #endif
    }

    void set_input(int index) {
        float *w = &weights->weights[index*NEURONS];

        #if USE_AVX2

        for (int i = 0; i < NEURONS; i += 8) {
            __m256 a = _mm256_load_ps(&acculumator[i]);
            __m256 b = _mm256_load_ps(&w[i]);

            __m256 tmp = _mm256_add_ps(a, b);

            _mm256_store_ps(&acculumator[i], tmp);
        }

        #else

        for (int i = 0; i < NEURONS; i++) {
            acculumator[i] += w[i];
        }

        #endif

        active_inputs[num_of_active_inputs++] = index;
    }


    void prefetch_input(int index) {
        constexpr int cacheline_size = 64;
        float *w = &weights->weights[index*NEURONS];

        for (int i = 0; i < NEURONS; i += cacheline_size / sizeof(float)) {
            __builtin_prefetch(&w[i]);
        }
    }


    void update() {
        #if USE_AVX2

        __m256 cmin = _mm256_set1_ps(0.0f);
        __m256 cmax = _mm256_set1_ps(1.0f);

        for (int i = 0; i < NEURONS; i += 8) {
            __m256 tmp = _mm256_load_ps(&acculumator[i]);
            tmp = _mm256_min_ps(tmp, cmax);
            tmp = _mm256_max_ps(tmp, cmin);
            _mm256_store_ps(&neurons[i], tmp);

        }

        #else

        for (int i = 0; i < NEURONS; i++) {
            neurons[i] = std::clamp(acculumator[i], 0.0f, 1.0f);
        }

        #endif

    }

    void back_propagate(training_perspective_weights<INPUTS, NEURONS> *gradients) {

        #if USE_AVX2


        __m256 zero = _mm256_set1_ps(0.0f);
        __m256 one = _mm256_set1_ps(1.0f);

        for (int i = 0; i < NEURONS; i += 8) {
            __m256 x = _mm256_load_ps(&acculumator[i]);
            __m256 dc = _mm256_load_ps(&grads[i]);

            __m256 cmp_ge = _mm256_cmp_ps(x, zero, _CMP_GE_OS); // x >= 0.0f
            __m256 cmp_le = _mm256_cmp_ps(x, one, _CMP_LE_OS);  // x <= 1.0f
            __m256 mask = _mm256_and_ps(cmp_ge, cmp_le);    // Combine comparisons

             __m256 da = _mm256_blendv_ps(zero, one, mask);

            _mm256_store_ps(&grads[i], _mm256_mul_ps(da, dc));

        }

        for (int i = 0; i < num_of_active_inputs; i++) {
            int input = active_inputs[i];

            for (int j = 0; j < NEURONS; j += 8) {
                __m256 g0 = _mm256_load_ps(&grads[j]);
                __m256 g1 = _mm256_load_ps(&gradients->weights[input*NEURONS + j]);

                _mm256_store_ps(&gradients->weights[input*NEURONS + j], _mm256_add_ps(g0, g1));

            }
        }

        for (int i = 0; i < NEURONS; i += 8) {
            __m256 g0 = _mm256_load_ps(&grads[i]);
            __m256 g1 = _mm256_load_ps(&gradients->biases[i]);

            _mm256_store_ps(&gradients->biases[i], _mm256_add_ps(g0, g1));

        }

        #else

        for (int i = 0; i < NEURONS; i++) {
            float x = acculumator[i];
            float dc = grads[i];
            float da = ((x > 1.0f || x < 0.0f) ? 0.0f : x);

            grads[i] = dc*da;
        }

        for (int i = 0; i < num_of_active_inputs; i++) {
            int input = active_inputs[i];
            for (int j = 0; j < NEURONS; j++) {
                gradients->weights[input*NEURONS + j] += grads[j];
            }
        }
        for (int i = 0; i < NEURONS; i++) {
            gradients->biases[i] += grads[i];
        }

        #endif
    }

    float *neurons;
    float *acculumator;
    float *grads;

    int *active_inputs;

    float *neurons_buffer;
    float *acculumator_buffer;
    float *grads_buffer;
    int *active_inputs_buffer;

    int num_of_active_inputs;

    int king_bucket;
    bool use_factorizer;

    training_perspective_weights<INPUTS, NEURONS> *weights;
};
