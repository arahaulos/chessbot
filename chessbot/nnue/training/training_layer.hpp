#pragma once

#include <cmath>

#include <x86gprintrin.h>
#include <x86intrin.h>

#include "network_defs.hpp"


template<typename T>
T* align_ptr(T *buffer_ptr)
{
    size_t ptr = (size_t)buffer_ptr;
    return (T*)(ptr + 64 - (ptr % 64));
}


template <int N>
inline void copy_vectorized(float *ov, const float *av, int tid, int tc)
{
    if (N < 512) {
        if (tid == 0) {
            tc = 1;
        } else {
            return;
        }
    }
    int per_thread = N / tc;
    int start = per_thread * tid;

    for (int i = start; i < start + per_thread; i += 8) {
        _mm256_store_ps(&ov[i], _mm256_load_ps(&av[i]));
    }
}

template <int N>
inline void add_vectorized(float *ov, const float *av, const float *bv, int tid, int tc)
{
    if (N < 512) {
        if (tid == 0) {
            tc = 1;
        } else {
            return;
        }
    }
    int per_thread = N / tc;
    int start = per_thread * tid;

    for (int i = start; i < start + per_thread; i += 8) {
        __m256 a = _mm256_load_ps(&av[i]);
        __m256 b = _mm256_load_ps(&bv[i]);
        __m256 o = _mm256_add_ps(a, b);
        _mm256_store_ps(&ov[i], o);
    }
}

template <int N>
inline void mult_vectorized(float *ov, const float *av, const float *bv, int tid, int tc)
{
    if (N < 512) {
        if (tid == 0) {
            tc = 1;
        } else {
            return;
        }
    }

    int per_thread = N / tc;
    int start = per_thread * tid;

    for (int i = start; i < start + per_thread; i += 8) {
        __m256 a = _mm256_load_ps(&av[i]);
        __m256 b = _mm256_load_ps(&bv[i]);
        __m256 o = _mm256_mul_ps(a, b);
        _mm256_store_ps(&ov[i], o);
    }
}


template <int N>
inline void mult_vectorized(float *ov, const float *av, float value, int tid, int tc)
{
    if (N < 512) {
        if (tid == 0) {
            tc = 1;
        } else {
            return;
        }
    }

    int per_thread = N / tc;
    int start = per_thread * tid;

    __m256 b = _mm256_set1_ps(value);

    for (int i = start; i < start + per_thread; i += 8) {
        __m256 a = _mm256_load_ps(&av[i]);
        __m256 o = _mm256_mul_ps(a, b);
        _mm256_store_ps(&ov[i], o);
    }
}


template <int N>
inline void set_vectorized(float *ov, float value, int tid, int tc)
{
    if (N < 512) {
        if (tid == 0) {
            tc = 1;
        } else {
            return;
        }
    }

    int per_thread = N / tc;
    int start = per_thread * tid;

    __m256 v = _mm256_set1_ps(value);

    for (int i = start; i < start + per_thread; i += 8) {
        _mm256_store_ps(&ov[i], v);
    }
}


template <int N>
inline void ema_vectorized(float *ov, float *av, float val, int tid, int tc)
{
    if (N < 512) {
        if (tid == 0) {
            tc = 1;
        } else {
            return;
        }
    }

    int per_thread = N / tc;
    int start = per_thread * tid;

    __m256 f0 = _mm256_set1_ps(val);
    __m256 f1 = _mm256_set1_ps(1.0f - val);

    for (int i = start; i < start + per_thread; i += 8) {
        __m256 a = _mm256_load_ps(&ov[i]);
        __m256 b = _mm256_load_ps(&av[i]);

        a = _mm256_mul_ps(a, f0);
        b = _mm256_mul_ps(b, f1);

        _mm256_store_ps(&ov[i], _mm256_add_ps(a, b));
    }
}

inline __m256 inv_sqrt_plus_eps(__m256 v, float eps) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 veps = _mm256_set1_ps(eps);

    __m256 is_zero = _mm256_cmp_ps(v, _mm256_set1_ps(0.0f), _CMP_EQ_OQ);
    __m256 r = _mm256_rsqrt_ps(v);
    __m256 t = _mm256_fmadd_ps(veps, r, one);

    __m256 inv_t = _mm256_rcp_ps(t);
    __m256 two = _mm256_set1_ps(2.0f);
    inv_t = _mm256_mul_ps(inv_t, _mm256_fnmadd_ps(t, inv_t, two));

    __m256 denom_inv = _mm256_mul_ps(r, inv_t);

    __m256 lim = _mm256_set1_ps(1.0f / eps);
    denom_inv = _mm256_blendv_ps(denom_inv, lim, is_zero);

    return denom_inv;
}


inline float sigmoid(float x)
{
    float ex = std::exp(x);
    return (ex/(1.0+ex));
}

inline float sigmoid_inv(float y)
{
    return std::log(y/(1.0f-y));
}

inline float activation_diff(float x)
{
    if (x < 0.0f || x > 1.0f) {
        return 0.0f;
        //return 0.01f;
    } else {
        return 1.0f;
    }
}

inline float activation_diff_out(float x)
{
    return 1.0f;
}


inline float activation_func(float x)
{
    return std::clamp(x, 0.0f, 1.0f);
    //return std::clamp(x, 0.01f*x, 1.0f + (x - 1.0f)*0.01f);
}

inline float activation_func_out(float x)
{
    return x;
}


template <int INPUTS, int NEURONS, int STACK_SIZE, bool IS_OUTPUT_LAYER>
struct training_layer_weights
{
    training_layer_weights() {
        weights_buffer = new float[INPUTS*NEURONS*STACK_SIZE + 64];
        biases_buffer = new float[NEURONS*STACK_SIZE + 64];

        weights = align_ptr(weights_buffer);
        biases = align_ptr(biases_buffer);
    }

    ~training_layer_weights() {
        delete [] weights_buffer;
        delete [] biases_buffer;
    }

    void save(std::ostream &stream) {
        stream.write((char*)weights, (NEURONS*INPUTS*STACK_SIZE)*sizeof(float));
        stream.write((char*)biases, NEURONS*STACK_SIZE*sizeof(float));
    }

    void load(std::istream &stream) {
        stream.read((char*)weights, (NEURONS*INPUTS*STACK_SIZE)*sizeof(float));
        stream.read((char*)biases, NEURONS*STACK_SIZE*sizeof(float));
    }

    void save_quantized(int16_t *data, size_t &index) {
        int fracs = layer_quantization_fractions;
        float quant_correction_frac = 1.0f;


        if (INPUTS == 2*num_perspective_neurons && IS_OUTPUT_LAYER) {
            fracs = output_quantization_fractions;

            quant_correction_frac = (double)output_quantization_fractions / (double)halfkp_quantization_fractions;

        } else if (INPUTS == 2*num_perspective_neurons) {
            quant_correction_frac = (double)layer_quantization_fractions / (double)halfkp_quantization_fractions;
        } else if (IS_OUTPUT_LAYER) {
            fracs = output_quantization_fractions;

            quant_correction_frac = (double)output_quantization_fractions / (double)layer_quantization_fractions;
        }


        float clamp_val = 32000.0f / fracs;

        for (int i = 0; i < NEURONS*STACK_SIZE; i++) {
            for (int j = 0; j < INPUTS; j++) {
                float val = std::clamp(weights[i*INPUTS + j]*quant_correction_frac, -clamp_val, clamp_val);
                int16_t qval = round(val*fracs);
                data[index++] = qval;
            }
        }
        for (int i = 0; i < NEURONS*STACK_SIZE; i++) {
            float val = std::clamp(biases[i], -clamp_val, clamp_val);
            int16_t qval = round(val*fracs);
            data[index++] = qval;
        }
    }

    void broadcast_bucket_weights(int bucket)
    {
        float *src_weights = &weights[INPUTS*NEURONS*bucket];
        float *src_biases = &biases[NEURONS*bucket];

        for (int i = 0; i < STACK_SIZE; i++) {
            if (i == bucket) {
                continue;
            }

            float *dst_weights = &weights[INPUTS*NEURONS*i];
            float *dst_biases = &biases[NEURONS*i];

            for (int j = 0; j < INPUTS*NEURONS; j++) {
                dst_weights[j] = src_weights[j];
            }
            for (int j = 0; j < NEURONS; j++) {
                dst_biases[j] = src_biases[j];
            }
        }

    }

    void copy_from(const training_layer_weights<INPUTS, NEURONS, STACK_SIZE, IS_OUTPUT_LAYER> &other, int tid, int tc) {
        copy_vectorized<NEURONS*STACK_SIZE>(biases, other.biases, tid, tc);
        copy_vectorized<NEURONS*INPUTS*STACK_SIZE>(weights, other.weights, tid, tc);
    }

    void zero(int tid, int tc)
    {
        set_vectorized<NEURONS*STACK_SIZE>(biases, 0.0f, tid, tc);
        set_vectorized<NEURONS*INPUTS*STACK_SIZE>(weights, 0.0f, tid, tc);
    }

    void mult(float val, int tid, int tc) {
        mult_vectorized<NEURONS*STACK_SIZE>(biases, biases, val, tid, tc);
        mult_vectorized<NEURONS*INPUTS*STACK_SIZE>(weights, weights, val, tid, tc);
    }

    void mult(const training_layer_weights<INPUTS, NEURONS, STACK_SIZE, IS_OUTPUT_LAYER> &other, float val, int tid, int tc) {
        mult_vectorized<NEURONS*STACK_SIZE>(biases, other.biases, val, tid, tc);
        mult_vectorized<NEURONS*INPUTS*STACK_SIZE>(weights, other.weights, val, tid, tc);
    }

    void squared(const training_layer_weights<INPUTS, NEURONS, STACK_SIZE, IS_OUTPUT_LAYER> &other, int tid, int tc) {
        mult_vectorized<NEURONS*STACK_SIZE>(biases, other.biases, other.biases, tid, tc);
        mult_vectorized<NEURONS*INPUTS*STACK_SIZE>(weights, other.weights, other.weights, tid, tc);
    }

    void exponential_smoothing(const training_layer_weights<INPUTS, NEURONS, STACK_SIZE, IS_OUTPUT_LAYER> &other, float factor, int tid, int tc) {
        ema_vectorized<NEURONS*STACK_SIZE>(biases, other.biases, factor, tid, tc);
        ema_vectorized<NEURONS*INPUTS*STACK_SIZE>(weights, other.weights, factor, tid, tc);
    }

    int num_of_biases() const {
        return NEURONS*STACK_SIZE;
    }

    int num_of_weights() const {
        return NEURONS*INPUTS*STACK_SIZE;
    }

    int num_of_quantized_params() const {
        return NEURONS*STACK_SIZE + (NEURONS*INPUTS*STACK_SIZE);
    }

    void add(const training_layer_weights<INPUTS, NEURONS, STACK_SIZE, IS_OUTPUT_LAYER> &other, int tid, int tc) {
        add_vectorized<NEURONS*STACK_SIZE>(biases, biases, other.biases, tid, tc);
        add_vectorized<NEURONS*INPUTS*STACK_SIZE>(weights, weights, other.weights, tid, tc);
    }

    void rmsprop(training_layer_weights<INPUTS, NEURONS, STACK_SIZE, IS_OUTPUT_LAYER> *grad, training_layer_weights<INPUTS, NEURONS, STACK_SIZE, IS_OUTPUT_LAYER> *pg, float learning_rate, int tid, int tc)
    {
        float clamp_min = layer_quantization_clamp_min;
        float clamp_max = layer_quantization_clamp_max;
        if (IS_OUTPUT_LAYER) {
            clamp_min = output_quantization_clamp_min;
            clamp_max = output_quantization_clamp_max;
        }

        __m256 cmin = _mm256_set1_ps(clamp_min);
        __m256 cmax = _mm256_set1_ps(clamp_max);

        __m256 lr = _mm256_set1_ps(learning_rate);

        float epsilon = 0.0000001f;

        int per_thread = INPUTS*NEURONS*STACK_SIZE / tc;
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
            for (int i = 0; i < NEURONS*STACK_SIZE; i += 8) {

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
    }



    float *weights;
    float *biases;

    float *weights_buffer;
    float *biases_buffer;
};



template <int IN, int OUT, int STACK_SIZE, bool IS_OUTPUT_LAYER>
struct training_layer
{
    training_layer(training_layer_weights<IN, OUT, STACK_SIZE, IS_OUTPUT_LAYER> *w) {
        weights = w;
        neurons_buffer = new float[OUT + 64];
        acculumator_buffer = new float[OUT + 64];
        grads_buffer = new float[OUT + 64];

        neurons = align_ptr(neurons_buffer);
        acculumator = align_ptr(acculumator_buffer);
        grads = align_ptr(grads_buffer);
    }

    ~training_layer() {
        delete [] neurons_buffer;
        delete [] acculumator_buffer;
        delete [] grads_buffer;
    }


    void reset() {
        for (int i = 0; i < OUT; i++) {
            neurons[i] = 0;
        }
    }


    void update(int neuron, int bucket, float *prev_layer) {
        __m256 n = _mm256_set1_ps(0.0f);

        for (int j = 0; j < IN; j += 8) {
            __m256 l0 = _mm256_load_ps(&prev_layer[j]);
            __m256 w = _mm256_load_ps(&weights->weights[IN*OUT*bucket + neuron*IN + j]);

            n = _mm256_fmadd_ps(l0, w, n);
        }

        __m128 a = _mm256_extractf128_ps(n, 0);
        __m128 b = _mm256_extractf128_ps(n, 1);

        __m128 h = _mm_hadd_ps(a, b);


        union {
            int i[4];
            float f[4];
        } data;

        data.i[0] = _mm_extract_ps(h, 0);
        data.i[1] = _mm_extract_ps(h, 1);
        data.i[2] = _mm_extract_ps(h, 2);
        data.i[3] = _mm_extract_ps(h, 3);

        acculumator[neuron] = data.f[0] + data.f[1] + data.f[2] + data.f[3] + weights->biases[OUT*bucket + neuron];

        if (IS_OUTPUT_LAYER) {
            neurons[neuron] = activation_func_out(acculumator[neuron]);
        } else {
            neurons[neuron] = activation_func(acculumator[neuron]);
        }
    }

    void update(int neuron, int bucket, float *prev_layer0, float *prev_layer1) {
        __m256 n = _mm256_set1_ps(0.0f);

        for (int j = 0; j < IN/2; j += 8) {
            __m256 l0 = _mm256_load_ps(&prev_layer0[j]);
            __m256 w = _mm256_load_ps(&weights->weights[IN*OUT*bucket + neuron*IN + j]);

            n = _mm256_fmadd_ps(l0, w, n);
        }

        for (int j = 0; j < IN/2; j += 8) {
            __m256 l0 = _mm256_load_ps(&prev_layer1[j]);
            __m256 w = _mm256_load_ps(&weights->weights[IN*OUT*bucket + neuron*IN + j + (IN/2)]);

            n = _mm256_fmadd_ps(l0, w, n);
        }

        __m128 a = _mm256_extractf128_ps(n, 0);
        __m128 b = _mm256_extractf128_ps(n, 1);

        __m128 h = _mm_hadd_ps(a, b);


        union {
            int i[4];
            float f[4];
        } data;

        data.i[0] = _mm_extract_ps(h, 0);
        data.i[1] = _mm_extract_ps(h, 1);
        data.i[2] = _mm_extract_ps(h, 2);
        data.i[3] = _mm_extract_ps(h, 3);

        acculumator[neuron] = data.f[0] + data.f[1] + data.f[2] + data.f[3] + weights->biases[OUT*bucket + neuron];

        if (IS_OUTPUT_LAYER) {
            neurons[neuron] = activation_func_out(acculumator[neuron]);
        } else {
            neurons[neuron] = activation_func(acculumator[neuron]);
        }
    }

    void update(int bucket, float *prev_layer) {
        for (int i = 0; i < OUT; i++) {
            update(i, bucket, prev_layer);
        }
    }

    void update(int bucket, float *prev_layer0, float *prev_layer1) {
        for (int i = 0; i < OUT; i++) {
            update(i, bucket, prev_layer0, prev_layer1);
        }
    }

    void back_propagate(int bucket, training_layer_weights<IN,OUT, STACK_SIZE,IS_OUTPUT_LAYER> *gradients, float *prev_layer_grads0, float *prev_layer_grads1, float *prev_layer_activations0, float *prev_layer_activations1) {

        for (int i = 0; i < IN/2; i++) {
            prev_layer_grads0[i] = 0;
            prev_layer_grads1[i] = 0;
        }

        for (int i = 0; i < OUT; i++) {
            //float a = neurons[i]; //Activvation
            float x = acculumator[i]; //Neurons input
            float dc = grads[i]; //cost diff
            float da = (IS_OUTPUT_LAYER ? activation_diff_out(x) : activation_diff(x)); //activation diff


            __m256 dcda = _mm256_set1_ps(dc*da);

            for (int j = 0; j < IN/2; j += 8) {
                __m256 l0 = _mm256_load_ps(&prev_layer_activations0[j]);
                __m256 w = _mm256_load_ps(&weights->weights[IN*OUT*bucket + i*IN + j]);
                __m256 l0_grads = _mm256_load_ps(&prev_layer_grads0[j]);
                __m256 g = _mm256_load_ps(&gradients->weights[IN*OUT*bucket + i*IN + j]);


                __m256 tmp = _mm256_fmadd_ps(dcda, w, l0_grads);
                __m256 tmp2 = _mm256_fmadd_ps(dcda, l0, g);

                _mm256_store_ps(&gradients->weights[IN*OUT*bucket + i*IN + j], tmp2);
                _mm256_store_ps(&prev_layer_grads0[j], tmp);
            }
            for (int j = 0; j < IN/2; j += 8) {
                __m256 l0 = _mm256_load_ps(&prev_layer_activations1[j]);
                __m256 w = _mm256_load_ps(&weights->weights[IN*OUT*bucket + i*IN + j + IN/2]);
                __m256 l0_grads = _mm256_load_ps(&prev_layer_grads1[j]);
                __m256 g = _mm256_load_ps(&gradients->weights[IN*OUT*bucket + i*IN + j + IN/2]);


                __m256 tmp = _mm256_fmadd_ps(dcda, w, l0_grads);
                __m256 tmp2 = _mm256_fmadd_ps(dcda, l0, g);

                _mm256_store_ps(&gradients->weights[IN*OUT*bucket + i*IN + j + IN/2], tmp2);
                _mm256_store_ps(&prev_layer_grads1[j], tmp);
            }

            gradients->biases[OUT*bucket + i] += dc*da;
        }
    }

    void back_propagate(int bucket, training_layer_weights<IN,OUT, STACK_SIZE,IS_OUTPUT_LAYER> *gradients, float *prev_layer_grads, float *prev_layer_activations) {

        for (int i = 0; i < IN; i++) {
            prev_layer_grads[i] = 0;
        }

        for (int i = 0; i < OUT; i++) {
            //float a = neurons[i]; //Activvation
            float x = acculumator[i]; //Neurons input
            float dc = grads[i]; //cost diff
            float da = (IS_OUTPUT_LAYER ? activation_diff_out(x) : activation_diff(x)); //activation diff

            __m256 dcda = _mm256_set1_ps(dc*da);

            for (int j = 0; j < IN; j += 8) {
                __m256 l0 = _mm256_load_ps(&prev_layer_activations[j]);
                __m256 w = _mm256_load_ps(&weights->weights[IN*OUT*bucket + i*IN + j]);
                __m256 l0_grads = _mm256_load_ps(&prev_layer_grads[j]);
                __m256 g = _mm256_load_ps(&gradients->weights[IN*OUT*bucket + i*IN + j]);

                __m256 tmp = _mm256_fmadd_ps(dcda, w, l0_grads);
                __m256 tmp2 = _mm256_fmadd_ps(dcda, l0, g);

                _mm256_store_ps(&prev_layer_grads[j], tmp);
                _mm256_store_ps(&gradients->weights[IN*OUT*bucket + i*IN + j], tmp2);
            }

            gradients->biases[OUT*bucket + i] += dc*da;
        }
    }

    float *neurons;
    float *acculumator;

    float *neurons_buffer;
    float *acculumator_buffer;

    float *grads;
    float *grads_buffer;


    training_layer_weights<IN, OUT, STACK_SIZE,IS_OUTPUT_LAYER> *weights;
};













