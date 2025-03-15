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
inline void add_vectorized(float *ov, const float *av, const float *bv)
{
    for (int i = 0; i < N; i += 8) {
        __m256 a = _mm256_load_ps(&av[i]);
        __m256 b = _mm256_load_ps(&bv[i]);
        __m256 o = _mm256_add_ps(a, b);
        _mm256_store_ps(&ov[i], o);
    }
}

template <int N>
inline void mult_vectorized(float *ov, const float *av, const float *bv)
{
    for (int i = 0; i < N; i += 8) {
        __m256 a = _mm256_load_ps(&av[i]);
        __m256 b = _mm256_load_ps(&bv[i]);
        __m256 o = _mm256_mul_ps(a, b);
        _mm256_store_ps(&ov[i], o);
    }
}


template <int N>
inline void mult_vectorized(float *ov, const float *av, float value)
{
    __m256 b = _mm256_set1_ps(value);

    for (int i = 0; i < N; i += 8) {
        __m256 a = _mm256_load_ps(&av[i]);
        __m256 o = _mm256_mul_ps(a, b);
        _mm256_store_ps(&ov[i], o);
    }
}


template <int N>
inline void set_vectorized(float *ov, float value)
{
    __m256 v = _mm256_set1_ps(value);

    for (int i = 0; i < N; i += 8) {
        _mm256_store_ps(&ov[i], v);
    }
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
}

inline float activation_func_out(float x)
{
    return x;
}


template <int INPUTS, int NEURONS>
struct training_layer_weights
{
    training_layer_weights() {
        weights_buffer = new float[INPUTS*NEURONS + 64];
        biases_buffer = new float[NEURONS + 64];

        weights = align_ptr(weights_buffer);
        biases = align_ptr(biases_buffer);
    }

    ~training_layer_weights() {
        delete [] weights_buffer;
        delete [] biases_buffer;
    }

    void save(std::ostream &stream) {
        stream.write((char*)weights, (NEURONS*INPUTS)*sizeof(float));
        stream.write((char*)biases, NEURONS*sizeof(float));
    }

    void load(std::istream &stream) {
        stream.read((char*)weights, (NEURONS*INPUTS)*sizeof(float));
        stream.read((char*)biases, NEURONS*sizeof(float));
    }


    void save_quantized(std::ostream &stream) {
        float clamp_min = layer_quantization_clamp_min;
        float clamp_max = layer_quantization_clamp_max;
        int fracs = layer_quantization_fractions;
        float quant_correction_frac = 1.0f;

        if (INPUTS == 2*num_perspective_neurons && NEURONS == 1) {
            clamp_min = output_quantization_clamp_min;
            clamp_max = output_quantization_clamp_max;
            fracs = output_quantization_fractions;

            quant_correction_frac = (double)output_quantization_fractions / (double)halfkp_quantization_fractions;

        } else if (INPUTS == 2*num_perspective_neurons) {
            quant_correction_frac = (double)layer_quantization_fractions / (double)halfkp_quantization_fractions;
        } else if (NEURONS == 1) {
            clamp_min = output_quantization_clamp_min;
            clamp_max = output_quantization_clamp_max;
            fracs = output_quantization_fractions;

            quant_correction_frac = (double)output_quantization_fractions / (double)layer_quantization_fractions;
        }



        for (int i = 0; i < NEURONS; i++) {
            for (int j = 0; j < INPUTS; j++) {
                float val = std::clamp(weights[i*INPUTS + j]*quant_correction_frac, clamp_min, clamp_max);
                int16_t qval = round(val*fracs);
                stream.write((char*)&qval, sizeof(int16_t));
            }
        }
        for (int i = 0; i < NEURONS; i++) {
            float val = std::clamp(biases[i], clamp_min, clamp_max);
            int16_t qval = round(val*fracs);
            stream.write((char*)&qval, sizeof(int16_t));
        }
    }

    void copy_from(const training_layer_weights<INPUTS, NEURONS> &other) {
        for (int i = 0; i < NEURONS; i++) {
            biases[i] = other.biases[i];
        }
        for (int i = 0; i < NEURONS*INPUTS; i++) {
            weights[i] = other.weights[i];
        }
    }

    void zero()
    {
        set_vectorized<NEURONS>(biases, 0.0f);
        set_vectorized<NEURONS*INPUTS>(weights, 0.0f);
    }

    void mult(float val) {
        mult_vectorized<NEURONS>(biases, biases, val);
        mult_vectorized<NEURONS*INPUTS>(weights, weights, val);
    }

    void normalize()
    {
        double sum_sqr = 0;
        for (int i = 0; i < NEURONS*INPUTS; i++) {
            sum_sqr += weights[i]*weights[i];
        }
        for (int i = 0; i < NEURONS; i++) {
            sum_sqr += biases[i]*biases[i];
        }
        mult(1.0f/sqrt(sum_sqr));
    }

    double sum_sq()
    {
        double sum_sqr = 0;
        for (int i = 0; i < NEURONS*INPUTS; i++) {
            sum_sqr += weights[i]*weights[i];
        }
        for (int i = 0; i < NEURONS; i++) {
            sum_sqr += biases[i]*biases[i];
        }
        return sum_sqr;
    }

    void squared(const training_layer_weights<INPUTS, NEURONS> &other) {
        mult_vectorized<NEURONS*INPUTS>(weights, other.weights, other.weights);
        mult_vectorized<NEURONS>(biases, other.biases, other.biases);
    }

    int num_of_biases() const {
        return NEURONS;
    }

    int num_of_weights() const {
        return NEURONS*INPUTS;
    }

    void add(const training_layer_weights<INPUTS, NEURONS> &other) {
        add_vectorized<NEURONS>(biases, biases, other.biases);
        add_vectorized<NEURONS*INPUTS>(weights, weights, other.weights);
    }

    void gradient_descent(training_layer_weights<INPUTS, NEURONS> *grad, float learning_rate)
    {
        float clamp_min = layer_quantization_clamp_min;
        float clamp_max = layer_quantization_clamp_max;
        if (NEURONS == 1) {
            clamp_min = output_quantization_clamp_min;
            clamp_max = output_quantization_clamp_max;
        }

        __m256 cmin = _mm256_set1_ps(clamp_min);
        __m256 cmax = _mm256_set1_ps(clamp_max);

        __m256 lr = _mm256_set1_ps(learning_rate);

        for (int i = 0; i < INPUTS*NEURONS; i += 8) {
            __m256 w = _mm256_load_ps(&weights[i]);
            __m256 g = _mm256_load_ps(&grad->weights[i]);

            /*__m256 fg = _mm256_mul_ps(g, lr);
            __m256 nw = _mm256_sub_ps(w, fg);*/
            __m256 nw = _mm256_fnmadd_ps(g, lr, w);

            nw = _mm256_min_ps(nw, cmax);
            nw = _mm256_max_ps(nw, cmin);

            _mm256_store_ps(&weights[i], nw);
        }
        for (int i = 0; i < NEURONS; i += 8) {

            __m256 b = _mm256_load_ps(&biases[i]);
            __m256 g = _mm256_load_ps(&grad->biases[i]);

            /*__m256 fg = _mm256_mul_ps(g, lr);
            __m256 nb = _mm256_sub_ps(b, fg);*/
            __m256 nb = _mm256_fnmadd_ps(g, lr, b);

            _mm256_store_ps(&biases[i], nb);
        }
    }


    void rmsprop(training_layer_weights<INPUTS, NEURONS> *grad, training_layer_weights<INPUTS, NEURONS> *pg, float learning_rate)
    {
        float clamp_min = layer_quantization_clamp_min;
        float clamp_max = layer_quantization_clamp_max;
        if (NEURONS == 1) {
            clamp_min = output_quantization_clamp_min;
            clamp_max = output_quantization_clamp_max;
        }

        __m256 cmin = _mm256_set1_ps(clamp_min);
        __m256 cmax = _mm256_set1_ps(clamp_max);

        __m256 lr = _mm256_set1_ps(learning_rate);

        __m256 epsilon = _mm256_set1_ps(0.0000001f);

        for (int i = 0; i < INPUTS*NEURONS; i += 8) {
            __m256 w = _mm256_load_ps(&weights[i]);
            __m256 g = _mm256_load_ps(&grad->weights[i]);

            __m256 v = _mm256_load_ps(&pg->weights[i]);

            __m256 mul = _mm256_mul_ps(lr, _mm256_rsqrt_ps(_mm256_add_ps(v, epsilon)));

            __m256 nw = _mm256_fnmadd_ps(g, mul, w);

            nw = _mm256_min_ps(nw, cmax);
            nw = _mm256_max_ps(nw, cmin);

            _mm256_store_ps(&weights[i], nw);
        }


        for (int i = 0; i < NEURONS; i += 8) {

            __m256 b = _mm256_load_ps(&biases[i]);
            __m256 g = _mm256_load_ps(&grad->biases[i]);

            __m256 v = _mm256_load_ps(&pg->biases[i]);

            __m256 mul = _mm256_mul_ps(lr, _mm256_rsqrt_ps(_mm256_add_ps(v, epsilon)));

            __m256 nb = _mm256_fnmadd_ps(g, mul, b);

            nb = _mm256_min_ps(nb, cmax);
            nb = _mm256_max_ps(nb, cmin);

            _mm256_store_ps(&biases[i], nb);
        }
    }



    float *weights;
    float *biases;

    float *weights_buffer;
    float *biases_buffer;
};



template <int IN, int OUT>
struct training_layer
{
    training_layer(training_layer_weights<IN, OUT> *w) {
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

    void update(float *prev_layer) {
        for (int i = 0; i < OUT; i++) {
            __m256 n = _mm256_set1_ps(0.0f);

            for (int j = 0; j < IN; j += 8) {
                __m256 l0 = _mm256_load_ps(&prev_layer[j]);
                __m256 w = _mm256_load_ps(&weights->weights[i*IN + j]);

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

            acculumator[i] = data.f[0] + data.f[1] + data.f[2] + data.f[3] + weights->biases[i];

            if (OUT == 1) {
                neurons[i] = activation_func_out(acculumator[i]);
            } else {
                neurons[i] = activation_func(acculumator[i]);
            }
        }
    }

    void update(float *prev_layer0, float *prev_layer1) {
        for (int i = 0; i < OUT; i++) {
            __m256 n = _mm256_set1_ps(0.0f);

            for (int j = 0; j < IN/2; j += 8) {
                __m256 l0 = _mm256_load_ps(&prev_layer0[j]);
                __m256 w = _mm256_load_ps(&weights->weights[i*IN + j]);

                n = _mm256_fmadd_ps(l0, w, n);
            }

            for (int j = 0; j < IN/2; j += 8) {
                __m256 l0 = _mm256_load_ps(&prev_layer1[j]);
                __m256 w = _mm256_load_ps(&weights->weights[i*IN + j + (IN/2)]);

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

            acculumator[i] = data.f[0] + data.f[1] + data.f[2] + data.f[3] + weights->biases[i];

            if (OUT == 1) {
                neurons[i] = activation_func_out(acculumator[i]);
            } else {
                neurons[i] = activation_func(acculumator[i]);
            }
        }
    }


    void back_propagate(training_layer_weights<IN,OUT> *gradients, float *prev_layer_grads0, float *prev_layer_grads1, float *prev_layer_activations0, float *prev_layer_activations1) {

        for (int i = 0; i < IN/2; i++) {
            prev_layer_grads0[i] = 0;
            prev_layer_grads1[i] = 0;
        }

        for (int i = 0; i < OUT; i++) {
            //float a = neurons[i]; //Activvation
            float x = acculumator[i]; //Neurons input
            float dc = grads[i]; //cost diff
            float da = ((OUT == 1) ? activation_diff_out(x) : activation_diff(x)); //activation diff


            __m256 dcda = _mm256_set1_ps(dc*da);

            for (int j = 0; j < IN/2; j += 8) {
                __m256 l0 = _mm256_load_ps(&prev_layer_activations0[j]);
                __m256 w = _mm256_load_ps(&weights->weights[i*IN + j]);
                __m256 l0_grads = _mm256_load_ps(&prev_layer_grads0[j]);
                __m256 g = _mm256_load_ps(&gradients->weights[i*IN + j]);

                /*__m256 grad = _mm256_mul_ps(dcda, l0);
                __m256 grad_l = _mm256_mul_ps(dcda, w);

                __m256 tmp = _mm256_add_ps(l0_grads, grad_l);
                __m256 tmp2 = _mm256_add_ps(g, grad);*/

                __m256 tmp = _mm256_fmadd_ps(dcda, w, l0_grads);
                __m256 tmp2 = _mm256_fmadd_ps(dcda, l0, g);

                _mm256_store_ps(&gradients->weights[i*IN + j], tmp2);
                _mm256_store_ps(&prev_layer_grads0[j], tmp);
            }
            for (int j = 0; j < IN/2; j += 8) {
                __m256 l0 = _mm256_load_ps(&prev_layer_activations1[j]);
                __m256 w = _mm256_load_ps(&weights->weights[i*IN + j + IN/2]);
                __m256 l0_grads = _mm256_load_ps(&prev_layer_grads1[j]);
                __m256 g = _mm256_load_ps(&gradients->weights[i*IN + j + IN/2]);

                /*__m256 grad = _mm256_mul_ps(dcda, l0);
                __m256 grad_l = _mm256_mul_ps(dcda, w);

                __m256 tmp = _mm256_add_ps(l0_grads, grad_l);
                __m256 tmp2 = _mm256_add_ps(g, grad);*/

                __m256 tmp = _mm256_fmadd_ps(dcda, w, l0_grads);
                __m256 tmp2 = _mm256_fmadd_ps(dcda, l0, g);

                _mm256_store_ps(&gradients->weights[i*IN + j + IN/2], tmp2);
                _mm256_store_ps(&prev_layer_grads1[j], tmp);
            }

            gradients->biases[i] = dc*da;
        }

        for (int i = 0; i < (IN/2); i++) {
            prev_layer_grads0[i] /= OUT;
            prev_layer_grads1[i] /= OUT;
        }
    }

    void back_propagate(training_layer_weights<IN,OUT> *gradients, float *prev_layer_grads, float *prev_layer_activations) {

        for (int i = 0; i < IN; i++) {
            prev_layer_grads[i] = 0;
        }

        for (int i = 0; i < OUT; i++) {
            //float a = neurons[i]; //Activvation
            float x = acculumator[i]; //Neurons input
            float dc = grads[i]; //cost diff
            float da = ((OUT == 1) ? activation_diff_out(x) : activation_diff(x)); //activation diff

            /*for (int j = 0; j < IN; j++) {
                float l0 = prev_layer_activations[j];

                float w = weights->weights[i*IN + j];
                float grad = dc*da*l0;
                float grad_l = dc*da*w;

                gradients->weights[i*IN + j] = grad;
                prev_layer_grads[j] += grad_l;
            }*/

            __m256 dcda = _mm256_set1_ps(dc*da);

            for (int j = 0; j < IN; j += 8) {
                __m256 l0 = _mm256_load_ps(&prev_layer_activations[j]);
                __m256 w = _mm256_load_ps(&weights->weights[i*IN + j]);
                __m256 l0_grads = _mm256_load_ps(&prev_layer_grads[j]);
                __m256 g = _mm256_load_ps(&gradients->weights[i*IN + j]);

                /*__m256 grad = _mm256_mul_ps(dcda, l0);
                __m256 grad_l = _mm256_mul_ps(dcda, w);

                __m256 tmp = _mm256_add_ps(l0_grads, grad_l);
                __m256 tmp2 = _mm256_add_ps(g, grad);*/

                __m256 tmp = _mm256_fmadd_ps(dcda, w, l0_grads);
                __m256 tmp2 = _mm256_fmadd_ps(dcda, l0, g);

                _mm256_store_ps(&prev_layer_grads[j], tmp);
                _mm256_store_ps(&gradients->weights[i*IN + j], tmp2);
            }

            gradients->biases[i] = dc*da;
        }

        for (int i = 0; i < IN; i++) {
            prev_layer_grads[i] /= OUT;
        }
    }

    float *neurons;
    float *acculumator;

    float *neurons_buffer;
    float *acculumator_buffer;

    float *grads;
    float *grads_buffer;


    training_layer_weights<IN, OUT> *weights;
};













