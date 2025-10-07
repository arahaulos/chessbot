#pragma once

#include <memory>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <math.h>
#include "training_layer.hpp"
#include "training_perspective.hpp"
#include "training_position.hpp"

#include "../nnue_defs.hpp"

class board_state;
struct piece;
struct square_index;
enum player_type_t: unsigned char;

struct training_weights
{
    training_perspective_weights<num_perspective_inputs + factorizer_inputs, num_perspective_neurons> perspective_weights;
    training_layer_weights<num_perspective_neurons*2, layer1_neurons, layer_stack_size, false> layer1_weights;
    training_layer_weights<layer1_neurons, layer2_neurons, layer_stack_size, false> layer2_weights;
    training_layer_weights<layer2_neurons, 1, layer_stack_size, true> output_weights;


    void copy_from(const training_weights &other, int tid, int tc) {
        perspective_weights.copy_from(other.perspective_weights, tid, tc);

        layer1_weights.copy_from(other.layer1_weights, tid, tc);
        layer2_weights.copy_from(other.layer2_weights, tid, tc);

        output_weights.copy_from(other.output_weights, tid, tc);
    }

    void add(const training_weights &other, int tid, int tc) {
        perspective_weights.add(other.perspective_weights, tid, tc);

        layer1_weights.add(other.layer1_weights, tid, tc);
        layer2_weights.add(other.layer2_weights, tid, tc);

        output_weights.add(other.output_weights, tid, tc);
    }

    void mult(float val, int tid, int tc) {
        perspective_weights.mult(val, tid, tc);

        layer1_weights.mult(val, tid, tc);
        layer2_weights.mult(val, tid, tc);

        output_weights.mult(val, tid, tc);
    }

    void mult(training_weights &other, float val, int tid, int tc) {
        perspective_weights.mult(other.perspective_weights, val, tid, tc);

        layer1_weights.mult(other.layer1_weights, val, tid, tc);
        layer2_weights.mult(other.layer2_weights, val, tid, tc);

        output_weights.mult(other.output_weights, val, tid, tc);
    }

    void squared(training_weights &other, int tid, int tc) {
        perspective_weights.squared(other.perspective_weights, tid, tc);

        layer1_weights.squared(other.layer1_weights, tid, tc);
        layer2_weights.squared(other.layer2_weights, tid, tc);

        output_weights.squared(other.output_weights, tid, tc);
    }

    void exponential_smoothing(training_weights &other, float factor, int tid, int tc) {
        perspective_weights.exponential_smoothing(other.perspective_weights, factor, tid, tc);

        layer1_weights.exponential_smoothing(other.layer1_weights, factor, tid, tc);
        layer2_weights.exponential_smoothing(other.layer2_weights, factor, tid, tc);

        output_weights.exponential_smoothing(other.output_weights, factor, tid, tc);
    }

    void divide(float val, int tid, int tc) {
        float inv_val = 1.0f / val;

        perspective_weights.mult(inv_val, tid, tc);

        layer1_weights.mult(inv_val, tid, tc);
        layer2_weights.mult(inv_val, tid, tc);

        output_weights.mult(inv_val, tid, tc);
    }

    void zero(int tid, int tc) {
        perspective_weights.zero(tid, tc);

        layer1_weights.zero(tid, tc);
        layer2_weights.zero(tid, tc);

        output_weights.zero(tid, tc);
    }


    void copy_from(const training_weights &other) {
        copy_from(other, 0, 1);
    }

    void add(const training_weights &other) {
        add(other, 0, 1);
    }

    void mult(float val) {
        mult(val, 0, 1);
    }

    void squared(training_weights &other) {
        squared(other, 0, 1);
    }

    void divide(float val) {
        divide(val, 0, 1);
    }

    void zero() {
        zero(0, 1);
    }



    void save_file(std::string path);
    void load_file(std::string path);


    void save_quantized(std::string path);
};


struct training_network
{
    training_network(std::shared_ptr<training_weights> w): black_side(&w->perspective_weights),
                                                           white_side(&w->perspective_weights),
                                                           layer1(&w->layer1_weights),
                                                           layer2(&w->layer2_weights),
                                                           output_layer(&w->output_weights), weights(w) { use_factorizer = true;};

    float evaluate(const board_state &s);
    float evaluate(const training_position &tp);

    bool use_factorizer;

    training_perspective<num_perspective_inputs + factorizer_inputs, num_perspective_neurons> black_side;
    training_perspective<num_perspective_inputs + factorizer_inputs, num_perspective_neurons> white_side;

    training_layer<num_perspective_neurons*2, layer1_neurons, layer_stack_size, false> layer1;
    training_layer<layer1_neurons, layer2_neurons, layer_stack_size, false> layer2;
    training_layer<layer2_neurons, 1, layer_stack_size, true> output_layer;

    std::shared_ptr<training_weights> weights;

    int output_bucket;
};

