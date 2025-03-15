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

#include "network_defs.hpp"

class board_state;
struct piece;
struct square_index;
enum player_type_t: unsigned char;



inline int encode_input(int type, int color, int sq_index)
{
    int index = sq_index*12;
    if (color == 0x8) {
        index += 1;
    }
    index += (type-1)*2;
    return index;
}

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
    return index + king_bucket*12*64;
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
    return index + king_bucket*12*64;
}


struct training_weights
{
    training_perspective_weights<num_perspective_inputs + factorizer_inputs, num_perspective_neurons> perspective_weights;
    training_layer_weights<num_perspective_neurons*2, 1> output_weights;


    void copy_from(const training_weights &other) {
        perspective_weights.copy_from(other.perspective_weights);
        output_weights.copy_from(other.output_weights);
    }

    void add(const training_weights &other) {
        perspective_weights.add(other.perspective_weights);
        output_weights.add(other.output_weights);
    }

    void mult(float val) {
        perspective_weights.mult(val);
        output_weights.mult(val);
    }

    void squared(training_weights &other) {
        perspective_weights.squared(other.perspective_weights);
        output_weights.squared(other.output_weights);
    }

    void divide(float val) {
        float inv_val = 1.0f / val;

        perspective_weights.mult(inv_val);
        output_weights.mult(inv_val);
    }

    void zero() {
        perspective_weights.zero();
        output_weights.zero();
    }

    void normalize()
    {
        perspective_weights.normalize();
        output_weights.normalize();
    }

    void save_file(std::string path);
    void load_file(std::string path);


    void save_quantized(std::string path);
};


struct training_network
{
    training_network(std::shared_ptr<training_weights> w): black_side(&w->perspective_weights),
                                                           white_side(&w->perspective_weights),
                                                           output_layer(&w->output_weights), weights(w) { use_factorizer = true;};

    float evaluate(const board_state &s);
    float evaluate(const training_position &tp);

    bool use_factorizer;

    training_perspective<num_perspective_inputs + factorizer_inputs, num_perspective_neurons> black_side;
    training_perspective<num_perspective_inputs + factorizer_inputs, num_perspective_neurons> white_side;
    training_layer<num_perspective_neurons*2, 1> output_layer;

    std::shared_ptr<training_weights> weights;
};

