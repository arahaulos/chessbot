#pragma once

#include <memory>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <math.h>

#include "layer.hpp"
#include "perspective.hpp"
#include "training/training_nnue.hpp"
#include "../defs.hpp"

class board_state;
struct piece;
struct square_index;
//enum player_type_t: unsigned char;


struct nnue_weights
{
    nnue_weights();

    nnue_perspective_weights<num_perspective_inputs, num_perspective_neurons> perspective_weights;
    nnue_layer_weights<num_perspective_neurons*2, output_buckets> output_weights;

    int rescale_factor0;
    int rescale_factor1;

    void load(std::string path);
    void load_sb(std::string path);
};

struct nnue_network
{
    nnue_network(std::shared_ptr<nnue_weights> w): weights(w),
                                                   black_side(&w->perspective_weights),
                                                   white_side(&w->perspective_weights),
                                                   output_layer(&w->output_weights) {};


    int16_t evaluate(const board_state &s);

    void refresh(const board_state &s, player_type_t stm);

    int16_t evaluate(player_type_t turn, int output_bucket);

    void set_piece(piece p, square_index sq, square_index white_king_sq, square_index black_king_sq);
    void unset_piece(piece p, square_index sq, square_index white_king_sq, square_index black_king_sq);

    void reset()
    {
        black_side.reset();
        white_side.reset();
    }

    void reset_acculumator_stack()
    {
        white_side.reset_stack();
        black_side.reset_stack();
    }

    void push_acculumator()
    {
        black_side.push_acculumator();
        white_side.push_acculumator();
    }

    void pop_acculumator()
    {
        black_side.pop_acculumator();
        white_side.pop_acculumator();
    }

    void enable_incremental_updates(bool enable, player_type_t type)
    {
        if (type == WHITE) {
            white_side.incremental_updates = enable;
        } else {
            black_side.incremental_updates = enable;
        }
    }

    std::shared_ptr<nnue_weights> weights;
private:
    nnue_perspective<num_perspective_inputs, num_perspective_neurons> black_side;
    nnue_perspective<num_perspective_inputs, num_perspective_neurons> white_side;
    nnue_layer<num_perspective_neurons*2, output_buckets, true> output_layer;
};
