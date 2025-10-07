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
    nnue_layer_weights<num_perspective_neurons*2, layer1_neurons, layer_stack_size> layer1_weights;
    nnue_layer_weights<layer1_neurons, layer2_neurons, layer_stack_size> layer2_weights;
    nnue_layer_weights<layer2_neurons, 1, layer_stack_size> output_weights;


    int rescale_factor0;
    int rescale_factor1;

    void load(std::string path);
    void save(std::string path);
};

struct nnue_board_state
{
    uint64_t bb[8][2];
    int white_king_sq;
    int black_king_sq;
};

struct acculumator_refresh_table_entry
{
    acculumator_refresh_table_entry() {
        acculumator_buffer = new int16_t[num_perspective_neurons+64];
        acculumator = align_ptr(acculumator_buffer);
        valid = false;
    }
    ~acculumator_refresh_table_entry() {
        delete [] acculumator_buffer;
    }

    void save(nnue_perspective<num_perspective_inputs, num_perspective_neurons> *p, nnue_board_state *s)
    {
        p->acculumator_copy(acculumator, p->acculumator);
        state = *s;
        valid = true;
    }

    bool valid;
    int16_t *acculumator;
    nnue_board_state state;

private:
    int16_t *acculumator_buffer;
};

struct nnue_network
{
    nnue_network(std::shared_ptr<nnue_weights> w): weights(w),
                                                   black_side(&w->perspective_weights),
                                                   white_side(&w->perspective_weights),
                                                   layer1(&w->layer1_weights),
                                                   layer2(&w->layer2_weights),
                                                   output_layer(&w->output_weights) { current_state = &state_stack[0]; test_flag = true; };


    int16_t evaluate(const board_state &s);


    void refresh(const board_state &s, player_type_t stm);
    int16_t evaluate(player_type_t stm);

    void set_piece(piece p, square_index sq);
    void unset_piece(piece p, square_index sq);
    void move_piece(piece p, piece captured_p, square_index from_sq, square_index to_sq);


    void reset()
    {
        black_side.reset();
        white_side.reset();
    }

    void reset_acculumator_stack()
    {
        white_side.reset_stack();
        black_side.reset_stack();

        state_stack[0] = *current_state;
        current_state = &state_stack[0];
    }

    void push_acculumator()
    {
        black_side.push_acculumator();
        white_side.push_acculumator();

        current_state[1] = current_state[0];
        current_state += 1;
    }

    void pop_acculumator()
    {
        black_side.pop_acculumator();
        white_side.pop_acculumator();

        current_state -= 1;
    }

    int16_t *get_perspective_activations(player_type_t stm)
    {
        if (stm == WHITE) {
            return white_side.neurons;
        } else {
            return black_side.neurons;
        }
    }

    std::shared_ptr<nnue_weights> weights;
    bool test_flag;
private:
    void full_refresh(const board_state &s, player_type_t stm);


    nnue_perspective<num_perspective_inputs, num_perspective_neurons> black_side;
    nnue_perspective<num_perspective_inputs, num_perspective_neurons> white_side;
    nnue_layer<num_perspective_neurons*2, layer1_neurons, layer_stack_size, false> layer1;
    nnue_layer<layer1_neurons, layer2_neurons, layer_stack_size, false> layer2;
    nnue_layer<layer2_neurons, 1, layer_stack_size, true> output_layer;

    nnue_board_state state_stack[130];
    nnue_board_state *current_state;

    acculumator_refresh_table_entry white_refresh_table[BOARD_SQUARES];
    acculumator_refresh_table_entry black_refresh_table[BOARD_SQUARES];
};
