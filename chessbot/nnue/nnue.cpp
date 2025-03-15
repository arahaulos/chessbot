#include "nnue.hpp"
#include "../state.hpp"
#include <algorithm>


void nnue_network::refresh(const board_state &s, player_type_t stm)
{
    //evaluate(s);
    //return;

    if (!weights->big_net) {
        return;
    }

    if (stm == WHITE) {
        white_side.reset();
        white_side.incremental_updates = true;

        int white_king_sq = s.white_king_square.index;

        uint64_t occupation = s.pieces_by_color[0] | s.pieces_by_color[1];

        while (occupation) {
            int i = bit_scan_forward_clear(occupation);

            piece p = s.get_square(i);

            int type = p.get_type();
            int color = p.get_player();
            int sq_index = i;

            white_side.set_input(encode_input_with_buckets(type, color, sq_index, white_king_sq));
        }
    } else {
        black_side.reset();
        black_side.incremental_updates = true;

        int black_king_sq = s.black_king_square.index ^ 56;

        uint64_t occupation = s.pieces_by_color[0] | s.pieces_by_color[1];

        while (occupation) {
            int i = bit_scan_forward_clear(occupation);

            piece p = s.get_square(i);

            int type = p.get_type();
            int color = p.get_player();

            int flipped_sq_index = i^56;
            int flipped_color = (color == BLACK ? WHITE : BLACK);

            black_side.set_input(encode_input_with_buckets(type, flipped_color, flipped_sq_index, black_king_sq));
        }
    }
}


int16_t nnue_network::evaluate(const board_state &s)
{
    black_side.reset();
    white_side.reset();

    black_side.incremental_updates = true;
    white_side.incremental_updates = true;

    int white_king_sq = s.white_king_square.index;
    int black_king_sq = s.black_king_square.index ^ 56;

    for (int i = 0; i < 64; i++) {
        piece p = s.get_square(i);

        if (p.get_type() == EMPTY) {
            continue;
        }

        int type = p.get_type();
        int color = p.get_player();

        int sq_index = i;
        int flipped_sq_index = i^56;
        int flipped_color = (color == BLACK ? WHITE : BLACK);

        if (weights->big_net) {
            white_side.set_input(encode_input_with_buckets(type, color, sq_index, white_king_sq));
            black_side.set_input(encode_input_with_buckets(type, flipped_color, flipped_sq_index, black_king_sq));
        } else {
            white_side.set_input(encode_input(type, color, sq_index));
            black_side.set_input(encode_input(type, flipped_color, flipped_sq_index));
        }
    }

    white_side.update();
    black_side.update();
    if (s.get_turn() == WHITE) {
        output_layer.update(white_side.neurons, black_side.neurons);
    } else {
        output_layer.update(black_side.neurons, white_side.neurons);
    }

    return (output_layer.out * 100 * weights->rescale_factor0) / (output_quantization_fractions * weights->rescale_factor1);
}

void nnue_network::set_piece(piece p, square_index sq, square_index white_king_sq, square_index black_king_sq)
{
    int type = p.get_type();
    int color = p.get_player();

    int sq_index = sq.get_index();
    int flipped_sq_index = sq_index^56;
    int flipped_color = (color == BLACK ? WHITE : BLACK);


    int wksq = white_king_sq.index;
    int bksq = black_king_sq.index^56;


    if (weights->big_net) {
        white_side.set_input(encode_input_with_buckets(type, color, sq_index, wksq));
        black_side.set_input(encode_input_with_buckets(type, flipped_color, flipped_sq_index, bksq));
    } else {
        white_side.set_input(encode_input(type, color, sq_index));
        black_side.set_input(encode_input(type, flipped_color, flipped_sq_index));
    }
}

void nnue_network::unset_piece(piece p, square_index sq, square_index white_king_sq, square_index black_king_sq)
{
    int type = p.get_type();
    int color = p.get_player();

    int sq_index = sq.get_index();
    int flipped_sq_index = sq_index^56;
    int flipped_color = (color == BLACK ? WHITE : BLACK);

    int wksq = white_king_sq.index;
    int bksq = black_king_sq.index^56;

    if (weights->big_net) {
        white_side.unset_input(encode_input_with_buckets(type, color, sq_index, wksq));
        black_side.unset_input(encode_input_with_buckets(type, flipped_color, flipped_sq_index, bksq));
    } else {
        white_side.unset_input(encode_input(type, color, sq_index));
        black_side.unset_input(encode_input(type, flipped_color, flipped_sq_index));
    }
}


int16_t nnue_network::evaluate(player_type_t turn)
{
    white_side.update();
    black_side.update();
    if (turn == WHITE) {
        output_layer.update(white_side.neurons, black_side.neurons);
    } else {
        output_layer.update(black_side.neurons, white_side.neurons);
    }

    return (output_layer.out * 100 * weights->rescale_factor0) / (output_quantization_fractions * weights->rescale_factor1);
}


extern unsigned char _binary_embedded_weights_nnue_start[];
extern unsigned char _binary_embedded_weights_nnue_end[];

size_t embedded_weights_size = _binary_embedded_weights_nnue_end - _binary_embedded_weights_nnue_start;
unsigned char* embedded_weights_data = _binary_embedded_weights_nnue_start;


nnue_weights::nnue_weights()
{
    rescale_factor0 = 1;
    rescale_factor1 = 1;


    big_net = (embedded_weights_size > 1024*1024*4);

    size_t index = 0;
    perspective_weights.load((int16_t*)embedded_weights_data, index, big_net);
    output_weights.load((int16_t*)embedded_weights_data, index);
}


void nnue_weights::load(std::string path)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    if (file.is_open()) {

        file.seekg(0, std::ios::end);
        size_t s = file.tellg();
        file.seekg(0, std::ios::beg);

        big_net = (s > 1024*1024*4);

        perspective_weights.load(file, big_net);
        output_weights.load(file);
    }
    file.close();
}










