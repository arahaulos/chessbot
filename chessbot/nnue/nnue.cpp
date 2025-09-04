#include "nnue.hpp"
#include "../state.hpp"
#include <algorithm>
#include "huffman.hpp"


void nnue_network::refresh(const board_state &s, player_type_t stm)
{
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

    bitboard non_pawn_pieces = (s.pieces_by_color[0] | s.pieces_by_color[1]) & ~(s.bitboards[PAWN][0] | s.bitboards[PAWN][1] | s.bitboards[KING][0] | s.bitboards[KING][1]);

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

        white_side.set_input(encode_input_with_buckets(type, color, sq_index, white_king_sq));
        black_side.set_input(encode_input_with_buckets(type, flipped_color, flipped_sq_index, black_king_sq));
    }

    int output_bucket = encode_output_bucket(non_pawn_pieces);

    white_side.update();
    black_side.update();
    if (s.get_turn() == WHITE) {
        output_layer.update(output_bucket, white_side.neurons, black_side.neurons);
    } else {
        output_layer.update(output_bucket, black_side.neurons, white_side.neurons);
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

    white_side.set_input(encode_input_with_buckets(type, color, sq_index, wksq));
    black_side.set_input(encode_input_with_buckets(type, flipped_color, flipped_sq_index, bksq));
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

    white_side.unset_input(encode_input_with_buckets(type, color, sq_index, wksq));
    black_side.unset_input(encode_input_with_buckets(type, flipped_color, flipped_sq_index, bksq));
}


int16_t nnue_network::evaluate(player_type_t turn, int output_bucket)
{
    white_side.update();
    black_side.update();
    if (turn == WHITE) {
        output_layer.update(output_bucket, white_side.neurons, black_side.neurons);
    } else {
        output_layer.update(output_bucket, black_side.neurons, white_side.neurons);
    }

    return (output_layer.out * 100 * weights->rescale_factor0) / (output_quantization_fractions * weights->rescale_factor1);
}


extern unsigned char _binary_embedded_weights_nnue_start[];
extern unsigned char _binary_embedded_weights_nnue_end[];

size_t embedded_weights_size = _binary_embedded_weights_nnue_end - _binary_embedded_weights_nnue_start;
unsigned char* embedded_weights_data = _binary_embedded_weights_nnue_start;


nnue_weights::nnue_weights()
{
    uint8_t *decoded_data;
    int decoded_size;
    huffman_coder::decode(embedded_weights_data, embedded_weights_size, decoded_data, decoded_size);

    rescale_factor0 = 1;
    rescale_factor1 = 1;

    size_t index = 0;
    perspective_weights.load((int16_t*)decoded_data, index);
    output_weights.load((int16_t*)decoded_data, index);

    delete [] decoded_data;
}


void nnue_weights::load(std::string path)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        size_t fsize = file.tellg();
        file.seekg(0, std::ios::beg);


        uint8_t *encoded_data = new uint8_t[fsize];

        file.read((char*)encoded_data, fsize);

        uint8_t *decoded_data;
        int decoded_size;
        huffman_coder::decode(encoded_data, fsize, decoded_data, decoded_size);

        size_t index = 0;
        perspective_weights.load((int16_t*)decoded_data, index);
        output_weights.load((int16_t*)decoded_data, index);

        delete [] encoded_data;
        delete [] decoded_data;
    }
    file.close();
}










