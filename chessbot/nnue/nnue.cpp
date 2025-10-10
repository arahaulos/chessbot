#include "nnue.hpp"
#include "../state.hpp"
#include <algorithm>
#include "huffman.hpp"


void nnue_network::full_refresh(const board_state &s, player_type_t stm)
{
    current_state->white_king_sq = s.white_king_square.index;
    current_state->black_king_sq = s.black_king_square.index;

    if (stm == WHITE) {
        white_side.reset();
        white_side.update_table->refresh = true;

        int white_king_sq = s.white_king_square.index;

        uint64_t occupation = s.pieces_by_color[0] | s.pieces_by_color[1];

        while (occupation) {
            int i = bit_scan_forward_clear(occupation);

            piece p = s.get_square(i);

            int type = p.get_type();
            int color = p.get_player();
            int sq_index = i;

            white_side.update_table->add(encode_input_with_buckets(type, color, sq_index, white_king_sq));
        }
    } else {
        black_side.reset();
        black_side.update_table->refresh = true;

        int black_king_sq = s.black_king_square.index ^ 56;

        uint64_t occupation = s.pieces_by_color[0] | s.pieces_by_color[1];

        while (occupation) {
            int i = bit_scan_forward_clear(occupation);

            piece p = s.get_square(i);

            int type = p.get_type();
            int color = p.get_player();

            int flipped_sq_index = i^56;
            int flipped_color = (color == BLACK ? WHITE : BLACK);

            black_side.update_table->add(encode_input_with_buckets(type, flipped_color, flipped_sq_index, black_king_sq));
        }
    }
}


void nnue_network::refresh(const board_state &s, player_type_t stm)
{
    current_state->white_king_sq = s.white_king_square.index;
    current_state->black_king_sq = s.black_king_square.index;

    if (stm == WHITE) {
        int white_king_sq = s.white_king_square.index;

        acculumator_refresh_table_entry *entry = &white_refresh_table[s.white_king_square.index];
        if (!entry->valid) {
            full_refresh(s, stm);

            return;
        }
        white_side.update_table->clear(white_side.acculumator, true);
        white_side.update_table->refresh = true;
        white_side.acculumator_copy(white_side.acculumator, entry->acculumator);
        for (int i = PAWN; i <= KING; i++) {
            for (int j = 0; j < 2; j++) {
                uint64_t add =   current_state->bb[i][j]  & (~entry->state.bb[i][j]);
                uint64_t sub = (~current_state->bb[i][j]) &   entry->state.bb[i][j];

                while (add) {
                    int sq_index = bit_scan_forward_clear(add);
                    white_side.update_table->add(encode_input_with_buckets(i, ((j != 0) ? BLACK : WHITE), sq_index, white_king_sq));
                }
                while (sub) {
                    int sq_index = bit_scan_forward_clear(sub);
                    white_side.update_table->sub(encode_input_with_buckets(i, ((j != 0) ? BLACK : WHITE), sq_index, white_king_sq));
                }
            }
        }
    } else {
        int black_king_sq = s.black_king_square.index ^ 56;

        acculumator_refresh_table_entry *entry = &black_refresh_table[s.black_king_square.index];
        if (!entry->valid) {
            full_refresh(s, stm);

            return;
        }
        black_side.update_table->clear(black_side.acculumator, true);
        black_side.update_table->refresh = true;
        black_side.acculumator_copy(black_side.acculumator, entry->acculumator);
        for (int i = PAWN; i <= KING; i++) {
            for (int j = 0; j < 2; j++) {
                uint64_t add =   current_state->bb[i][j]  & (~entry->state.bb[i][j]);
                uint64_t sub = (~current_state->bb[i][j]) &   entry->state.bb[i][j];

                while (add) {
                    int sq_index = bit_scan_forward_clear(add);
                    black_side.update_table->add(encode_input_with_buckets(i, ((j != 0) ? WHITE : BLACK), sq_index ^ 56, black_king_sq));
                }
                while (sub) {
                    int sq_index = bit_scan_forward_clear(sub);
                    black_side.update_table->sub(encode_input_with_buckets(i, ((j != 0) ? WHITE : BLACK), sq_index ^ 56, black_king_sq));
                }
            }
        }
    }
}


void nnue_network::set_piece(piece p, square_index sq)
{
    int type = p.get_type();
    int color = p.get_player();

    int sq_index = sq.get_index();
    int flipped_sq_index = sq_index^56;
    int flipped_color = (color == BLACK ? WHITE : BLACK);

    current_state->bb[type][color == BLACK] |= (0x1ULL << sq_index);

    int wksq = current_state->white_king_sq;
    int bksq = current_state->black_king_sq^56;

    white_side.update_table->add(encode_input_with_buckets(type, color, sq_index, wksq));
    black_side.update_table->add(encode_input_with_buckets(type, flipped_color, flipped_sq_index, bksq));
}

void nnue_network::unset_piece(piece p, square_index sq)
{
    int type = p.get_type();
    int color = p.get_player();

    int sq_index = sq.get_index();
    int flipped_sq_index = sq_index^56;
    int flipped_color = (color == BLACK ? WHITE : BLACK);

    current_state->bb[type][color == BLACK] &= ~(0x1ULL << sq_index);

    int wksq = current_state->white_king_sq;
    int bksq = current_state->black_king_sq^56;

    white_side.update_table->sub(encode_input_with_buckets(type, color, sq_index, wksq));
    black_side.update_table->sub(encode_input_with_buckets(type, flipped_color, flipped_sq_index, bksq));
}

void nnue_network::move_piece(piece p, piece old_p, square_index from_sq, square_index to_sq)
{
    int type = p.get_type();
    int color = p.get_player();

    int from_sq_index = from_sq.get_index();
    int to_sq_index = to_sq.get_index();

    int from_flipped_sq_index = from_sq_index^56;
    int to_flipped_sq_index = to_sq_index^56;

    int flipped_color = (color == BLACK ? WHITE : BLACK);

    current_state->bb[type][color == BLACK] &= ~(0x1ULL << from_sq_index);
    current_state->bb[type][color == BLACK] |=  (0x1ULL << to_sq_index);

    int wksq = current_state->white_king_sq;
    int bksq = current_state->black_king_sq^56;

    if (old_p.get_type() == EMPTY) {
        white_side.update_table->addsub(encode_input_with_buckets(type, color, to_sq_index,   wksq),
                                        encode_input_with_buckets(type, color, from_sq_index, wksq));

        black_side.update_table->addsub(encode_input_with_buckets(type, flipped_color, to_flipped_sq_index,   bksq),
                                        encode_input_with_buckets(type, flipped_color, from_flipped_sq_index, bksq));
    } else {
        int old_type = old_p.get_type();
        int old_color = old_p.get_player();

        int old_flipped_color = (old_p.get_player() == BLACK ? WHITE : BLACK);

        current_state->bb[old_type][old_color == BLACK] &= ~(0x1ULL << to_sq_index);

        white_side.update_table->addsubsub(encode_input_with_buckets(type,     color,     to_sq_index,   wksq),
                                           encode_input_with_buckets(type,     color,     from_sq_index, wksq),
                                           encode_input_with_buckets(old_type, old_color, to_sq_index,   wksq));

        black_side.update_table->addsubsub(encode_input_with_buckets(type,     flipped_color,     to_flipped_sq_index,   bksq),
                                           encode_input_with_buckets(type,     flipped_color,     from_flipped_sq_index, bksq),
                                           encode_input_with_buckets(old_type, old_flipped_color, to_flipped_sq_index,   bksq));

    }
}


int16_t nnue_network::evaluate(player_type_t stm)
{
    uint64_t non_pawn_pieces = current_state->bb[BISHOP][0] | current_state->bb[BISHOP][1] |
                               current_state->bb[KNIGHT][0] | current_state->bb[KNIGHT][1] |
                               current_state->bb[ROOK][0]   | current_state->bb[ROOK][1]   |
                               current_state->bb[QUEEN][0]  | current_state->bb[QUEEN][1];

    int output_bucket = encode_output_bucket(non_pawn_pieces);

    if (white_side.apply_all_updates()) {
        white_refresh_table[current_state->white_king_sq].save(&white_side, current_state);
    }
    if (black_side.apply_all_updates()) {
        black_refresh_table[current_state->black_king_sq].save(&black_side, current_state);
    }

    white_side.update();
    black_side.update();
    if (stm == WHITE) {
        layer1.update(output_bucket, white_side.neurons, white_side.outputs_idx, white_side.num_of_outputs, black_side.neurons, black_side.outputs_idx, black_side.num_of_outputs);
    } else {
        layer1.update(output_bucket, black_side.neurons, black_side.outputs_idx, black_side.num_of_outputs, white_side.neurons, white_side.outputs_idx, white_side.num_of_outputs);
    }
    layer2.update(output_bucket, layer1.neurons);
    output_layer.update(output_bucket, layer2.neurons);

    return (output_layer.out * 100 * weights->rescale_factor0) / (output_quantization_fractions * weights->rescale_factor1);
}

int16_t nnue_network::evaluate(const board_state &s)
{
    for (int i = PAWN; i <= KING; i++) {
        for (int j = 0; j < 2; j++) {
            current_state->bb[i][j] = s.bitboards[i][j];
        }
    }
    current_state->white_king_sq = s.white_king_square.index;
    current_state->black_king_sq = s.black_king_square.index;

    refresh(s, WHITE);
    refresh(s, BLACK);

    return evaluate(s.get_turn());
}


extern unsigned char _binary_embedded_weights_nnue_start[];
extern unsigned char _binary_embedded_weights_nnue_end[];

size_t embedded_weights_size = _binary_embedded_weights_nnue_end - _binary_embedded_weights_nnue_start;
unsigned char* embedded_weights_data = _binary_embedded_weights_nnue_start;


nnue_weights::nnue_weights()
{
    //std::cout << "Decompressing embedded NNUE... ";

    uint8_t *decoded_data;
    int decoded_size;
    huffman_coder::decode(embedded_weights_data, embedded_weights_size, decoded_data, decoded_size);

    //std::cout << "done.\nLoading embedded NNUE... ";

    rescale_factor0 = 1;
    rescale_factor1 = 1;

    size_t index = 0;
    perspective_weights.load((int16_t*)decoded_data, index);
    layer1_weights.load((int16_t*)decoded_data, index);
    layer2_weights.load((int16_t*)decoded_data, index);
    output_weights.load((int16_t*)decoded_data, index);

    delete [] decoded_data;

    //std::cout << "done." << std::endl;
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
        layer1_weights.load((int16_t*)decoded_data, index);
        layer2_weights.load((int16_t*)decoded_data, index);
        output_weights.load((int16_t*)decoded_data, index);

        delete [] encoded_data;
        delete [] decoded_data;
    }
    file.close();
}

void nnue_weights::save(std::string path)
{
    std::ofstream file(path.c_str(), std::ios::binary);
    if (file.is_open()) {

        size_t total_params = perspective_weights.num_of_biases() + perspective_weights.num_of_weights() +
                              layer1_weights.num_of_biases() + layer1_weights.num_of_weights() +
                              layer2_weights.num_of_biases() + layer2_weights.num_of_weights() +
                              output_weights.num_of_biases() + output_weights.num_of_weights();


        uint8_t *raw_data = new uint8_t[sizeof(int16_t)*total_params];
        size_t index = 0;
        perspective_weights.save((int16_t*)raw_data, index);
        layer1_weights.save((int16_t*)raw_data, index);
        layer2_weights.save((int16_t*)raw_data, index);
        output_weights.save((int16_t*)raw_data, index);

        uint8_t *encoded_data;
        int encoded_size;
        huffman_coder::encode(raw_data, total_params*sizeof(int16_t), encoded_data, encoded_size);

        file.write((char*)encoded_data, encoded_size);

        delete [] encoded_data;
        delete [] raw_data;
    }
    file.close();
}










