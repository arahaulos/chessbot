#include "training_nnue.hpp"
#include "../../state.hpp"
#include "../huffman.hpp"


float training_network::evaluate(const board_state &s)
{
    white_side.use_factorizer = use_factorizer;
    black_side.use_factorizer = use_factorizer;

    black_side.reset();
    white_side.reset();

    int white_king_sq = s.white_king_square.index;
    int black_king_sq = s.black_king_square.index ^ 56;

    white_side.king_bucket = get_king_bucket(white_king_sq);
    black_side.king_bucket = get_king_bucket(black_king_sq);

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

        if (use_factorizer) {
            white_side.set_input(encode_factorizer_input(type, color, sq_index, white_king_sq));
            black_side.set_input(encode_factorizer_input(type, flipped_color, flipped_sq_index, black_king_sq));
        }
    }

    white_side.update();
    black_side.update();

    output_bucket = encode_output_bucket(non_pawn_pieces);

    if (s.get_turn() == WHITE) {
        layer1.update(output_bucket, white_side.neurons, black_side.neurons);
    } else {
        layer1.update(output_bucket, black_side.neurons, white_side.neurons);
    }
    layer2.update(output_bucket, layer1.neurons);
    output_layer.update(output_bucket, layer2.neurons);

    return output_layer.neurons[0];
}

float training_network::evaluate(const training_position &tp)
{
    white_side.use_factorizer = use_factorizer;
    black_side.use_factorizer = use_factorizer;

    black_side.reset();
    white_side.reset();

    int num_of_pieces = tp.count_pieces();
    uint64_t occupation = tp.occupation;
    int index = 0;
    piece p;
    int sq_index;


    int white_king_sq = 0;
    int black_king_sq = 0;
    for (int i = 0; i < num_of_pieces; i++) {
        p.d = tp.iterate_pieces(occupation, index, sq_index);

        if (p.get_player() == WHITE && p.get_type() == KING) {
            white_king_sq = sq_index;
        } else if (p.get_player() == BLACK && p.get_type() == KING) {
            black_king_sq = sq_index;
        }
    }

    black_king_sq ^= 56;

    white_side.king_bucket = get_king_bucket(white_king_sq);
    black_side.king_bucket = get_king_bucket(black_king_sq);


    num_of_pieces = tp.count_pieces();
    occupation = tp.occupation;
    index = 0;

    bitboard non_pawn_pieces = 0;

    for (int i = 0; i < num_of_pieces; i++) {
        p.d = tp.iterate_pieces(occupation, index, sq_index);

        if (p.get_type() != PAWN && p.get_type() != KING) {
            non_pawn_pieces |= ((uint64_t)0x1 << sq_index);
        }

        int type = p.get_type();
        int color = p.get_player();

        int flipped_sq_index = sq_index^56;
        int flipped_color = (color == BLACK ? WHITE : BLACK);

        int white_input = encode_input_with_buckets(type, color, sq_index, white_king_sq);
        int black_input = encode_input_with_buckets(type, flipped_color, flipped_sq_index, black_king_sq);

        white_side.prefetch_input(white_input);
        black_side.prefetch_input(black_input);

        if (use_factorizer) {
            white_side.set_input(encode_factorizer_input(type, color, sq_index, white_king_sq));
            black_side.set_input(encode_factorizer_input(type, flipped_color, flipped_sq_index, black_king_sq));
        }

        white_side.set_input(white_input);
        black_side.set_input(black_input);
    }

    output_bucket = encode_output_bucket(non_pawn_pieces);

    white_side.update();
    black_side.update();
    if (tp.get_turn() == WHITE) {
        layer1.update(output_bucket, white_side.neurons, black_side.neurons);
    } else {
        layer1.update(output_bucket, black_side.neurons, white_side.neurons);
    }

    layer2.update(output_bucket, layer1.neurons);
    output_layer.update(output_bucket, layer2.neurons);

    return output_layer.neurons[0];
}

void training_weights::save_file(std::string path)
{
    std::ofstream file(path.c_str(), std::ios::binary);
    if (file.is_open()) {
        perspective_weights.save(file);
        layer1_weights.save(file);
        layer2_weights.save(file);
        output_weights.save(file);
    }
    file.close();
}

void training_weights::load_file(std::string path)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    if (file.is_open()) {
        perspective_weights.load(file);
        layer1_weights.load(file);
        layer2_weights.load(file);
        output_weights.load(file);
    }
    file.close();
}


void training_weights::save_quantized(std::string path)
{
    int16_t *weights = new int16_t[perspective_weights.num_of_quantized_params() +
                                   layer1_weights.num_of_quantized_params() +
                                   layer2_weights.num_of_quantized_params() +
                                   output_weights.num_of_quantized_params()];
    size_t index = 0;

    perspective_weights.save_quantized(weights, index);
    layer1_weights.save_quantized(weights, index);
    layer2_weights.save_quantized(weights, index);
    output_weights.save_quantized(weights, index);

    uint8_t *encoded_data;
    int encoded_size;

    huffman_coder::encode((uint8_t*)weights, index*2, encoded_data, encoded_size);

    std::ofstream file(path.c_str(), std::ios::binary);
    if (file.is_open()) {
        file.write((char*)encoded_data, encoded_size);
    }
    file.close();

    delete [] weights;
    delete [] encoded_data;
}










