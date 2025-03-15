#include "training_nnue.hpp"
#include "../../state.hpp"


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
    if (s.get_turn() == WHITE) {
        output_layer.update(white_side.neurons, black_side.neurons);
    } else {
        output_layer.update(black_side.neurons, white_side.neurons);
    }

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

    for (int i = 0; i < num_of_pieces; i++) {
        p.d = tp.iterate_pieces(occupation, index, sq_index);

        int type = p.get_type();
        int color = p.get_player();

        int flipped_sq_index = sq_index^56;
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
    if (tp.get_turn() == WHITE) {
        output_layer.update(white_side.neurons, black_side.neurons);
    } else {
        output_layer.update(black_side.neurons, white_side.neurons);
    }
    return output_layer.neurons[0];
}


void training_weights::save_file(std::string path)
{
    std::ofstream file(path.c_str(), std::ios::binary);
    if (file.is_open()) {
        perspective_weights.save(file);
        output_weights.save(file);
    }
    file.close();
}

void training_weights::load_file(std::string path)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        size_t s = file.tellg();
        file.seekg(0, std::ios::beg);

        bool big_net = (s > 1024*1024*8);

        perspective_weights.load(file, big_net);
        output_weights.load(file);
    }
    file.close();
}


void training_weights::save_quantized(std::string path)
{
    std::ofstream file(path.c_str(), std::ios::binary);
    if (file.is_open()) {
        perspective_weights.save_quantized(file);
        output_weights.save_quantized(file);
    }
    file.close();
}










