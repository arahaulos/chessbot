
#include "training_position.hpp"
#include "../../state.hpp"
#include "../../util/misc.hpp"

void training_position::get_best_move(chess_move &best_move) {
    best_move.from = (bm >> 10) & 0x3F;
    best_move.to = (bm >> 4) & 0x3F;
    best_move.promotion = (piece_type_t)(bm & 0xF);
    best_move.encoded_pieces = bm_pieces;
}



void training_position::encode_best_move(const board_state &state, const chess_move &best_move)
{
    bm = (best_move.from.index << 10) | (best_move.to.index << 4) | best_move.promotion;
    bm_pieces = move_generator::encode_move_pieces(state, best_move);
}


void training_position::init(const board_state &state, float wdl, int32_t score)
{
    flags = 0;
    eval = score;
    bm_pieces = 0;
    bm = 0;

    if ((wdl == 1.0f && state.get_turn() == WHITE) ||
        (wdl == 0.0f && state.get_turn() == BLACK)) {
        flags |= RESULT_STM_WIN;
    } else if (wdl == 0.5f) {
        flags |= RESULT_DRAW;
    }
    if (state.get_turn() == WHITE) {
        flags |= WHITE_TURN;
    }

    occupation = state.pieces_by_color[0] | state.pieces_by_color[1];

    int write_index = 0;

    for (int i = 0; i < 64; i++) {
        piece p = state.get_square(i);
        if (p.get_type() == EMPTY) {
            continue;
        }

        uint8_t bp = p.d & 0xF;

        int byte_index = write_index / 2;
        if ((write_index & 0x1) == 0) {
            packed_pieces[byte_index] = bp;
        } else {
            packed_pieces[byte_index] |= (bp << 4);
        }
        write_index++;
    }
}

training_position::training_position(const selfplay_result &spr) {
    board_state state;
    state.load_fen(spr.fen);

    init(state, spr.wdl, spr.eval);

    encode_best_move(state, spr.bm);
}

training_position::training_position(const board_state &state, const selfplay_result &spr)
{
    init(state, spr.wdl, spr.eval);

    encode_best_move(state, spr.bm);
}
