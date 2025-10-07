
#include "training_position.hpp"
#include "../../state.hpp"
#include "../../util/misc.hpp"


void training_position::init(const board_state &state, float wdl, int32_t score)
{
    flags = 0;
    eval = score;

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
}

training_position::training_position(const board_state &state, const selfplay_result &spr)
{
    init(state, spr.wdl, spr.eval);
}
