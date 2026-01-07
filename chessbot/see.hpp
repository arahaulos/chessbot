#pragma once


inline int32_t static_exchange_evaluation(board_state &state, chess_move m)
{
    //SEE does not work for en passant. Return 0
    static const int32_t see_piece_values[7] = {0, 100, 320, 330, 500, 900, 5000};

    uint_fast8_t to_sq = m.to.index;
    bitboard to_sq_bb = ((uint64_t)1) << m.to.index;
    bitboard from_sq_bb = ((uint64_t)1) << m.from.index;

    bitboard bitboards[8][2];
    bitboard occupation;

    for (int i = PAWN; i <= KING; i++) {
        for (int j = 0; j < 2; j++) {
            bitboards[i][j] = state.bitboards[i][j];
        }
    }
    occupation = state.pieces_by_color[0] | state.pieces_by_color[1];

    bool turn = (m.get_moving_piece().get_player() == BLACK);

    uint_fast8_t attacker_type = state.get_square(m.from).get_type();
    uint_fast8_t victim_type = state.get_square(m.to).get_type();

    int32_t see_eval[32];
    int32_t victim_values[32];
    int num_of_captures = 0;

    bitboard mask = ~((uint64_t)0);

    do {
        occupation &= ~from_sq_bb;
        occupation |= to_sq_bb;

        bitboards[attacker_type][turn] &= ~from_sq_bb;
        bitboards[attacker_type][turn] |= to_sq_bb;
        bitboards[victim_type][!turn] &= ~to_sq_bb;

        victim_values[num_of_captures++] = see_piece_values[victim_type];

        turn = !turn;
        victim_type = attacker_type;

        bitboard attacks[7] = {0, bitboard_utils.pawn_attack(to_sq, occupation, !turn, mask, 0),
                                  bitboard_utils.knight_attack(to_sq, mask),
                                  bitboard_utils.bishop_attack(to_sq, occupation, mask),
                                  bitboard_utils.rook_attack(to_sq, occupation, mask),
                                  bitboard_utils.queen_attack(to_sq, occupation, mask),
                                  bitboard_utils.king_attack(to_sq, mask)};

        attacker_type = EMPTY;
        for (int i = PAWN; i <= KING; i++) {
            bitboard bb = bitboards[i][turn] & attacks[i];
            if (bb != 0) {
                attacker_type = i;
                from_sq_bb = (bb & (-bb));
                break;
            }
        }
    } while (attacker_type != EMPTY);

    see_eval[num_of_captures] = 0;
    for (int i = num_of_captures-1; i >= 1; i--) {
        see_eval[i] = std::max(0, victim_values[i] - see_eval[i+1]);
    }
    see_eval[0] = victim_values[0] - see_eval[1];

    return see_eval[0];
}









