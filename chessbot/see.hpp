#pragma once


inline int32_t static_exchange_evaluation(board_state &state, chess_move m)
{
    //SEE does not work for en passant. Return 0
    bool non_capture = (state.get_square(m.to).get_type() == EMPTY);

    static const int32_t see_piece_values[7] = {0, 100, 320, 330, 500, 900, 5000};

    uint_fast8_t to_sq = m.to.index;
    bitboard to_sq_bb = ((uint64_t)1) << m.to.index;
    bitboard from_sq_bb = ((uint64_t)1) << m.from.index;

    bitboard bitboards[8][2];
    bitboard pieces_by_color[2];

    for (int i = PAWN; i <= KING; i++) {
        for (int j = 0; j < 2; j++) {
            bitboards[i][j] = state.bitboards[i][j];
        }
    }

    pieces_by_color[0] = state.pieces_by_color[0];
    pieces_by_color[1] = state.pieces_by_color[1];

    bool turn = (state.get_square(m.from.index).get_player() == BLACK);

    uint_fast8_t attacker_type = state.get_square(m.from).get_type();
    uint_fast8_t victim_type = state.get_square(m.to).get_type();

    int32_t see_eval[64];
    int32_t victim_values[64];
    int num_of_captures = 0;


    pieces_by_color[turn] &= ~from_sq_bb;
    pieces_by_color[turn] |= to_sq_bb;

    bitboards[attacker_type][turn] &= ~from_sq_bb;
    bitboards[attacker_type][turn] |= to_sq_bb;

    if (!non_capture) {
        bitboards[victim_type][!turn] &= ~to_sq_bb;
        pieces_by_color[!turn] &= ~to_sq_bb;
    }

    while (true) {
        victim_values[num_of_captures] = see_piece_values[victim_type];
        num_of_captures++;

        turn = !turn;
        victim_type = attacker_type;

        bitboard occupation = pieces_by_color[0] | pieces_by_color[1];
        bitboard mask = ~((uint64_t)0);

        bitboard attacks[7] = {0, bitboard_utils.pawn_attack(to_sq, occupation, !turn, mask, 0),
                                  bitboard_utils.knight_attack(to_sq, mask),
                                  bitboard_utils.bishop_attack(to_sq, occupation, mask),
                                  bitboard_utils.rook_attack(to_sq, occupation, mask),
                                  bitboard_utils.queen_attack(to_sq, occupation, mask),
                                  bitboard_utils.king_attack(to_sq, mask)};

        bool found = false;

        for (int i = PAWN; i <= KING; i++) {
            bitboard bb = bitboards[i][turn] & attacks[i];
            if (bb != 0) {
                attacker_type = i;
                from_sq_bb = (bb & (-bb));

                found = true;
                break;
            }
        }
        if (!found) {
            break;
        }

        pieces_by_color[turn] &= ~from_sq_bb;
        pieces_by_color[turn] |= to_sq_bb;

        bitboards[attacker_type][turn] &= ~from_sq_bb;
        bitboards[attacker_type][turn] |= to_sq_bb;

        bitboards[victim_type][!turn] &= ~to_sq_bb;
        pieces_by_color[!turn] &= ~to_sq_bb;

    }
    see_eval[num_of_captures] = 0;
    for (int i = num_of_captures-1; i >= 1; i--) {
        see_eval[i] = std::max(0, victim_values[i] - see_eval[i+1]);
    }
    see_eval[0] = victim_values[0] - see_eval[1];


    return see_eval[0];
}




















