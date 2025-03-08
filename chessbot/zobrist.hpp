#pragma once

#include <stdint.h>
#include "state.hpp"

class zobrist
{
public:
    zobrist();
    uint64_t get_hash(const board_state &state, player_type_t turn) {
        uint64_t h = (turn == WHITE ? 0 : turn_xor);

        for (int y = 0; y < BOARD_HEIGHT; y++) {
            for (int x = 0; x < BOARD_HEIGHT; x++) {
                h ^= table_xor[y*BOARD_WIDTH + x][state.get_square(square_index(x,y)).d];
            }
        }

        if ((state.flags & EN_PASSANT_AVAILABLE) != 0) {
            h ^= en_passant_xor[state.en_passant_square.get_x()];
        }

        if ((state.flags & WHITE_QSIDE_CASTLE_VALID) != 0) {
            h ^= castling_xor[0];
        }

        if ((state.flags & WHITE_KSIDE_CASTLE_VALID) != 0) {
            h ^= castling_xor[1];
        }

        if ((state.flags & BLACK_QSIDE_CASTLE_VALID) != 0) {
            h ^= castling_xor[2];
        }

        if ((state.flags & BLACK_KSIDE_CASTLE_VALID) != 0) {
            h ^= castling_xor[3];
        }

        return h;
    }

    uint64_t get_singular_search_hash(uint64_t h, chess_move m)
    {
        h ^= singular_search_hashing_from[m.from.index];
        h ^= singular_search_hashing_to[m.to.index];
        h ^= singular_search_promotion[m.promotion];
        return h;
    }

    uint64_t update_hash(uint64_t prev_hash, const board_state &state, chess_move m) {
        uint64_t h = (prev_hash ^ turn_xor);

        if ((state.flags & EN_PASSANT_AVAILABLE) != 0) {
            h ^= en_passant_xor[state.en_passant_square.get_x()];
        }

        h ^= table_xor[m.to.index][state.get_square(m.to).d];
        h ^= table_xor[m.from.index][state.get_square(m.from).d];

        if (state.get_square(m.from).get_type() == PAWN) {
            piece p = state.get_square(m.from);
            if (m.to.get_y() == 0 || m.to.get_y() == BOARD_HEIGHT-1) {
                if (state.get_square(m.from).get_player() == WHITE) {
                    p = piece(WHITE, m.promotion);
                } else {
                    p = piece(BLACK, m.promotion);
                }

                h ^= table_xor[m.to.index][p.d];

                return h;
            } else if (m.to == state.en_passant_square && ((state.flags & EN_PASSANT_AVAILABLE) != 0)) {
                h ^= table_xor[state.en_passant_target_square.index][state.get_square(state.en_passant_target_square).d];
            } else if (m.from.get_y() + 2 == m.to.get_y() || m.from.get_y() - 2 == m.to.get_y()) {
                h ^= en_passant_xor[m.from.get_x()];
            }
        } else if (state.get_square(m.from).get_type() == KING) {
            if (state.get_square(m.from).get_player() == WHITE) {
                if ((state.flags & WHITE_QSIDE_CASTLE_VALID) != 0) {
                    h ^= castling_xor[0];
                }
                if ((state.flags & WHITE_KSIDE_CASTLE_VALID) != 0) {
                    h ^= castling_xor[1];
                }
            } else {
                if ((state.flags & BLACK_QSIDE_CASTLE_VALID) != 0) {
                    h ^= castling_xor[2];
                }
                if ((state.flags & BLACK_KSIDE_CASTLE_VALID) != 0) {
                    h ^= castling_xor[3];
                }
            }

            if (m.to.get_x() == 2 && m.from.get_x() == 4) {

                h ^= table_xor[m.from.get_y() * BOARD_WIDTH + 0][state.get_square(square_index(0, m.from.get_y())).d];
                h ^= table_xor[m.from.get_y() * BOARD_WIDTH + 3][state.get_square(square_index(0, m.from.get_y())).d];
            } else if (m.to.get_x() == 6 && m.from.get_x() == 4) {
                h ^= table_xor[m.from.get_y() * BOARD_WIDTH + 7][state.get_square(square_index(7, m.from.get_y())).d];
                h ^= table_xor[m.from.get_y() * BOARD_WIDTH + 5][state.get_square(square_index(7, m.from.get_y())).d];
            }
        } else if (state.get_square(m.from).get_type() == ROOK) {
            if (state.get_square(m.from).get_player() == WHITE) {
                if (m.from.get_x() == 0) {
                    if ((state.flags & WHITE_QSIDE_CASTLE_VALID) != 0) {
                        h ^= castling_xor[0];
                    }
                } else if (m.from.get_x() == BOARD_WIDTH-1) {
                    if ((state.flags & WHITE_KSIDE_CASTLE_VALID) != 0) {
                        h ^= castling_xor[1];
                    }
                }
            } else {
                if (m.from.get_x() == 0) {
                    if ((state.flags & BLACK_QSIDE_CASTLE_VALID) != 0) {
                        h ^= castling_xor[2];
                    }
                } else if (m.from.get_x() == BOARD_WIDTH-1) {
                    if ((state.flags & BLACK_KSIDE_CASTLE_VALID) != 0) {
                        h ^= castling_xor[3];
                    }
                }
            }
        }

        h ^= table_xor[m.to.index][state.get_square(m.from).d];

        return h;
    }

    uint64_t get_hash(const board_state &state, player_type_t turn, chess_move m)
    {
        uint64_t zhash = get_hash(state, turn);
        return update_hash(zhash, state, m);
    }

    uint64_t next_turn_hash(uint64_t h, const board_state &state) {
        if ((state.flags & EN_PASSANT_AVAILABLE) != 0) {
            h ^= en_passant_xor[state.en_passant_square.get_x()];
        }
        return (h ^ turn_xor);
    }
private:
    uint64_t turn_xor;
    uint64_t castling_xor[4];
    uint64_t table_xor[BOARD_HEIGHT*BOARD_WIDTH][16];
    uint64_t en_passant_xor[BOARD_WIDTH];

    uint64_t rand;
    uint64_t rand_bitstring();

    uint64_t singular_search_hashing_from[64];
    uint64_t singular_search_hashing_to[64];
    uint64_t singular_search_promotion[8];
};

extern zobrist hashgen;
