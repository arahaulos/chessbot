#pragma once

#include "history.hpp"

struct move_generator
{
    static chess_move *serialize_moves(chess_move &m, bitboard moves, chess_move *buffer)
    {
        while (moves) {
            uint_fast8_t target_index = bit_scan_forward (moves);
            moves &= moves-1;

            m.to.index = target_index;

            *buffer = m;
            buffer++;
        }
        return buffer;
    }


    static bitboard piece_attack(const board_state &state, piece p, square_index sq)
    {
        uint_fast8_t color = (p.get_player() == BLACK);
        bitboard occupation = state.pieces_by_color[0] | state.pieces_by_color[1];
        bitboard mask = ~state.pieces_by_color[color];

        switch (p.get_type()) {
            case PAWN:
                return bitboard_utils.pawn_attack(sq.index, occupation, color, mask, 0);
            case KNIGHT:
                return bitboard_utils.knight_attack(sq.index, mask);
            case BISHOP:
                return bitboard_utils.bishop_attack(sq.index, occupation, mask);
            case ROOK:
                return bitboard_utils.rook_attack(sq.index, occupation, mask);
            case QUEEN:
                return bitboard_utils.queen_attack(sq.index, occupation, mask);
            case KING:
                return bitboard_utils.king_attack(sq.index, mask);
            default:
                return 0;
        }
        return 0;
    }

    static chess_move *generate_pawn_attack(const board_state &state, bool color, const bitboard &occupation, const bitboard &en_passant, const bitboard &mask, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;

        bitboard iterate_pieces = state.bitboards[PAWN][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            m.from.index = index;

            bitboard iterate_moves = bitboard_utils.pawn_attack(index, occupation, color, mask, en_passant);

            while (iterate_moves) {
                uint_fast8_t target_index = bit_scan_forward (iterate_moves);
                iterate_moves &= iterate_moves-1;

                m.to.index = target_index;

                if (m.to.get_y() == 0 || m.to.get_y() == 7) {
                    for (int i = QUEEN; i >= KNIGHT; i--) {
                        m.promotion = static_cast<piece_type_t>(i);
                        *buffer = m;
                        buffer++;
                    }
                    m.promotion = EMPTY;
                } else {
                    *buffer = m;
                    buffer++;
                }
            }
        }
        return buffer;
    }

    static chess_move *generate_pawn_advance(const board_state &state, bool color, const bitboard &occupation, const bitboard &mask, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;

        bitboard iterate_pieces = state.bitboards[PAWN][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            m.from.index = index;

            bitboard iterate_moves = bitboard_utils.pawn_advance(index, occupation, color) & mask;

            while (iterate_moves) {
                uint_fast8_t target_index = bit_scan_forward (iterate_moves);
                iterate_moves &= iterate_moves-1;

                m.to.index = target_index;

                if (m.to.get_y() == 0 || m.to.get_y() == 7) {
                    for (int i = QUEEN; i >= KNIGHT; i--) {
                        m.promotion = static_cast<piece_type_t>(i);
                        *buffer = m;
                        buffer++;
                    }
                    m.promotion = EMPTY;
                } else {
                    *buffer = m;
                    buffer++;
                }
            }
        }
        return buffer;
    }


    static chess_move *generate_knight(const board_state &state, bool color, const bitboard &occupation, const bitboard &mask, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;

        bitboard iterate_pieces = state.bitboards[KNIGHT][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            bitboard moves = bitboard_utils.knight_attack(index, mask);

            m.from.index = index;

            buffer = serialize_moves(m, moves, buffer);
        }

        return buffer;
    }

    static chess_move *generate_bishop(const board_state &state, bool color, const bitboard &occupation, const bitboard &mask, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;

        bitboard iterate_pieces = state.bitboards[BISHOP][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            bitboard moves = bitboard_utils.bishop_attack(index, occupation, mask);

            m.from.index = index;

            buffer = serialize_moves(m, moves, buffer);
        }

        return buffer;
    }

    static chess_move *generate_rook(const board_state &state, bool color, const bitboard &occupation, const bitboard &mask, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;

        bitboard iterate_pieces = state.bitboards[ROOK][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            bitboard moves = bitboard_utils.rook_attack(index, occupation, mask);

            m.from.index = index;

            buffer = serialize_moves(m, moves, buffer);
        }

        return buffer;
    }

    static chess_move *generate_queen(const board_state &state, bool color, const bitboard &occupation, const bitboard &mask, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;

        bitboard iterate_pieces = state.bitboards[QUEEN][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            bitboard moves = bitboard_utils.queen_attack(index, occupation, mask);

            m.from.index = index;

            buffer = serialize_moves(m, moves, buffer);
        }

        return buffer;
    }

    static chess_move *generate_king(const board_state &state, bool color, const bitboard &occupation, const bitboard &mask, bool castling, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;

        bitboard iterate_pieces = state.bitboards[KING][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            bitboard moves = bitboard_utils.king_attack(index, mask);

            m.from.index = index;

            buffer = serialize_moves(m, moves, buffer);
        }


        if (!castling) {
            return buffer;
        }

        player_type_t player = (color ? BLACK : WHITE);

        bool qside_castling;
        bool kside_castling;
        square_index pos;
        if (!color) {
            qside_castling = ((state.flags & WHITE_QSIDE_CASTLE_VALID) != 0);
            kside_castling = ((state.flags & WHITE_KSIDE_CASTLE_VALID) != 0);
            pos = state.white_king_square;
        } else {
            qside_castling = ((state.flags & BLACK_QSIDE_CASTLE_VALID) != 0);
            kside_castling = ((state.flags & BLACK_KSIDE_CASTLE_VALID) != 0);
            pos = state.black_king_square;
        }

        if (qside_castling || kside_castling) {
            if (state.is_square_threatened(pos, player)) {
                return buffer;
            }
        }

        int_fast8_t posy = pos.get_y();

        m.from = pos;

        if (kside_castling) {
            if (state.get_square(square_index(5, posy)).get_type() == EMPTY &&
                state.get_square(square_index(6, posy)).get_type() == EMPTY &&
                state.get_square(square_index(7, posy)).get_type() == ROOK &&
                state.get_square(square_index(7, posy)).get_player() == player) {
                if (state.is_square_threatened(square_index(5, posy), player) == false && state.is_square_threatened(square_index(6, posy), player) == false) {
                    m.to = square_index(6, posy);
                    *buffer = m;
                    buffer++;
                }
            }
        }
        if (qside_castling) {
            if (state.get_square(square_index(1, posy)).get_type() == EMPTY &&
                state.get_square(square_index(2, posy)).get_type() == EMPTY &&
                state.get_square(square_index(3, posy)).get_type() == EMPTY &&
                state.get_square(square_index(0, posy)).get_type() == ROOK &&
                state.get_square(square_index(0, posy)).get_player() == player) {
                if (state.is_square_threatened(square_index(2, posy), player) == false && state.is_square_threatened(square_index(3, posy), player) == false) {
                    m.to = square_index(2, posy);
                    *buffer = m;
                    buffer++;
                }
            }
        }

        return buffer;
    }

    static int generate_quiet_moves(const board_state &state, player_type_t player, chess_move *movelist) {
        chess_move *buffer_ptr = movelist;

        uint_fast8_t color = (player == BLACK);

        bitboard occupation = ~state.bitboards[EMPTY][0];
        bitboard mask = state.bitboards[EMPTY][0];

        buffer_ptr = generate_pawn_advance(state, color, occupation, ~(uint64_t)0xFF000000000000FF, buffer_ptr);
        buffer_ptr = generate_knight(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_bishop(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_rook(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_queen(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_king(state, color, occupation, mask, true, buffer_ptr);

        return buffer_ptr - movelist;
    }

    static int generate_promotion_moves(const board_state &state, player_type_t player, chess_move *movelist) {
        chess_move *buffer_ptr = movelist;

        uint_fast8_t color = (player == BLACK);

        bitboard occupation = ~state.bitboards[EMPTY][0];

        buffer_ptr = generate_pawn_advance(state, color, occupation, (uint64_t)0xFF000000000000FF, buffer_ptr);

        return buffer_ptr - movelist;
    }

    static int generate_capture_moves(const board_state &state, player_type_t player, chess_move *movelist)
    {
        uint_fast8_t color = (player == BLACK);

        chess_move *buffer_ptr = movelist;

        bitboard en_passant = ((state.flags & EN_PASSANT_AVAILABLE) != 0 ? (((uint64_t)0x1) << state.en_passant_square.index) : 0);

        bitboard own_pieces = state.pieces_by_color[color];
        bitboard enemy_pieces = state.pieces_by_color[!color];

        bitboard mask = (~own_pieces) & enemy_pieces;
        bitboard occupation = own_pieces | enemy_pieces;

        buffer_ptr = generate_pawn_attack(state, color, occupation, en_passant, mask | en_passant, buffer_ptr);
        buffer_ptr = generate_knight(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_bishop(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_rook(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_queen(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_king(state, color, occupation, mask, false, buffer_ptr);

        return buffer_ptr - movelist;
    }

    static int generate_all_pseudo_legal_moves(const board_state &state, player_type_t player, chess_move *movelist)
    {
        uint_fast8_t color = (player == BLACK);


        chess_move *buffer_ptr = movelist;

        bitboard en_passant = ((state.flags & EN_PASSANT_AVAILABLE) != 0 ? (((uint64_t)0x1) << state.en_passant_square.index) : 0);

        bitboard own_pieces = state.pieces_by_color[color];
        bitboard enemy_pieces = state.pieces_by_color[!color];

        bitboard mask = ~own_pieces;
        bitboard occupation = own_pieces | enemy_pieces;

        buffer_ptr = generate_pawn_advance(state, color, occupation, ~(uint64_t)0, buffer_ptr);
        buffer_ptr = generate_pawn_attack(state, color, occupation, en_passant, mask, buffer_ptr);
        buffer_ptr = generate_knight(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_bishop(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_rook(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_queen(state, color, occupation, mask, buffer_ptr);
        buffer_ptr = generate_king(state, color, occupation, mask, true, buffer_ptr);

        return (buffer_ptr - movelist);
    }

    static int generate_killer_moves(const board_state &state, history_heurestic_table &history_table, int ply, chess_move *movelist)
    {
        int c = history_table.get_killers(state, movelist, ply);
        for (int i = 0; i < c; i++) {
            bool dublicate = false;
            for (int j = 0; j < i; j++) {
                if (movelist[i] == movelist[j]) {
                    dublicate = true;
                    break;
                }
            }
            if (dublicate) {
                std::swap(movelist[i], movelist[c-1]);
                c--; i--;
                continue;
            }

            bool is_promotion = (movelist[i].promotion != 0) || (state.get_square(movelist[i].from).get_type() == PAWN && (movelist[i].to.get_y() == 1 || movelist[i].to.get_y() == 7));
            bool is_en_passant = state.get_square(movelist[i].from).get_type() == PAWN && movelist[i].to == state.en_passant_square && (state.flags & EN_PASSANT_AVAILABLE) != 0;
            bool is_capture = state.get_square(movelist[i].to).get_type() != EMPTY;

            movelist[i].promotion = EMPTY;

            if (!is_move_valid(state, movelist[i]) || is_en_passant || is_promotion || is_capture) {
                std::swap(movelist[i], movelist[c-1]);
                c--;
                i--;
            }
        }
        return c;
    }

    static bool is_move_valid(const board_state &state, chess_move mov)
    {
        if (mov.to == mov.from) {
            return false;
        }

        if (state.get_turn() != state.get_square(mov.from).get_player()) {
            return false;
        }

        uint_fast8_t color = (state.get_turn() == BLACK);

        bitboard en_passant = ((state.flags & EN_PASSANT_AVAILABLE) != 0 ? (((uint64_t)0x1) << state.en_passant_square.index) : 0);

        bitboard own_pieces = state.pieces_by_color[color];
        bitboard enemy_pieces = state.pieces_by_color[!color];

        bitboard mask = ~own_pieces;
        bitboard occupation = own_pieces | enemy_pieces;

        bitboard target_square = (uint64_t)0x1 << mov.to.index;

        bitboard bb = 0;

        switch (state.get_square(mov.from).get_type()) {
        case EMPTY:
            break;
        case PAWN:
            bb = bitboard_utils.pawn_advance(mov.from.index, occupation, color) | bitboard_utils.pawn_attack(mov.from.index, occupation, color, mask, en_passant);
            break;
        case KNIGHT:
            bb = bitboard_utils.knight_attack(mov.from.index, mask);
            break;
        case BISHOP:
            bb = bitboard_utils.bishop_attack(mov.from.index, occupation, mask);
            break;
        case ROOK:
            bb = bitboard_utils.rook_attack(mov.from.index, occupation, mask);
            break;
        case QUEEN:
            bb = bitboard_utils.queen_attack(mov.from.index, occupation, mask);
            break;
        case KING:
            bb = bitboard_utils.king_attack(mov.from.index, mask);

            player_type_t player = (color ? BLACK : WHITE);

            bool qside_castling;
            bool kside_castling;
            square_index pos = mov.from;
            if (!color) {
                qside_castling = ((state.flags & WHITE_QSIDE_CASTLE_VALID) != 0);
                kside_castling = ((state.flags & WHITE_KSIDE_CASTLE_VALID) != 0);
            } else {
                qside_castling = ((state.flags & BLACK_QSIDE_CASTLE_VALID) != 0);
                kside_castling = ((state.flags & BLACK_KSIDE_CASTLE_VALID) != 0);
            }

            if (state.is_square_threatened(pos, player)) {
                break;
            }

            int_fast8_t posy = pos.get_y();

            if (kside_castling) {
                if (state.get_square(square_index(5, posy)).get_type() == EMPTY &&
                    state.get_square(square_index(6, posy)).get_type() == EMPTY &&
                    state.get_square(square_index(7, posy)).get_type() == ROOK &&
                    state.get_square(square_index(7, posy)).get_player() == player) {
                    if (state.is_square_threatened(square_index(5, posy), player) == false && state.is_square_threatened(square_index(6, posy), player) == false) {
                        bb |= (uint64_t)0x1 << square_index(6, posy).index;
                    }
                }
            }
            if (qside_castling) {
                if (state.get_square(square_index(1, posy)).get_type() == EMPTY &&
                    state.get_square(square_index(2, posy)).get_type() == EMPTY &&
                    state.get_square(square_index(3, posy)).get_type() == EMPTY &&
                    state.get_square(square_index(0, posy)).get_type() == ROOK &&
                    state.get_square(square_index(0, posy)).get_player() == player) {
                    if (state.is_square_threatened(square_index(2, posy), player) == false && state.is_square_threatened(square_index(3, posy), player) == false) {
                        bb |= (uint64_t)0x1 << square_index(2, posy).index;
                    }
                }
            }
            break;
        }
        return ((bb & target_square) != 0);
    }
};


























