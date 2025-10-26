#pragma once

#include "history.hpp"

struct move_generator
{
    static chess_move *serialize_moves(chess_move &m, const board_state &state, bitboard moves, chess_move *buffer)
    {
        while (moves) {
            uint_fast8_t target_index = bit_scan_forward (moves);
            moves &= moves-1;

            *buffer = m;
            buffer->to.index = target_index;
            buffer->encoded_pieces |= chess_move::encode_captured_piece(state.get_square(target_index));
            buffer++;
        }
        return buffer;
    }

    static uint8_t encode_move_pieces(const board_state &state, chess_move mov)
    {
        piece moving_piece = state.get_square(mov.from);
        piece captured_piece = state.get_square(mov.to);

        if (moving_piece.get_type() == PAWN && mov.to == state.en_passant_square && (state.flags & EN_PASSANT_AVAILABLE) != 0) {
            captured_piece = piece(next_turn(moving_piece.get_player()), PAWN);
        }

        return chess_move::encode_captured_piece(captured_piece) | chess_move::encode_moving_piece(moving_piece);
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


    static bitboard piece_quiet_moves(const board_state &state, piece p, square_index sq)
    {
        uint_fast8_t color = (p.get_player() == BLACK);
        bitboard occupation = state.pieces_by_color[0] | state.pieces_by_color[1];
        bitboard mask = ~occupation;

        switch (p.get_type()) {
            case PAWN:
                return bitboard_utils.pawn_advance(sq.index, occupation, color);
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
        m.encoded_pieces = chess_move::encode_moving_piece(piece((color ? BLACK : WHITE), PAWN));

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
                        *buffer = m;
                        buffer->encoded_pieces |= chess_move::encode_captured_piece(state.get_square(target_index));
                        buffer->promotion = static_cast<piece_type_t>(i);
                        buffer++;
                    }
                } else {
                    *buffer = m;
                    buffer->encoded_pieces |= chess_move::encode_captured_piece(state.get_square(target_index));

                    if (target_index == state.en_passant_square.get_index() && (state.flags & EN_PASSANT_AVAILABLE) != 0) {
                        buffer->encoded_pieces |= chess_move::encode_captured_piece(piece((color ? WHITE : BLACK), PAWN));
                    }

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
        m.encoded_pieces = chess_move::encode_moving_piece(piece((color ? BLACK : WHITE), PAWN));

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
        m.encoded_pieces = chess_move::encode_moving_piece(piece((color ? BLACK : WHITE), KNIGHT));

        bitboard iterate_pieces = state.bitboards[KNIGHT][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            bitboard moves = bitboard_utils.knight_attack(index, mask);

            m.from.index = index;

            buffer = serialize_moves(m, state, moves, buffer);
        }

        return buffer;
    }

    static chess_move *generate_bishop(const board_state &state, bool color, const bitboard &occupation, const bitboard &mask, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;
        m.encoded_pieces = chess_move::encode_moving_piece(piece((color ? BLACK : WHITE), BISHOP));

        bitboard iterate_pieces = state.bitboards[BISHOP][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            bitboard moves = bitboard_utils.bishop_attack(index, occupation, mask);

            m.from.index = index;

            buffer = serialize_moves(m, state, moves, buffer);
        }

        return buffer;
    }

    static chess_move *generate_rook(const board_state &state, bool color, const bitboard &occupation, const bitboard &mask, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;
        m.encoded_pieces = chess_move::encode_moving_piece(piece((color ? BLACK : WHITE), ROOK));

        bitboard iterate_pieces = state.bitboards[ROOK][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            bitboard moves = bitboard_utils.rook_attack(index, occupation, mask);

            m.from.index = index;

            buffer = serialize_moves(m, state, moves, buffer);
        }

        return buffer;
    }

    static chess_move *generate_queen(const board_state &state, bool color, const bitboard &occupation, const bitboard &mask, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;
        m.encoded_pieces = chess_move::encode_moving_piece(piece((color ? BLACK : WHITE), QUEEN));

        bitboard iterate_pieces = state.bitboards[QUEEN][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            bitboard moves = bitboard_utils.queen_attack(index, occupation, mask);

            m.from.index = index;

            buffer = serialize_moves(m, state, moves, buffer);
        }

        return buffer;
    }

    static chess_move *generate_king(const board_state &state, bool color, const bitboard &occupation, const bitboard &mask, bool castling, chess_move *buffer)
    {
        chess_move m;
        m.promotion = EMPTY;
        m.encoded_pieces = chess_move::encode_moving_piece(piece((color ? BLACK : WHITE), KING));

        bitboard iterate_pieces = state.bitboards[KING][color];
        while (iterate_pieces) {
            uint_fast8_t index = bit_scan_forward (iterate_pieces);
            iterate_pieces &= iterate_pieces-1;

            bitboard moves = bitboard_utils.king_attack(index, mask);

            m.from.index = index;

            buffer = serialize_moves(m, state, moves, buffer);
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

        buffer_ptr = generate_pawn_advance(state, color, occupation, ~0xFF000000000000FFull, buffer_ptr);
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

        buffer_ptr = generate_pawn_advance(state, color, occupation, 0xFF000000000000FFull, buffer_ptr);

        return buffer_ptr - movelist;
    }

    static int generate_quiet_checks(const board_state &state, player_type_t player, chess_move *movelist)
    {
        //Doesn't consider castling checks!!!


        uint_fast8_t color = (player == BLACK);

        chess_move *buffer_ptr = movelist;

        bitboard own_pieces = state.pieces_by_color[color];
        bitboard enemy_pieces = state.pieces_by_color[!color];

        bitboard occupation = own_pieces | enemy_pieces;

        uint_fast8_t opponent_king = (player == WHITE ? state.black_king_square.index : state.white_king_square.index);

        bitboard bishop_target = bitboard_utils.bishop_attack(opponent_king, occupation, ~(uint64_t)0);
        bitboard rook_target = bitboard_utils.rook_attack(opponent_king, occupation, ~(uint64_t)0);
        bitboard knight_target = bitboard_utils.knight_attack(opponent_king, ~occupation);
        bitboard pawn_target = bitboard_utils.pawn_attack(opponent_king, ~(uint64_t)0, !color, ~occupation, 0);

        buffer_ptr = generate_pawn_advance(state, color, occupation, pawn_target, buffer_ptr);
        buffer_ptr = generate_knight(state, color, occupation, knight_target, buffer_ptr);
        buffer_ptr = generate_bishop(state, color, occupation, bishop_target & (~occupation), buffer_ptr);
        buffer_ptr = generate_rook(state, color, occupation, rook_target & (~occupation), buffer_ptr);
        buffer_ptr = generate_queen(state, color, occupation, (bishop_target | rook_target) & (~occupation), buffer_ptr);

        bitboard potential_blockers = 0;
        if ((bishop_target & own_pieces) != 0) {
            bitboard bishop_xray = bitboard_utils.bishop_xray_attack(opponent_king, occupation, ~(uint64_t)0);
            if ((bishop_xray & state.bitboards[BISHOP][color]) != 0 ||
                (bishop_xray & state.bitboards[QUEEN][color]) != 0) {

                potential_blockers |= (bishop_target & own_pieces);
            }
        }

        if ((rook_target & own_pieces) != 0) {
            bitboard rook_xray = bitboard_utils.rook_xray_attack(opponent_king, occupation, ~(uint64_t)0);
            if ((rook_xray & state.bitboards[ROOK][color]) != 0 ||
                (rook_xray & state.bitboards[QUEEN][color]) != 0) {

                potential_blockers |= (rook_target & own_pieces);
            }
        }

        while (potential_blockers) {
            uint_fast8_t index = bit_scan_forward (potential_blockers);
            potential_blockers &= potential_blockers-1;

            piece p = state.get_square(index);

            chess_move mov;
            mov.from = index;
            mov.promotion = EMPTY;
            mov.encoded_pieces = chess_move::encode_moving_piece(p);

            bitboard moves = piece_quiet_moves(state, p, index);

            if (p.get_type() == PAWN) {
                moves &= ~0xFF000000000000FFull;
            }

            while (moves) {
                uint_fast8_t target_index = bit_scan_forward (moves);
                moves &= moves-1;

                mov.to = target_index;

                if (state.causes_check(mov, next_turn(player))) {
                    int num_of_moves = buffer_ptr - movelist;
                    bool found = false;
                    for (int i = 0; i < num_of_moves; i++) {
                        if (movelist[i] == mov) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        *buffer_ptr = mov;
                        buffer_ptr++;
                    }
                }
            }
        }

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
        int c = history_table.get_killers(movelist, ply);

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

            bool is_promotion = (movelist[i].promotion != 0) || (state.get_square(movelist[i].from).get_type() == PAWN && (movelist[i].to.get_y() == 0 || movelist[i].to.get_y() == 7));
            bool is_en_passant = state.get_square(movelist[i].from).get_type() == PAWN && movelist[i].to == state.en_passant_square && (state.flags & EN_PASSANT_AVAILABLE) != 0;
            bool is_capture = state.get_square(movelist[i].to).get_type() != EMPTY;

            movelist[i].promotion = EMPTY;
            movelist[i].encoded_pieces = chess_move::encode_moving_piece(state.get_square(movelist[i].from));

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


























