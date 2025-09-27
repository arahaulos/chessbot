#include "state.hpp"
#include <iostream>
#include "movegen.hpp"
#include "zobrist.hpp"
#include <sstream>
#include <memory>

unmake_restore board_state::make_null_move() {
    unmake_restore restore;
    restore.flags = flags;
    restore.en_passant_square = en_passant_square;
    restore.en_passant_target_square = en_passant_target_square;
    restore.half_move_clock = half_move_clock;
    restore.zhash = zhash;
    restore.structure_hash = structure_hash;

    zhash = hashgen.next_turn_hash(zhash, *this);

    half_move_clock++;

    flags &= ~EN_PASSANT_AVAILABLE;
    flags |= NULL_MOVE;

    push_hash_stack(zhash, true);

    return restore;
}

void board_state::unmake_null_move(const unmake_restore &restore) {
    flags = restore.flags;
    en_passant_square = restore.en_passant_square;
    en_passant_target_square = restore.en_passant_target_square;
    half_move_clock = restore.half_move_clock;
    zhash = restore.zhash;
    structure_hash = restore.structure_hash;

    pop_hash_stack();
}

unmake_restore board_state::make_move(chess_move m) {
    return make_move(m, hashgen.update_hash(zhash, *this, m));
}


unmake_restore board_state::make_move(chess_move m, uint64_t new_zhash) {
    piece p = get_square(m.from);

    bool needs_refresh = false;
    if (nnue) {
        nnue->push_acculumator();
        flags |= INCREMENT_NNUE;

        if (p.get_type() == KING) {
            int prev_bucket = get_king_bucket(p.get_player() == BLACK ? m.from.index ^ 56 : m.from.index);
            int new_bucket = get_king_bucket(p.get_player() == BLACK ? m.to.index ^ 56 : m.to.index);
            bool side_changed = (m.from.index & 0x4) != (m.to.index & 0x4);

            if (prev_bucket != new_bucket || side_changed) {
                needs_refresh = true;
            }
        }
    }

    unmake_restore restore;
    restore.to = get_square(m.to);
    restore.from = get_square(m.from);
    restore.flags = flags;
    restore.en_passant_square = en_passant_square;
    restore.en_passant_target_square = en_passant_target_square;
    restore.half_move_clock = half_move_clock;
    restore.en_passant_used = false;
    restore.zhash = zhash;
    restore.structure_hash = structure_hash;

    zhash = new_zhash;
    structure_hash = hashgen.update_structure_hash(structure_hash, *this, m);

    bool irreversible = (get_square(m.to).get_type() != EMPTY);

    flags &= ~EN_PASSANT_AVAILABLE;
    flags &= ~NULL_MOVE;


    if (p.get_type() == PAWN) {
        irreversible = true;
        if (m.to.get_y() == 0 || m.to.get_y() == BOARD_HEIGHT-1) {
            p = piece(p.get_player(), m.promotion);
        }

        if (m.to.get_x() == m.from.get_x()) {
            if (m.from.get_y() + 2 == m.to.get_y()) {
                en_passant_square = square_index(m.to.get_x(), m.from.get_y() + 1);
                en_passant_target_square = m.to;
                flags |= EN_PASSANT_AVAILABLE;
            } else if (m.from.get_y() - 2 == m.to.get_y()) {
                en_passant_square = square_index(m.to.get_x(), m.from.get_y() - 1);
                en_passant_target_square = m.to;
                flags |= EN_PASSANT_AVAILABLE;
            }
        } else if (m.to == en_passant_square && ((restore.flags & EN_PASSANT_AVAILABLE) != 0)) {
            set_square(en_passant_target_square, piece(WHITE, EMPTY));
            restore.en_passant_used = true;
        }
    } else if (p.get_type() == ROOK) {
        if (p.get_player() == WHITE) {
            if (m.from.get_x() == 0) {
                irreversible |= ((flags & WHITE_QSIDE_CASTLE_VALID) != 0);

                flags &= ~WHITE_QSIDE_CASTLE_VALID;
            } else if (m.from.get_x() == BOARD_WIDTH-1) {
                irreversible |= ((flags & WHITE_KSIDE_CASTLE_VALID) != 0);

                flags &= ~WHITE_KSIDE_CASTLE_VALID;
            }
        } else {
            if (m.from.get_x() == 0) {
                irreversible |= ((flags & BLACK_QSIDE_CASTLE_VALID) != 0);

                flags &= ~BLACK_QSIDE_CASTLE_VALID;
            } else if (m.from.get_x() == BOARD_WIDTH-1) {
                irreversible |= ((flags & BLACK_KSIDE_CASTLE_VALID) != 0);

                flags &= ~BLACK_KSIDE_CASTLE_VALID;
            }
        }
    } else if (p.get_type() == KING) {
        if (p.get_player() == WHITE) {
            irreversible |= ((flags & (WHITE_QSIDE_CASTLE_VALID | WHITE_KSIDE_CASTLE_VALID)) != 0);
            flags &= ~WHITE_QSIDE_CASTLE_VALID;
            flags &= ~WHITE_KSIDE_CASTLE_VALID;

            white_king_square = m.to;

        } else {
            irreversible |= ((flags & (BLACK_QSIDE_CASTLE_VALID | BLACK_KSIDE_CASTLE_VALID)) != 0);
            flags &= ~BLACK_QSIDE_CASTLE_VALID;
            flags &= ~BLACK_KSIDE_CASTLE_VALID;

            black_king_square = m.to;
        }

        if (m.to.get_x() == 2 && m.from.get_x() == 4) {
            set_square(square_index(0, m.from.get_y()), piece(WHITE, EMPTY));
            set_square(square_index(3, m.from.get_y()), piece(p.get_player(), ROOK));
        } else if (m.to.get_x() == 6 && m.from.get_x() == 4) {
            set_square(square_index(7, m.from.get_y()), piece(WHITE, EMPTY));
            set_square(square_index(5, m.from.get_y()), piece(p.get_player(), ROOK));
        }
    }


    if (nnue && m.promotion == EMPTY) {
        flags &= ~INCREMENT_NNUE;
        nnue->move_piece(p, get_square(m.to), m.from, m.to);
    }

    set_square(m.to, p);
    set_square(m.from, piece(WHITE, EMPTY));


    half_move_clock += 1;
    if (irreversible) {
        half_move_clock = half_move_clock & 0x1;
    }

    push_hash_stack(zhash, irreversible);

    if (needs_refresh) {
        nnue->refresh(*this, p.get_player());
    }

    return restore;
}

void board_state::unmake_move(chess_move m, const unmake_restore &restore) {
    flags = restore.flags & (~INCREMENT_NNUE);
    set_square(m.from, restore.from);
    set_square(m.to, restore.to);
    en_passant_square = restore.en_passant_square;
    en_passant_target_square = restore.en_passant_target_square;
    half_move_clock = restore.half_move_clock;
    zhash = restore.zhash;
    structure_hash = restore.structure_hash;

    if (get_square(m.from).get_type() == KING) {
        if (get_square(m.from).get_player() == WHITE) {
            white_king_square = m.from;
        } else {
            black_king_square = m.from;
        }


        if (m.to.get_x() == 2 && m.from.get_x() == 4) {
            set_square(square_index(3, m.from.get_y()), piece(WHITE, EMPTY));
            set_square(square_index(0, m.from.get_y()), piece(get_square(m.from).get_player(), ROOK));
        } else if (m.to.get_x() == 6 && m.from.get_x() == 4) {
            set_square(square_index(5, m.from.get_y()), piece(WHITE, EMPTY));
            set_square(square_index(7, m.from.get_y()), piece(get_square(m.from).get_player(), ROOK));
        }
    }

    if (restore.en_passant_used) {
        if (get_square(m.from).get_player() == WHITE) {
            set_square(en_passant_target_square, piece(BLACK, PAWN));
        } else {
            set_square(en_passant_target_square, piece(WHITE, PAWN));
        }
    }

    pop_hash_stack();

    if (nnue) {
        nnue->pop_acculumator();
    }
}



void board_state::init_clear()
{
    half_move_clock = 0;
    flags = 0;
    hash_stack_top = 0;

    white_king_square = square_index(0);
    black_king_square = square_index(0);

    for (int i = 0; i < BOARD_SQUARES; i++) {
        board[i] = piece(WHITE, EMPTY);
    }

    recalculate_bitboards();

    zhash = hashgen.get_hash(*this, get_turn());
    structure_hash = hashgen.get_structure_hash(*this, get_turn());
}

void board_state::set_initial_state()
{
    load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

std::vector<square_index> board_state::get_legal_moves(square_index pos) const {
    chess_move buffer[256];
    int move_count = move_generator::generate_all_pseudo_legal_moves(*this, get_square(pos).get_player(), buffer);

    std::vector<square_index> moves;
    for (int i = 0; i < move_count; i++) {
        if (buffer[i].from == pos) {
            if (!causes_check(buffer[i], get_turn())) {
                moves.push_back(buffer[i].to);
            }
        }
    }
    return moves;
}

std::vector<square_index> board_state::get_capture_moves(square_index pos) const {
    chess_move buffer[256];
    int move_count = move_generator::generate_capture_moves(*this, get_square(pos).get_player(), buffer);

    std::vector<square_index> moves;
    for (int i = 0; i < move_count; i++) {
        if (buffer[i].from == pos) {
            if (!causes_check(buffer[i], get_turn())) {
                moves.push_back(buffer[i].to);
            }
        }
    }
    return moves;
}

std::vector<chess_move> board_state::get_all_legal_moves(player_type_t player) const
{
    chess_move buffer[256];
    int move_count = move_generator::generate_all_pseudo_legal_moves(*this, player, buffer);

    std::vector<chess_move> moves;
    for (int i = 0; i < move_count; i++) {
        if (!causes_check(buffer[i], get_turn())) {
            moves.push_back(buffer[i]);
        }
    }
    return moves;
}

int board_state::count_legal_moves(player_type_t player) const
{
    int legal_moves = 0;

    chess_move buffer[256];
    int move_count = move_generator::generate_all_pseudo_legal_moves(*this, player, buffer);

    for (int i = 0; i < move_count; i++) {
        if (!causes_check(buffer[i], get_turn())) {
            legal_moves += 1;
        }
    }
    return legal_moves;
}


void board_state::load_fen(std::string fen_str)
{
    init_clear();

    enum parse_state_t {PARSE_POSITION = 0, PARSE_TURN = 1, PARSE_CASTLING = 2, PARSE_EN_PASSANT = 3, PARSE_HALFMOVE = 4, PARSE_FULLMOVE = 5};

    int parse_state = PARSE_POSITION;

    int x = 0;
    int y = 0;

    flags = 0;

    for (size_t i = 0; i < fen_str.size(); i++) {
        char c = fen_str[i];
        if (c == ' ') {
            parse_state += 1;
        }

        if (parse_state == PARSE_POSITION) {
            static char piece_chars[13] = "PNBRQKpnbrqk";
            static piece piece_pieces[12] = {piece(WHITE, PAWN), piece(WHITE, KNIGHT), piece(WHITE, BISHOP), piece(WHITE, ROOK), piece(WHITE, QUEEN), piece(WHITE, KING),
                                             piece(BLACK, PAWN), piece(BLACK, KNIGHT), piece(BLACK, BISHOP), piece(BLACK, ROOK), piece(BLACK, QUEEN), piece(BLACK, KING)};

            if (c >= '1' && c <= '9') {
                x += c - '0';
            } else if (c == '/') {
                y += 1;
                x = 0;
            } else {
                if (x < 0 || x > 7 || y < 0 || y > 7) {
                    continue;
                }

                for (int j = 0; j < 12; j++) {
                    if (c == piece_chars[j]) {

                        board[square_index(x, y).index] = piece_pieces[j];
                        x += 1;

                        break;
                    }
                }
            }
        } else if (parse_state == PARSE_TURN) {
            if (c == 'b') {
                half_move_clock = 1;
            } else if (c == 'w') {
                half_move_clock = 0;
            }
        } else if (parse_state == PARSE_CASTLING) {
            if (c == 'k') {
                flags |= BLACK_KSIDE_CASTLE_VALID;
            } else if (c == 'q') {
                flags |= BLACK_QSIDE_CASTLE_VALID;
            } else if (c == 'K') {
                flags |= WHITE_KSIDE_CASTLE_VALID;
            } else if (c == 'Q') {
                flags |= WHITE_QSIDE_CASTLE_VALID;
            }
        } else if (parse_state == PARSE_EN_PASSANT) {
            if (c != '-' && c != ' ') {
                en_passant_square.from_uci(fen_str.substr(i, 2));

                if (en_passant_square.get_y() == 2) {
                    en_passant_target_square.set_xy(en_passant_square.get_x(), 3);
                } else if (en_passant_square.get_y() == 5) {
                    en_passant_target_square.set_xy(en_passant_square.get_x(), 4);
                }
                flags |= EN_PASSANT_AVAILABLE;

                i += 1;
            }
        } else if (parse_state == PARSE_HALFMOVE) {

        } else if (parse_state == PARSE_FULLMOVE) {

        }


    }


    for (int y = 0; y < BOARD_HEIGHT; y++) {
        for (int x = 0; x < BOARD_WIDTH; x++) {
            square_index i(x, y);
            piece p = get_square(i);
            if (p.get_player() == WHITE) {
                if (p.get_type() == KING) {
                    white_king_square = i;
                }
            } else {
                if (p.get_type() == KING) {
                    black_king_square = i;
                }
            }
        }
    }

    recalculate_bitboards();

    zhash = hashgen.get_hash(*this, get_turn());
    structure_hash = hashgen.get_structure_hash(*this, get_turn());
}


std::string board_state::generate_fen() const
{
    static char piece_chars[13] = "PNBRQKpnbrqk";
    static piece piece_pieces[12] = {piece(WHITE, PAWN), piece(WHITE, KNIGHT), piece(WHITE, BISHOP), piece(WHITE, ROOK), piece(WHITE, QUEEN), piece(WHITE, KING),
                                     piece(BLACK, PAWN), piece(BLACK, KNIGHT), piece(BLACK, BISHOP), piece(BLACK, ROOK), piece(BLACK, QUEEN), piece(BLACK, KING)};

    std::stringstream ss;

    for (int y = 0; y < BOARD_HEIGHT; y++) {
        int e = 0;
        for (int x = 0; x < BOARD_WIDTH; x++) {
            square_index sq(x, y);
            if (get_square(sq).get_type() == EMPTY) {
                e += 1;
            } else {
                if (e > 0) {
                    ss << e;
                }
                for (int i = 0; i < 12; i++) {
                    if (piece_pieces[i] == get_square(sq)) {
                        ss << piece_chars[i];
                        break;
                    }
                }
                e = 0;
            }
        }
        if (e > 0) {
            ss << e;
        }
        if (y != 7) {
            ss << "/";
        }
    }
    if (get_turn() == WHITE) {
        ss << " w ";
    } else {
        ss << " b ";
    }

    int any_castling = WHITE_QSIDE_CASTLE_VALID | WHITE_KSIDE_CASTLE_VALID | BLACK_QSIDE_CASTLE_VALID | BLACK_KSIDE_CASTLE_VALID;
    if ((flags & any_castling) == 0) {
        ss << "- ";
    } else {
        bool castling_available = false;
        if ((flags & WHITE_KSIDE_CASTLE_VALID) != 0 && get_square(63).d == WHITE_ROOK) {
            ss << "K";
            castling_available = true;
        }
        if ((flags & WHITE_QSIDE_CASTLE_VALID) != 0 && get_square(56).d == WHITE_ROOK) {
            ss << "Q";
            castling_available = true;
        }
        if ((flags & BLACK_KSIDE_CASTLE_VALID) != 0 && get_square(7).d == BLACK_ROOK) {
            ss << "k";
            castling_available = true;
        }
        if ((flags & BLACK_QSIDE_CASTLE_VALID) != 0 && get_square(0).d == BLACK_ROOK) {
            ss << "q";
            castling_available = true;
        }
        if (castling_available) {
            ss << " ";
        } else {
            ss << "- ";
        }
    }
    if ((flags & EN_PASSANT_AVAILABLE) != 0) {
        ss << en_passant_square.to_uci();
        ss << " ";
    } else {
        ss << "-";
    }

    return ss.str();
}




void board_state::recalculate_bitboards()
{
    for (int i  = 0; i < 8; i++) {
        bitboards[i][0] = 0;
        bitboards[i][1] = 0;
    }

    pieces_by_color[0] = 0;
    pieces_by_color[1] = 0;

    for (int i = 0; i < 64; i++) {
        piece p = board[i];
        bitboard bb = ((uint64_t)0x1) << i;
        if (p.get_type() == EMPTY) {
            bitboards[EMPTY][WHITE] |= bb;
            continue;
        }
        bool color = (p.get_player() == BLACK);

        bitboards[p.get_type()][color] |= bb;

        pieces_by_color[color] |= bb;
    }


    material_conf = 0;
    material_conf |= pop_count(bitboards[PAWN][0]) << 0;
    material_conf |= pop_count(bitboards[PAWN][1]) << 4;

    material_conf |= pop_count(bitboards[KNIGHT][0]) << 8;
    material_conf |= pop_count(bitboards[KNIGHT][1]) << 11;

    material_conf |= pop_count(bitboards[BISHOP][0]) << 14;
    material_conf |= pop_count(bitboards[BISHOP][1]) << 17;

    material_conf |= pop_count(bitboards[ROOK][0]) << 20;
    material_conf |= pop_count(bitboards[ROOK][1]) << 23;

    material_conf |= pop_count(bitboards[QUEEN][0]) << 26;
    material_conf |= pop_count(bitboards[QUEEN][1]) << 29;
}


bool board_state::validate_bitboards() const
{
    bitboard occ = 0;
    for (int i = 0; i < 8; i++) {
        if (i != EMPTY) {
            occ |= bitboards[i][0];
            occ |= bitboards[i][1];
        }
    }

    if (~occ != bitboards[EMPTY][WHITE]) {
        return false;
    }


    for (int i = 0; i < 64; i++) {
        piece p = board[i];

        bitboard bb = ((uint64_t)0x1) << i;

        if (p.get_type() == EMPTY) {
            if ((bitboards[EMPTY][WHITE] & bb) == 0) {
                return false;
            }
            continue;
        }

        bool color = (p.get_player() == BLACK);

        for (int j = 0; j < 8; j++) {
            if (j == p.get_type()) {
                if ((bitboards[j][color] & bb) == 0) {
                    return false;
                }
                if ((bitboards[j][!color] & bb) != 0) {
                    return false;
                }
            } else {
                if (((bitboards[j][0] & bb) != 0) ||
                    ((bitboards[j][1] & bb) != 0)) {
                    return false;
                }
            }
        }
    }

    return true;
}


std::string player_to_str(player_type_t t)
{
    if (t == WHITE) {
        return std::string("White");
    } else {
        return std::string("Black");
    }
}


std::string piece_to_str(piece_type_t t)
{
    if (t == PAWN) {
        return std::string("Pawn");
    } else if (t == KNIGHT) {
        return std::string("Knight");
    } else if (t == BISHOP) {
        return std::string("Bishop");
    } else if (t == ROOK) {
        return std::string("Rook");
    } else if (t == QUEEN) {
        return std::string("Queen");
    } else if (t == KING) {
        return std::string("King");
    } else {
        return std::string("Empty");
    }
}






