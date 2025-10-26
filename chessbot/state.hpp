#pragma once

#include <cstring>
#include <stdint.h>
#include <vector>
#include <iostream>
#include "bitboard.hpp"
#include "defs.hpp"
#include "nnue/nnue.hpp"


inline player_type_t next_turn(player_type_t player)
{
    if (player == WHITE) {
        return BLACK;
    } else {
        return WHITE;
    }
}

std::string player_to_str(player_type_t t);
std::string piece_to_str(piece_type_t t);


struct square_index
{
    square_index() {};

    square_index(uint_fast8_t pos_x, uint_fast8_t pos_y) {
        index = pos_y * BOARD_WIDTH + pos_x;
    }
    square_index(uint_fast8_t i) {
        index = i;
    }

    uint8_t index;

    inline int get_index() const {
        return index;
    }

    inline void set_xy(int x, int y) {
        index = y * BOARD_WIDTH + x;
    }

    inline void set_x(int x) {
        set_xy(get_y(), x);
    }

    inline void set_y(int y) {
        set_xy(y, get_x());
    }

    inline uint_fast8_t get_x() const {
        return index % BOARD_WIDTH;
    }

    inline uint_fast8_t get_y() const {
        return index / BOARD_WIDTH;
    }

    bool is_dark() const {
        return ((((int)index * 9) & 0x8) == 0);
    }

    bool valid() const {
        return (index < 64);
    }

    inline bool operator == (const square_index &other) const {
        return (index == other.index);
    }

    inline bool operator != (const square_index &other) const {
        return (index != other.index);
    }


    std::string to_uci() const {
        static const char *ranks = "abcdefgh";
        static const char *files = "87654321";

        char cstr[3] = {ranks[get_x()], files[get_y()], 0};
        return std::string(cstr);
    }

    void from_uci(std::string str) {
        static const char *ranks = "abcdefgh";
        static const char *files = "87654321";
        char r = str[0];
        char f = str[1];

        int x, y;

        for (int i = 0; i < 8; i++) {
            if (ranks[i] == r) {
                x = i;
            }
            if (files[i] == f) {
                y = i;
            }
        }
        set_xy(x, y);

    }
};

struct piece
{
    piece() {};

    piece(player_type_t player, piece_type_t type) {
        d = player | type;
    }
    piece(piece_player_type_t type) {
        d = type;
    }

    inline player_type_t get_player() const {
        return static_cast<player_type_t>(d & 0x8);
    }

    inline piece_type_t get_type() const {
        return static_cast<piece_type_t>(d & 0x7);
    }

    bool operator == (const piece &p) const {
        return (d == p.d);
    }

    bool operator != (const piece &p) const {
        return (d != p.d);
    }

    std::string to_string() {
        return player_to_str(get_player()) + " " + piece_to_str(get_type());
    }

    uint8_t d;
};

struct chess_move
{
    square_index from;
    square_index to;

    piece_type_t promotion;
    uint8_t encoded_pieces;

    static chess_move null_move() {
        chess_move nm;
        nm.promotion = EMPTY;
        nm.to.index = 0;
        nm.from.index = 0;
        nm.encoded_pieces = 0;
        return nm;
    }

    static uint8_t encode_moving_piece(piece p) {
        return ((uint8_t)p.d << 4);
    }

    static uint8_t encode_captured_piece(piece p) {
        return ((uint8_t)p.d & 0xF);
    }

    piece get_captured_piece() const {
        return piece((piece_player_type_t)(encoded_pieces & 0xF));
    }

    piece get_moving_piece() const {
        return piece((piece_player_type_t)((encoded_pieces >> 4) & 0xF));
    }

    bool is_capture() const {
        return ((encoded_pieces & 0xF) != 0);
    }

    bool is_promotion() const {
        return (promotion != EMPTY);
    }

    bool is_castling() const {
        return (get_moving_piece().get_type() == KING && std::abs(from.get_x() - to.get_x()) > 1);
    }

    inline bool operator == (const chess_move &other) const {
        return (from == other.from && to == other.to && promotion == other.promotion);
    }

    inline bool operator != (const chess_move &other) const {
        return !(*this == other);
    }

    bool valid() const {
        return (from.valid() && to.valid() && to.index != from.index);
    }

    std::string to_uci() const {
        if (from == to) {
            return std::string("0000");
        }

        static const char *promotions = "ppnbrq";

        std::string str = from.to_uci() + to.to_uci();
        if (promotion >= KNIGHT && promotion <= QUEEN) {
            str += promotions[promotion];
        }
        return str;
    }

    void from_uci(std::string uci_str) {
        static const char *promotions = "ppnbrq";

        promotion = EMPTY;

        from.from_uci(uci_str.substr(0, 2));
        to.from_uci(uci_str.substr(2, 2));

        if (uci_str.length() > 4) {
            char p = uci_str[4];
            for (int i = KNIGHT; i <= QUEEN; i += 1) {
                if (p == promotions[i]) {
                    promotion = static_cast<piece_type_t>(i);
                }
            }
        }
    }
};


struct unmake_restore
{
    piece from;
    piece to;
    uint16_t flags;
    uint16_t half_move_clock;
    square_index en_passant_square;
    square_index en_passant_target_square;
    uint64_t zhash;
    uint64_t structure_hash;
    bool en_passant_used;
};

class board_state
{
public:
    std::shared_ptr<nnue_network> nnue;

    void set_initial_state();
    void init_clear();

    inline piece get_square(const square_index &pos) const {
        return board[pos.index];
    }

    inline piece get_square(const uint_fast8_t &index) const {
        return board[index];
    }


    inline void set_square(const square_index &pos, const piece &new_piece) {

        static const uint32_t mat_conf_add[16] = {
        0, 1 << 0, 1 << 8,  1 << 14, 1 << 20, 1 << 26, 0, 0,
        0, 1 << 4, 1 << 11, 1 << 17, 1 << 23, 1 << 29, 0, 0,
        };

        piece old_piece = board[pos.index];

        bool old_color = (old_piece.get_player() == BLACK);
        bool new_color = (new_piece.get_player() == BLACK);

        bitboard bb = (uint64_t)0x1 << pos.index;

        bitboards[old_piece.get_type()][old_color] &= ~bb;
        bitboards[new_piece.get_type()][new_color] |=  bb;

        if (old_piece.get_type() != EMPTY) {
            pieces_by_color[old_color] &= ~bb;
            material_conf -= mat_conf_add[old_piece.d];
            if ((flags & INCREMENT_NNUE) != 0) {
                nnue->unset_piece(old_piece, pos.index);
            }
        }
        if (new_piece.get_type() != EMPTY) {
            pieces_by_color[new_color] |= bb;
            material_conf += mat_conf_add[new_piece.d];
            if ((flags & INCREMENT_NNUE) != 0) {
                nnue->set_piece(new_piece, pos.index);
            }
        }

        board[pos.index] = new_piece;
    }

    std::vector<square_index> get_legal_moves(square_index pos) const;
    std::vector<square_index> get_capture_moves(square_index pos) const;
    std::vector<chess_move> get_all_legal_moves(player_type_t player) const;
    int count_legal_moves(player_type_t player) const;


    bool causes_check(chess_move &mov, player_type_t player) const {
        bool color = (player == BLACK);

        bitboard occupation = pieces_by_color[0] | pieces_by_color[1];

        int king_sq = white_king_square.index;
        if (player == BLACK) {
            king_sq = black_king_square.index;
        }
        if (mov.from == king_sq) {
            king_sq = mov.to.index;
        }

        bitboard pieces[8][2];
        for (int i = PAWN; i <= KING; i++) {
            pieces[i][0] = bitboards[i][0];
            pieces[i][1] = bitboards[i][1];
        }

        bitboard from_bb = ((uint64_t)0x1 << mov.from.index);
        bitboard to_bb = ((uint64_t)0x1 << mov.to.index);

        occupation &= ~from_bb;

        piece moving = get_square(mov.from);
        piece captured = get_square(mov.to);

        bool moving_color = (moving.get_player() == BLACK);

        pieces[captured.get_type()][!moving_color] &= ~to_bb;
        pieces[moving.get_type()][moving_color] &= ~from_bb;
        pieces[(mov.promotion == 0 ? moving.get_type() : mov.promotion)][moving_color] |= to_bb;

        if (mov.to == en_passant_square && moving.get_type() == PAWN && (flags & EN_PASSANT_AVAILABLE) != 0) {
            occupation &= ~((uint64_t)0x1 << en_passant_target_square.index);
            pieces[PAWN][!moving_color] &= ~((uint64_t)0x1 << en_passant_target_square.index);
        }

        if (moving.get_type() == KING) {
            if (mov.to.get_x() == 2 && mov.from.get_x() == 4) {
                square_index old_rook_sq = square_index(0, mov.from.get_y());
                square_index new_rook_sq = square_index(3, mov.from.get_y());
                occupation &= ~((uint64_t)0x1 << old_rook_sq.index);
                occupation |=  ((uint64_t)0x1 << new_rook_sq.index);

                pieces[ROOK][moving_color] &= ~((uint64_t)0x1 << old_rook_sq.index);
                pieces[ROOK][moving_color] |=  ((uint64_t)0x1 << new_rook_sq.index);
            } else if (mov.to.get_x() == 6 && mov.from.get_x() == 4) {
                square_index old_rook_sq = square_index(7, mov.from.get_y());
                square_index new_rook_sq = square_index(5, mov.from.get_y());
                occupation &= ~((uint64_t)0x1 << old_rook_sq.index);
                occupation |=  ((uint64_t)0x1 << new_rook_sq.index);

                pieces[ROOK][moving_color] &= ~((uint64_t)0x1 << old_rook_sq.index);
                pieces[ROOK][moving_color] |=  ((uint64_t)0x1 << new_rook_sq.index);
            }
        }

        occupation |= to_bb;

        bitboard mask = ~(0ull);

        bitboard attacks[7] = {0, bitboard_utils.pawn_attack(king_sq, occupation, color, mask, 0),
                                  bitboard_utils.knight_attack(king_sq, mask),
                                  bitboard_utils.bishop_attack(king_sq, occupation, mask),
                                  bitboard_utils.rook_attack(king_sq, occupation, mask),
                                  bitboard_utils.queen_attack(king_sq, occupation, mask),
                                  bitboard_utils.king_attack(king_sq, mask)};

        for (int i = PAWN; i <= KING; i++) {
            if ((attacks[i] & pieces[i][!color]) != 0) {
                return true;
            }
        }
        return false;
    }


    bool is_square_threatened(square_index pos, player_type_t player) const {
        bool color = (player == BLACK);

        bitboard mask = ~(0ull);

        bitboard occupation = pieces_by_color[0] | pieces_by_color[1];

        bitboard attacks[7] = {0, bitboard_utils.pawn_attack(pos.index, occupation, color, mask, 0),
                                  bitboard_utils.knight_attack(pos.index, mask),
                                  bitboard_utils.bishop_attack(pos.index, occupation, mask),
                                  bitboard_utils.rook_attack(pos.index, occupation, mask),
                                  bitboard_utils.queen_attack(pos.index, occupation, mask),
                                  bitboard_utils.king_attack(pos.index, mask)};

        for (int i = PAWN; i <= KING; i++) {
            if ((attacks[i] & bitboards[i][!color]) != 0) {
                return true;
            }
        }

        return false;
    }

    bool is_square_threatened_by_pawn(const square_index &pos, const player_type_t &player) const {
        bool color = (player == BLACK);

        bitboard pawn_attack = bitboard_utils.pawn_attack(pos.index, ~((uint64_t)0), color, ~((uint64_t)0), 0);

        return ((pawn_attack & bitboards[PAWN][!color]) != 0);
    }

    bool is_square_threatened_by_minor(const square_index &pos, const player_type_t &player) const {
        bool color = (player == BLACK);

        bitboard occupation = pieces_by_color[0] | pieces_by_color[1];

        bitboard bishop_style = bitboard_utils.bishop_attack(pos.index, occupation, ~(uint64_t)0);
        bitboard knight_style = bitboard_utils.knight_attack(pos.index, ~(uint64_t)0);

        return (((bishop_style & bitboards[BISHOP][!color]) != 0) ||
                ((knight_style & bitboards[KNIGHT][!color]) != 0));
    }

    bool is_square_threatened_by_major(square_index pos, player_type_t player) const {
        bool color = (player == BLACK);

        bitboard occupation = pieces_by_color[0] | pieces_by_color[1];

        bitboard bishop_style = bitboard_utils.bishop_attack(pos.index, occupation, ~(uint64_t)0);
        bitboard rook_style = bitboard_utils.rook_attack(pos.index, occupation, ~(uint64_t)0);

        bitboard queen_style = bishop_style | rook_style;

        return (((rook_style  & bitboards[ROOK][!color]) != 0) ||
                ((queen_style & bitboards[QUEEN][!color]) != 0));
    }

    bool is_square_threatened_by_lesser(square_index pos, player_type_t player, piece_type_t piece) const {
        bool color = (player == BLACK);
        if (piece == PAWN) {
            return false;
        } else if (piece <= BISHOP) {
            bitboard pawn_style = bitboard_utils.pawn_attack(pos.index, ~(uint64_t)0, color, ~(uint64_t)0, 0);
            return ((pawn_style & bitboards[PAWN][!color]) != 0);
        } else {
            bitboard occupation = pieces_by_color[0] | pieces_by_color[1];

            bitboard knight_style = bitboard_utils.knight_attack(pos.index, ~(uint64_t)0);
            bitboard pawn_style = bitboard_utils.pawn_attack(pos.index, occupation, color, ~(uint64_t)0, 0);
            bitboard bishop_style = bitboard_utils.bishop_attack(pos.index, occupation, ~(uint64_t)0);

            if ((pawn_style & bitboards[PAWN][!color]) != 0 ||
                (knight_style & bitboards[KNIGHT][!color]) != 0 ||
                (bishop_style & bitboards[BISHOP][!color]) != 0) {
                return true;
            }

            bitboard rook_style = bitboard_utils.rook_attack(pos.index, occupation, ~(uint64_t)0);

            switch (piece)
            {
            case KING:
                if (((bishop_style | rook_style) & bitboards[QUEEN][!color]) != 0) {
                    return true;
                }
            case QUEEN:
                if ((rook_style & bitboards[ROOK][!color]) != 0) {
                    return true;
                }
            default:
                return false;
            }
        }

        return false;
    }

    piece_type_t least_valuable_threatener(square_index pos, player_type_t player) const {
        bitboard mask = ~(1ull << pos.index);
        bitboard occupation = pieces_by_color[0] | pieces_by_color[1];

        bool color = (player == BLACK);

        bitboard attacks[7] = {0, bitboard_utils.pawn_attack(pos.index, occupation, color, mask, 0),
                                  bitboard_utils.knight_attack(pos.index, mask),
                                  bitboard_utils.bishop_attack(pos.index, occupation, mask),
                                  bitboard_utils.rook_attack(pos.index, occupation, mask),
                                  bitboard_utils.queen_attack(pos.index, occupation, mask),
                                  bitboard_utils.king_attack(pos.index, mask)};

        for (int i = PAWN; i <= KING; i++) {
            if ((attacks[i] & bitboards[i][!color]) != 0) {
                return (piece_type_t)i;
            }
        }
        return EMPTY;
    }

    bool in_check(player_type_t player) const {
        if (player == WHITE) {
            return is_square_threatened(white_king_square, WHITE);
        } else {
            return is_square_threatened(black_king_square, BLACK);
        }
    }

    int count_pieces() const {
        return pop_count(pieces_by_color[0] | pieces_by_color[1]);
    }

    int count_non_pawn_pieces(player_type_t player) const {
        bool color = (player == BLACK);
        return pop_count(pieces_by_color[color] & (~bitboards[PAWN][color]));
    }

    int count_non_pawn_pieces() const {
        return count_non_pawn_pieces(BLACK) + count_non_pawn_pieces(WHITE);
    }

    bool is_insufficient_material() const {
        int num_of_pieces = pop_count(pieces_by_color[0] | pieces_by_color[1]);

        if (num_of_pieces == 2) {
            return true;
        } else if (num_of_pieces == 3) {
            if (pop_count(bitboards[BISHOP][0] | bitboards[BISHOP][1]) == 1) {
                return true;
            } else if (pop_count(bitboards[KNIGHT][0] | bitboards[KNIGHT][1]) == 1) {
                return true;
            }
        } else if (num_of_pieces == 4) {
            if (pop_count(bitboards[BISHOP][0]) == 1 &&
                pop_count(bitboards[BISHOP][1]) == 1) {

                square_index sq0 = square_index(bit_scan_forward(bitboards[BISHOP][0]));
                square_index sq1 = square_index(bit_scan_forward(bitboards[BISHOP][1]));

                if (sq0.is_dark() == sq1.is_dark()) {
                    return true;
                }
            }
        }
        return false;
    }



    unmake_restore make_null_move();
    void unmake_null_move(const unmake_restore &restore);
    unmake_restore make_move(chess_move m, uint64_t next_zhash);
    unmake_restore make_move(chess_move m);

    void unmake_move(chess_move m, const unmake_restore &restore);

    inline player_type_t get_turn() const {
        if ((half_move_clock & 0x1) == 0) {
            return WHITE;
        } else {
            return BLACK;
        }
    }

    bool operator == (const board_state &other) const {
        if (en_passant_square != other.en_passant_square ||
            en_passant_target_square != other.en_passant_target_square ||
            flags != other.flags ||
            half_move_clock != other.half_move_clock) {
            return false;
        }

        for (int i = 0; i < BOARD_SQUARES; i++) {
            if (board[i] != other.board[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator != (const board_state &other) const {
        return !(*this == other);
    }



    int check_repetition(const uint64_t &zh)
    {
        if (hash_stack_top == 0) {
            return 0;
        }
        uint64_t *ptr = &hash_stack[hash_stack_top-1];

        int c = 0;
        while (ptr >= hash_stack) {
            if (*ptr == zh) {
                c++;
            } else if (*ptr == 0) {
                break;
            }
            ptr--;
        }
        return c;
    }

    int check_repetition() {
        return check_repetition(zhash);
    }

    uint32_t minor_hash() const {
        return ((structure_hash & MINOR_STRUCTURE_HASH_MASK) >> MINOR_STRUCTURE_HASH_SHIFT);
    }
    uint32_t major_hash() const {
        return ((structure_hash & MAJOR_STRUCTURE_HASH_MASK) >> MAJOR_STRUCTURE_HASH_SHIFT);
    }
    uint32_t pawn_hash() const {
        return ((structure_hash & PAWN_STRUCTURE_HASH_MASK) >> PAWN_STRUCTURE_HASH_SHIFT);
    }

    uint32_t material_hash() const {
        uint32_t h = material_conf;

        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;

        return h;
    }

    uint64_t zhash;
    uint64_t structure_hash;

    void recalculate_bitboards();
    void recalculate_piece_counter();
    bool validate_bitboards() const;


    bitboard bitboards[8][2];
    bitboard pieces_by_color[2];


    void load_fen(std::string fen_str);
    std::string generate_fen() const;

    square_index white_king_square;
    square_index black_king_square;

    piece board[BOARD_SQUARES];
    uint16_t flags;
    square_index en_passant_square;
    square_index en_passant_target_square;
    uint16_t half_move_clock;
    uint32_t material_conf;

private:
    void push_hash_stack(uint64_t zhash, bool irreversible)
    {
        if (hash_stack_top >= 1022) {
            std::cout << "Board state hash stack overflow!" << std::endl;
            return;
        }

        if (irreversible) {
            hash_stack[hash_stack_top] = 0;
            hash_stack_top += 1;
        }

        hash_stack[hash_stack_top] = zhash;
        hash_stack_top += 1;
    }

    void pop_hash_stack()
    {
        hash_stack_top -= 1;
        if (hash_stack_top > 0) {
            if (hash_stack[hash_stack_top-1] == 0) {
                hash_stack_top--;
            }
        }
        if (hash_stack_top < 0) {
            std::cout << "Board state hash stack underflow!" << std::endl;
            hash_stack_top = 0;
        }
    }

    uint64_t hash_stack[1024];
    int hash_stack_top;
};








