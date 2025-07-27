#pragma once

#include <limits>


constexpr int MAX_DEPTH = 128;
constexpr int KILLER_MOVE_SLOTS = 2;
constexpr int MAX_THREADS = 32;
constexpr int MIN_THREADS = 1;
constexpr int MAX_MULTI_PV = 16;


constexpr int TT_SIZE = 64;
constexpr int EVAL_CACHE_SIZE = 4;
constexpr int DEFAULT_THREADS = 1;
constexpr int TT_ENTRIES_IN_BUCKET = 4;

constexpr int BOARD_WIDTH = 8;
constexpr int BOARD_HEIGHT = 8;
constexpr int BOARD_SQUARES = BOARD_WIDTH*BOARD_HEIGHT;

enum node_type_t {PV_NODE = 1, CUT_NODE = 2, ALL_NODE = 3};



enum piece_type_t: unsigned char {PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6, EMPTY = 0};
enum player_type_t: unsigned char {WHITE = 0x00, BLACK = 0x8, EMPTY_COLOR = WHITE};


enum board_state_flags {WHITE_QSIDE_CASTLE_VALID = (1 << 0), WHITE_KSIDE_CASTLE_VALID = (1 << 1), BLACK_QSIDE_CASTLE_VALID = (1 << 2), BLACK_KSIDE_CASTLE_VALID = (1 << 3), EN_PASSANT_AVAILABLE = (1 << 4), NULL_MOVE = (1 << 5), INCREMENT_NNUE = (1 << 6)};


enum piece_player_type_t: unsigned char {WHITE_PAWN = 1,       WHITE_KNIGHT = 2,       WHITE_BISHOP = 3,       WHITE_ROOK = 4,       WHITE_QUEEN = 5,       WHITE_KING = 6,
                                         BLACK_PAWN = 1 | 0x8, BLACK_KNIGHT = 2 | 0x8, BLACK_BISHOP = 3 | 0x8, BLACK_ROOK = 4 | 0x8, BLACK_QUEEN = 5 | 0x8, BLACK_KING = 6 | 0x8};


constexpr int32_t MAX_EVAL = std::numeric_limits<int32_t>::max() - 65536;
constexpr int32_t MIN_EVAL = -MAX_EVAL;

constexpr int32_t INVALID_EVAL = MIN_EVAL;

inline int get_mate_distance(int32_t score)
{
    if (score < MIN_EVAL + MAX_DEPTH) {
        return (score - MIN_EVAL);
    } else if (score > MAX_EVAL - MAX_DEPTH) {
        return (MAX_EVAL - score);
    }
    return 0;
}

inline bool is_mate_score(int32_t score)
{
    if (score > MIN_EVAL && score < MAX_EVAL) {
        if (score <= MIN_EVAL + MAX_DEPTH || score >= MAX_EVAL - MAX_DEPTH) {
            return true;
        }
    }
    return false;
}

inline bool is_score_valid(int32_t score) {
    return (score > MIN_EVAL && score < MAX_EVAL);
}
