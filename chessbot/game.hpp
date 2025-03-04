#pragma once

#include "state.hpp"
#include <stack>
#include <string>

enum game_win_type_t {NO_WIN = 0, WHITE_WIN = 1, BLACK_WIN = 2, DRAW = 3};
enum game_error_type_t {NO_ERROR = 0, WRONG_TURN_ERROR = 1, ILLEGAL_MOVE_ERROR = 2, NULL_MOVE_ERROR = 3};

inline std::string win_type_to_str(game_win_type_t win)
{
    if (win == WHITE_WIN) {
        return "White won!";
    } else if (win == BLACK_WIN) {
        return "Black won!";
    } else if (win == DRAW) {
        return "Draw!";
    } else {
        return "";
    }
}

class game_state
{
public:
    game_state();

    void reset();

    player_type_t get_turn() {
        return state.get_turn();
    }

    board_state &get_state() {
        return state;
    }

    game_win_type_t make_move(const chess_move &mov);
    void unmove();

    std::vector<chess_move> moves_played;

    game_error_type_t get_last_error() {
        game_error_type_t e = error;
        error = NO_ERROR;
        return e;
    }
    bool is_check();
    bool is_checkmate();
    bool is_stalemate();
    bool is_insufficient_material();

private:

    board_state state;

    std::vector<unmove_data> unmoves;
    game_error_type_t error;

};
