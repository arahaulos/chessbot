#include "game.hpp"
#include <iostream>
#include <algorithm>
#include "zobrist.hpp"

game_state::game_state()
{
    state.set_initial_state();

    error = NO_ERROR;
}

void game_state::reset()
{
    state.set_initial_state();
    moves_played = std::vector<chess_move>();
    unmoves = std::vector<unmake_restore>();

    error = NO_ERROR;
}

bool game_state::is_check()
{
    return state.in_check(state.get_turn());
}

bool game_state::is_checkmate()
{
    if (is_check() && state.count_legal_moves(state.get_turn()) == 0) {
        return true;
    } else {
        return false;
    }
}

bool game_state::is_stalemate()
{
    if (!is_check() && state.count_legal_moves(state.get_turn()) == 0) {
        return true;
    } else {
        return false;
    }
}

bool game_state::is_insufficient_material()
{
    return state.is_insufficient_material();
}

void game_state::unmove()
{
    if (!moves_played.empty()) {
        chess_move last_move = moves_played.back();
        unmake_restore last_restore = unmoves.back();

        moves_played.pop_back();
        unmoves.pop_back();

        state.unmake_move(last_move, last_restore);
    }
}


game_win_type_t game_state::make_move(const chess_move &mov)
{
    if (get_turn() != state.get_square(mov.from).get_player()) {
        std::cout << "Wrong turn!" << std::endl;
        error = WRONG_TURN_ERROR;
        return NO_WIN;
    }

    if (mov.to == mov.from) {
        std::cout << "Error: null move not allowed" << std::endl;
        error = NULL_MOVE_ERROR;
        return NO_WIN;
    }
    player_type_t turn = state.get_turn();

    std::vector<square_index> valid_moves = state.get_legal_moves(mov.from);

    if (std::find(valid_moves.begin(), valid_moves.end(), mov.to) != std::end(valid_moves)) {
        unmake_restore restore = state.make_move(mov);

        moves_played.push_back(mov);
        unmoves.push_back(restore);

        if (is_checkmate()) {
            if (turn == BLACK) {
                return BLACK_WIN;
            } else {
                return WHITE_WIN;
            }
        }
        if (is_stalemate()) {
            return DRAW;
        }

        if (state.check_repetition() >= 3 || state.half_move_clock >= 99) {
            return DRAW;
        }

        if (is_insufficient_material()) {
            return DRAW;
        }

    } else {
        std::cout << state.generate_fen() << " " << mov.to_uci() << std::endl;
        std::cout << "Error: move is not legal" << std::endl;
        error = ILLEGAL_MOVE_ERROR;
    }
    return NO_WIN;

}

