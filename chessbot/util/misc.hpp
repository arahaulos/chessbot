#pragma once

#include <math.h>
#include <functional>
#include "../game.hpp"
#include "../search.hpp"

inline double score_to_elo(double score)
{
    return -std::log10(1.0/score-1.0)*400.0;
}

inline double elo_to_score(double elo)
{
   return 1.0f / (std::pow(10.0f, -elo/400.0f)+1.0f);
}

struct match_stats
{
    match_stats() {
        wins = 0;
        losses = 0;
        draws = 0;

        s0_depth = 0;
        s1_depth = 0;

        s0_nps = 0;
        s1_nps = 0;

        s0_moves = 0;
        s1_moves = 0;
    }


    int wins;
    int losses;
    int draws;

    int s0_depth;
    int s1_depth;

    int64_t s0_nps;
    int64_t s1_nps;

    int s0_moves;
    int s1_moves;
};


std::vector<std::string> load_opening_suite(std::string filename);

int play_game_pair(searcher &s0, searcher &s1, const std::string &opening_fen, int base_time, int time_inc, bool s0_playing_black, match_stats &stats);


void iterate_pgn_positions(const std::string &pgn_text, std::function<void(const board_state &state, chess_move next_mov, game_win_type_t game_result, std::string *comment)> callback);
