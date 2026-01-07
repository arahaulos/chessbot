#pragma once

#include <math.h>
#include "../game.hpp"
#include "../search.hpp"

struct position_evaluation
{
    position_evaluation(const board_state &state, int32_t score, chess_move mov)
    {
        fen = state.generate_fen();
        eval = score;
        bm = mov;
    }

    std::string fen;
    int32_t eval;
    chess_move bm;
};



struct selfplay_result
{
    //fen;bestmove;score;wdl
    selfplay_result() {};


    selfplay_result(const std::string &p, const std::string &best_move, int32_t score, float w)
    {
        fen = p;
        wdl = w;
        eval = score;
        bm.from_uci(best_move);
    }

    selfplay_result(const position_evaluation &pos, game_win_type_t win)
    {
        fen = pos.fen;
        if (win == WHITE_WIN) {
            wdl = 1.0f;
        } else if (win == BLACK_WIN) {
            wdl = 0.0f;
        } else {
            wdl = 0.5f;
        }

        bm = pos.bm;
        eval = pos.eval;
    }

    std::string fen;
    float wdl;
    int32_t eval;
    chess_move bm;
};


std::vector<std::string> load_opening_suite(std::string filename);
bool save_selfplay_results(std::vector<selfplay_result> &results, std::string filename);
bool load_selfplay_results(std::vector<selfplay_result> &results, std::string filename);

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


int play_game_pair(searcher &s0, searcher &s1, const std::string &opening_fen, int base_time, int time_inc, bool s0_playing_black, match_stats &stats);
