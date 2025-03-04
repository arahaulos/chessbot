#pragma once

#include <string>
#include "game.hpp"
#include "search.hpp"


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

bool load_selfplay_results(std::vector<selfplay_result> &results, std::string filename);
bool save_selfplay_results(std::vector<selfplay_result> &results, std::string filename);


struct tuning_utility
{
    static void selfplay(std::string output_file, std::string nnue_file, int threads, int games, int depth, int nodes, bool random_opening);

    static int test(int games, int time, int time_inc, alphabeta_search &bot0, alphabeta_search &bot1, int workers, std::string uci_engine, double elo0 = 0.0f, double elo1 = 3.0f, double alpha = 0.05f, double beta = 0.05f);
    static void tune_search_params(int time, int time_inc, int threads, int num_of_games, const std::vector<std::string> &params_to_tune);
};
