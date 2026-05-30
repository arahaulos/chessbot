#pragma once

#include <memory>

#include "chessbot/game.hpp"
#include "chessbot/search.hpp"
#include "chessbot/uci.hpp"


struct position_analysis_result
{
    chess_move bm;
    chess_move played;

    int cp;

    float win_p;
    float draw_p;
    float loss_p;
};



class application
{
public:
    application();

    void run();
    void run_tests();
    void run_benchmark();
    void eval_trace(std::string fen);

    std::vector<position_analysis_result> analyze_game(std::string startpos, std::vector<chess_move> &moves, int min_nodes, int min_depth);
    float game_accuracy(std::vector<position_analysis_result> &analysis, player_type_t player);


    std::shared_ptr<searcher> get_searcher() {
        return alphabeta;
    }

private:
    int perft_test(std::string position_fen, std::vector<int> expected_results);
    int test_incremental_updates();

    uint64_t bench_position(std::string position_fen, int depth);

    std::shared_ptr<game_state> game;
    std::shared_ptr<searcher> alphabeta;
    std::shared_ptr<uci_interface> uci;

    std::string pgn_lines;
};
