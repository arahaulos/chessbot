#pragma once

#include <memory>

#include "chessbot/game.hpp"
#include "chessbot/search.hpp"
#include "chessbot/uci.hpp"




class application
{
public:
    application();

    void run();
    void run_tests();
    void run_benchmark();
    void eval_trace(std::string fen);
    void datagen(std::string folder, std::string nnue_file, std::string opening_suite, int threads, int depth, int nodes, bool multi_pv_opening, int opening_moves, int games_per_file, int max_files);
private:
    int perft_test(std::string position_fen, std::vector<int> expected_results);
    int test_incremental_updates();

    uint64_t bench_position(std::string position_fen, int depth);

    std::shared_ptr<game_state> game;
    std::shared_ptr<alphabeta_search> alphabeta;
    std::shared_ptr<uci_interface> uci;
};
