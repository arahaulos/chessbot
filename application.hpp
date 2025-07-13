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

    void selfplay(std::string folder, std::string nnue_file, int threads, int depth, int nodes, bool random, int games_per_file, int max_files);
private:
    int perft_test(std::string position_fen, std::vector<int> expected_results);
    int test_incremental_updates();

    std::shared_ptr<game_state> game;
    std::shared_ptr<alphabeta_search> alphabeta;
    std::shared_ptr<uci_interface> uci;
};
