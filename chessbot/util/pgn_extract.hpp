#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <atomic>
#include "../state.hpp"
#include "misc.hpp"



struct pgn_extractor
{
    pgn_extractor(int num_of_threads, float extract_rate, int nodes_per_position);

    ~pgn_extractor();

    void add_pgn(const std::string &str);

    int pgns_queued();

    int get_total_pgns();

    std::vector<selfplay_result> get_results();
    void worker_thread(float extract_rate, int nodes_per_position);

private:
    bool pop_pgn_queue(std::string &pgn);
    void push_evaluations(const std::vector<position_evaluation> &evals, game_win_type_t result);

    std::vector<std::thread> workers;
    std::mutex queue_lock;
    std::mutex results_lock;
    std::queue<std::string> pgn_queue;
    std::vector<selfplay_result> results;

    std::atomic<bool> running;

    int total_pgns;
};

namespace pgn_extract
{
    void create_dataset(const std::string &pgn_folder, std::string output_file, float max_elo_diff, float min_elo, float exctract_rate, int nodes);
}
