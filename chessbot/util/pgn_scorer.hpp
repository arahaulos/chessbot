#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <atomic>
#include "../state.hpp"
#include "misc.hpp"



struct pgn_scorer_pool
{
    pgn_scorer_pool(int num_of_threads, int nodes_per_position);

    ~pgn_scorer_pool();

    void add_pgn(const std::string &str);

    int pgns_queued();

    int get_total_pgns();

    std::string get_result();
    void worker_thread(int nodes_per_position);

private:
    bool pop_pgn_queue(std::string &pgn);

    std::vector<std::thread> workers;
    std::mutex queue_lock;
    std::mutex results_lock;
    std::queue<std::string> pgn_queue;
    std::vector<std::string> results;

    std::atomic<bool> running;

    int total_pgns;
};

namespace pgn_scorer
{
    void create_dataset(const std::string &pgn_folder, std::string output_file, float max_elo_diff, float min_elo, int nodes);
}
