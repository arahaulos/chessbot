#pragma once

#include <vector>
#include <string>

struct tuning_utility
{
    static void tune_search_params(int time, int time_inc, int threads, int num_of_games, std::string output_file, const std::vector<std::string> &params_to_tune, std::string opening_suite);
};
