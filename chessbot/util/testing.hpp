#pragma once

#include "../search.hpp"
#include <string>

struct testing_utility
{
    static int test(int games, int time, int time_inc, alphabeta_search &bot0, alphabeta_search &bot1, int threads, std::string opening_suite, double elo0 = 0.0f, double elo1 = 3.0f, double alpha = 0.05f, double beta = 0.05f);
};
