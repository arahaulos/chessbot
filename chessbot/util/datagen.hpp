#pragma once

#include <string>

struct datagen_config
{
    int threads;
    int games;
    int depth;
    int nodes;
    bool forward_pruning;
    bool multi_pv_opening;
    bool filter_dublicate_openings;
    int opening_moves;
    std::string opening_suite;

    bool adjucate_wins;
    int adjucate_win_cp_treshold;
    int adjucate_win_min_plies;

    bool adjucate_draws;
    int adjucate_draw_min_plies;
    int adjucate_draw_cp_treshold;
    int adjucate_draw_plies_below_treshold;
};


struct training_datagen
{
    static void datagen(std::string output_file, datagen_config &config);
};
