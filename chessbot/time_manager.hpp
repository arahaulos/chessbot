#pragma once

#include "state.hpp"
#include <vector>


struct search_iteration_result
{
    search_iteration_result(int d, chess_move m, int32_t e, int32_t t): depth(d), bm(m), eval(e), timepoint(t) {};
    int depth;
    chess_move bm;
    int32_t eval;
    int32_t timepoint;
};

struct time_manager
{
    time_manager();
    time_manager(int tleft, int tinc, int ply);

    static int get_default_overhead();
    void init(int tleft, int tinc, int ply);

    bool end_of_iteration(int depth, chess_move bm, int32_t eval, int32_t time);

    float avg_branching_factor();

    bool should_stop_new_iteration(int tused);

    bool should_stop_search(int tused) {
        time_used = tused;
        return (time_used > target_time);
    }

    int get_time_used();

    void set_overhead(int val);

    bool test_flag;
private:
    float move_overhead;
    float time_left;
    float time_inc;

    std::vector<search_iteration_result> iterations;

    float target_time;
    float max_time;

    float time_used;

    float opening_factor;
};
