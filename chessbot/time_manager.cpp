#include "time_manager.hpp"


time_manager::time_manager() {
    test_flag = false;
    move_overhead = get_default_overhead();
}

time_manager::time_manager(int tleft, int tinc) {
    test_flag = false;
    move_overhead = get_default_overhead();
    init(tleft, tinc);
}

int time_manager::get_default_overhead() {
    return 100;
}

void time_manager::init(int tleft, int tinc) {
    time_left = tleft - move_overhead;
    time_inc = tinc;

    int base_time = (time_left / 18) + time_inc/2;
    int max_base_time = (time_left / 8) + time_inc/2;

    max_time = std::min(time_left, max_base_time);
    target_time = std::min(time_left, base_time);

    time_used = 0;

    iterations.clear();
}

bool time_manager::end_of_iteration(int depth, chess_move bm, int32_t eval, int32_t time)
{
    time_used = time;

    iterations.emplace_back(depth, bm, eval, time);

    if (iterations.size() > 10) {
        bool bm_stable = true;
        bool eval_stable = true;

        int iterations_to_compare = 5;
        for (int i = iterations.size()-2; i >= iterations.size()-iterations_to_compare-2; i--) {
            if (iterations[i+1].bm != iterations[i].bm) {
                bm_stable = false;
            }

            if (eval < 300 && (std::abs(iterations[i].eval - iterations[i+1].eval) > 50 || std::abs(iterations[i-1].eval - iterations[i+1].eval) > 100)) {
                eval_stable = false;
            }
        }

        int time_modifier = 18;
        if (bm_stable && eval_stable) {
            time_modifier = 25;
        } else if ((!bm_stable && eval_stable) || (!eval_stable && bm_stable)) {
            time_modifier = 14;
        } else if (!bm_stable && !eval_stable) {
            time_modifier = 8;
        }
        target_time = std::min((time_left / time_modifier) + time_inc/2, max_time);
    }
    return should_stop_new_iteration(time);
}

float time_manager::avg_branching_factor()
{
    float branching_factor = 0.0f;
    int branching_factor_samples = 0;
    for (int i = 1; i < iterations.size()-1; i++) {
        int ittime0 = iterations[i+1].timepoint - iterations[i].timepoint;
        int ittime1 = iterations[i].timepoint - iterations[i-1].timepoint;

        if (ittime0 > 5 && ittime1 > 5) {
            branching_factor += (float)ittime0 / ittime1;
            branching_factor_samples++;
        }
    }
    if (branching_factor_samples > 0) {
        return branching_factor / branching_factor_samples;
    } else {
        return 2.0;
    }
}

bool time_manager::should_stop_new_iteration(int tused)
{
    time_used = tused;

    if (iterations.size() > 10) {
        int prev_iteration_time = iterations[iterations.size()-1].timepoint - iterations[iterations.size()-2].timepoint;

        int estimated_complete_time = time_used + std::clamp(avg_branching_factor(), 1.0f, 2.0f)*prev_iteration_time / 2;

        return (estimated_complete_time > target_time);
    } else {
        return (time_used > target_time);
    }
}


int time_manager::get_time_used()
{
    return time_used;
}

void time_manager::set_overhead(int val) {
    move_overhead = val;
}
