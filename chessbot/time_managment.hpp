#pragma once




inline int get_target_search_time(int time_left, int time_inc)
{
    int base_time = (time_left / 25) + time_inc;

    return std::min(time_left-100, base_time);
}

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
    time_manager(int tleft, int tinc) {
        init(tleft, tinc);
    }

    void init(int tleft, int tinc) {
        time_left = tleft;
        time_inc = tinc;

        int base_time = (time_left / 25) + time_inc;
        int max_base_time = (time_left / 15) + time_inc;

        max_time = std::min(time_left-100, max_base_time);
        target_time = std::min(time_left-100, base_time);

        time_used = 0;

        iterations.clear();
    }

    bool end_of_iteration(int depth, chess_move bm, int32_t eval, int32_t time)
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

            int time_modifier = 25;
            if (bm_stable && eval_stable) {
                time_modifier = 30;
            } else if (!bm_stable && eval_stable) {
                time_modifier = 20;
            } else if (!bm_stable && !eval_stable) {
                time_modifier = 15;
            }
            target_time = std::min((time_left / time_modifier) + time_inc, max_time);
        }
        return should_stop_new_iteration(time);
    }

    float avg_branching_factor()
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

    bool should_stop_new_iteration(int tused)
    {
        time_used = tused;

        if (iterations.size() > 10) {
            int prev_iteration_time = iterations[iterations.size()-1].timepoint - iterations[iterations.size()-2].timepoint;

            int estimated_complete_time = time_used + std::clamp(avg_branching_factor(), 1.0f, 2.0f)*prev_iteration_time;

            return (estimated_complete_time > target_time);
        } else {
            return (time_used > target_time);
        }
    }

    bool should_stop_search(int tused) {
        time_used = tused;
        return (time_used > target_time);
    }

    int get_time_used()
    {
        return time_used;
    }

    bool experimental_features;
private:
    int time_left;
    int time_inc;

    std::vector<search_iteration_result> iterations;

    int target_time;
    int max_time;

    int time_used;
};
