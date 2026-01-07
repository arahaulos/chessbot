#include "tuning.hpp"
#include <memory>
#include <thread>
#include <fstream>
#include <atomic>
#include <cstdio>
#include "misc.hpp"
#include "testing.hpp"



float remap_value(float val, float min0, float max0, float min1, float max1)
{
    float frac = (val - min0) / (max0 - min0);
    return frac*(max1 - min1) + min1;
}


//This is test model for testing convergence of algorithms
constexpr int strength_table_res = 128;
struct tuning_test_model
{
    tuning_test_model() {
        strength_table.resize(search_params::num_of_params()*strength_table_res);
        for (float &val : strength_table) {
            val = 0.0f;
        }
    }


    static tuning_test_model get_test_model()
    {
        tuning_test_model m;

        m.add_datapoints_for_param("rfmargin_base", {{40, 2900}, {50, 2960}, {55, 3008}, {60, 3015}, {65, 3012}, {70, 3005}});
        m.add_datapoints_for_param("rfmargin_mult", {{40, 2800}, {50, 2940}, {55, 3000}, {60, 3050}, {65, 3005}, {100,2980}});
        m.add_datapoints_for_param("rfmargin_improving_modifier", {{20, 2995}, {80, 3005}, {100,3000}});
        m.add_datapoints_for_param("fmargin_base", {{40, 2980}, {50, 2995}, {55, 3000}, {60, 3010}, {65, 3000}, {100,3000}});
        m.add_datapoints_for_param("fmargin_mult", {{40, 2700}, {50, 2900}, {55, 3050}, {60, 3020}, {65, 3000}, {150,2990}});
        m.add_datapoints_for_param("hmargin_mult", {{2000, 2400}, {3000, 2900}, {4000, 3020}, {5000, 3000}, {6000, 2990}, {7000, 2990}});
        m.add_datapoints_for_param("lmr_hist_adjust", {{4000, 2000}, {5000, 2900}, {6000, 3020}, {7000, 3030}, {8000, 2990}, {9000, 2990}});
        m.add_datapoints_for_param("cap_see_margin_mult", {{80, 2000}, {100, 3050}, {110, 3020}, {150, 3010}, {200, 2950}});
        m.add_datapoints_for_param("quiet_see_margin_mult", {{10, 3000}, {15, 3020}, {20, 3020}, {50, 3000}});
        m.add_datapoints_for_param("lmr_modifier", {{10,2500}, {15, 3100}, {20, 3080}, {30, 2900}});
        m.add_datapoints_for_param("razoring_margin", {{150,2900}, {200, 3050}, {250, 2990}, {300, 2980}});
        m.add_datapoints_for_param("probcut_margin", {{100,2800}, {200, 3050}, {250, 3010}, {300, 3000}});
        m.add_datapoints_for_param("good_quiet_treshold", {{-3000,2800}, {-1500, 3000}, {1500, 3050}, {3000, 3000}});

        return m;
    }


    static int simulate_game(float strength0, float strength1, float draw_rate) {
        double expected_score = elo_to_score(strength0 - strength1);

        constexpr int rand_fracs = 500;

        int pwin = ((expected_score - 0.5*draw_rate)*rand_fracs + 0.5f);
        int pdraw = (draw_rate*rand_fracs + 0.5f);

        int r = rand() % (rand_fracs+1);

        if (r < pwin) {
            return 1;
        } else if (r < pwin + pdraw) {
            return 0;
        } else {
            return -1;
        }
    }

    int simulate_match(std::vector<float> &params0, std::vector<float> &params1, float draw_rate, int num_of_games)
    {
        float s0 = get_strength(params0);
        float s1 = get_strength(params1);

        int result = 0;
        for (int i = 0; i < num_of_games; i++) {
            result += simulate_game(s0, s1, draw_rate);
        }
        return result;
    }

    void add_datapoints_for_param(std::string param_name, std::vector<std::pair<int, float>> points)
    {
        std::sort(points.begin(), points.end(), [] (auto &a, auto &b) {return a.first < b.first;});

        int param = search_params::get_variable_index(param_name);

        int minv, maxv;
        search_params::get_limits(param, minv, maxv);

        for (int i = 0; i < points.size()-1; i++) {
            float x0 = points[i].first;
            float y0 = points[i].second;

            float x1 = points[i+1].first;
            float y1 = points[i+1].second;

            if (x0 == x1) {
                continue;
            }

            float b = (float)(y1 - y0) / (x1 - x0);
            float a =  y0 - (x0 * b);

            int tx0 = (int)(remap_value(x0, minv, maxv, 0, strength_table_res-1) + 0.5f);
            int tx1 = (int)(remap_value(x1, minv, maxv, 0, strength_table_res-1) + 0.5f);

            int stx = (i == 0 ? 0 : tx0);
            int etx = (i == points.size()-2 ? strength_table_res-1 : tx1);

            stx = std::clamp(stx, 0, strength_table_res-1);
            etx = std::clamp(etx, 0, strength_table_res-1);

            for (int tx = stx; tx <= etx; tx++) {
                float x = remap_value(tx, 0, strength_table_res-1, minv, maxv);

                strength_table[param*strength_table_res + tx] = a + b*x;
            }
        }
    }

    float get_strength(std::vector<float> &params)
    {
        float strength = 0;
        for (int i = 0; i < search_params::num_of_params(); i++) {
            strength += get_param_strength(i, params[i]);
        }
        return strength / search_params::num_of_params();
    }

    float get_maximum_strength()
    {
        float strength = 0;
        for (int i = 0; i < search_params::num_of_params(); i++) {
            float s = strength_table[i*strength_table_res + 0];
            for (int j = 1; j < strength_table_res; j++) {
                s = std::max(s, strength_table[i*strength_table_res + j]);
            }
            strength += s;
        }
        return strength / search_params::num_of_params();
    }

    float get_optimal_value_for_param(int param)
    {
        float max_strength = strength_table[param*strength_table_res + 0];
        int value_idx = 0;
        for (int i = 1; i < strength_table_res; i++) {
            float s = strength_table[param*strength_table_res + i];
            if (s > max_strength) {
                value_idx = i;
                max_strength = s;
            }
        }
        int minv, maxv;
        search_params::get_limits(param, minv, maxv);

        return remap_value(value_idx, 0, strength_table_res-1, minv, maxv);
    }

    std::vector<float> get_randomized_params(float deviation)
    {
        std::vector<float> vec(search_params::num_of_params());
        for (int i = 0; i < search_params::num_of_params(); i++) {
            int minv, maxv;
            search_params::get_limits(i, minv, maxv);
            float val = get_optimal_value_for_param(i) + (float)(rand() % (maxv - minv + 1))*deviation;

            vec[i] = std::clamp(val, (float)minv, (float)maxv);
        }
        return vec;
    }

private:
    float get_param_strength(int param, float value) {
        int val = static_cast<int>(value + 0.5f);
        int minv, maxv;
        search_params::get_limits(param, minv, maxv);


        float x = std::clamp((int)(remap_value(val, minv, maxv, 0, strength_table_res-1) + 0.5f), 0, strength_table_res-1);
        float frac = x - std::floor(x);

        int idx0 = (int)x;
        int idx1 = std::min((int)x+1, strength_table_res-1);

        return strength_table[param*strength_table_res + idx0] * (1.0f - frac) + strength_table[param*strength_table_res + idx1] * frac;
    }

    std::vector<float> strength_table;
};



struct tuner_worker
{
    tuner_worker(int base_time, int time_inc, int N, std::shared_ptr<std::vector<float>> p, std::shared_ptr<std::atomic<int>> g, std::vector<std::string> ptt, std::shared_ptr<std::vector<std::string>> opening_suite)
    {
        static int worker_count = 0;
        worker_id = worker_count;
        worker_count++;

        games_left = g;
        params = p;
        params_to_tune = ptt;

        openings = opening_suite;

        t = std::thread(&tuner_worker::tuner_thread, this, base_time, time_inc, N);
    }

    ~tuner_worker()
    {
        t.join();
    }

    void tuner_thread(int base_time, int time_inc, int N) {
        std::srand(time(NULL) + worker_id);

        searcher s0;
        searcher s1;

        s0.set_threads(1);
        s1.set_threads(1);

        match_stats stats;

        auto add_delta_vec = [this] (std::vector<float> &vec0, std::vector<float> &vec1, float frac) {
            if (vec0.size() != search_params::num_of_params() || vec1.size() != search_params::num_of_params()) {
                std::cout << "Error: bad sized param vector" << std::endl;
                return;
            }

            for (std::string &param_name : params_to_tune) {
                int i = search_params::get_variable_index(param_name);

                int minv, maxv;
                search_params::get_limits(i, minv, maxv);

                float old_val = vec0[i];
                float new_val = vec0[i] + frac*vec1[i];

                if (std::round(old_val) == std::round(new_val)) {
                    vec0[i] = (frac*vec1[i] > 0 ? old_val + 1.0f : old_val - 1.0f);
                } else {
                    vec0[i] = new_val;
                }
            }
        };



        //tuning_test_model test_model = tuning_test_model::get_test_model();
        //*params = test_model.get_randomized_params(0.25);

        auto play_match = [&] (std::vector<float> &params0, std::vector<float> &params1) {
            for (std::string &param_name : params_to_tune) {
                int i = search_params::get_variable_index(param_name);

                s0.sp.data[i] = std::round(params0[i]);
                s1.sp.data[i] = std::round(params1[i]);
            }

            std::string opening_fen = (*openings)[rand() % openings->size()];

            return play_game_pair(s0, s1, opening_fen, base_time, time_inc, rand()&0x1, stats);

            //return test_model.simulate_match(params0, params1, 0.5, 2);
        };

        auto rand_perturbation_vec = [this] (std::vector<float> &vec) {
            if (vec.size() != search_params::num_of_params()) {
                vec.resize(search_params::num_of_params());
            }
            for (std::string &param_name : params_to_tune) {
                int i = search_params::get_variable_index(param_name);

                int minv, maxv;
                search_params::get_limits(i, minv, maxv);
                vec[i] = ((rand() & 0x1) ? 1.0f : -1.0f)*(maxv-minv);
            }
        };

        std::vector<float> delta;
        std::vector<float> params_plus;
        std::vector<float> params_minus;

        double alpha = 0.602f;
        double gamma = 0.101f;

        double A = N/10;
        double a = 0.01f;
        double c = 0.15f;

        while (*games_left > 0) {

            int k = N - (*games_left);
            rand_perturbation_vec(delta);

            double ak = a / std::pow((double)(k + 1 + A), alpha);
            double ck = c / std::pow((double)(k + 1), gamma);

            params_plus = *params;
            params_minus = *params;

            add_delta_vec(params_plus, delta, ck);
            add_delta_vec(params_minus, delta, -ck);

            float result = play_match(params_plus, params_minus);

            for (std::string &param_name : params_to_tune) {
                int i = search_params::get_variable_index(param_name);

                int minv, maxv;
                search_params::get_limits(i, minv, maxv);
                int range = maxv - minv;
                (*params)[i] += (ak*result*range*range) / (ck*delta[i]);
                (*params)[i] = std::clamp((*params)[i], (float)minv, (float)maxv);
            }

            (*games_left) -= 1;

            /*if ((k % 100) == 0) {
                std::cout << test_model.get_maximum_strength() << " " << test_model.get_strength(*params) << std::endl;
            }*/

        }
    }
    int worker_id;
    std::thread t;

    std::vector<std::string> params_to_tune;
    std::shared_ptr<std::atomic<int>> games_left;
    std::shared_ptr<std::vector<float>> params;
    std::shared_ptr<std::vector<std::string>> openings;
};


int read_tuning_file(std::string tuning_file, std::vector<float> &params)
{
    std::ifstream file(tuning_file);
    if (!file.is_open()) {
        return 0;
    }
    int games_played = 0;
    std::string line;
    while (getline(file, line)) {
        if (line.find(";") != std::string::npos) {
            char varname[64];
            float value;
            std::sscanf(line.c_str(), "%s = %f;", varname, &value);

            int param_index = search_params::get_variable_index(std::string(varname));

            if (param_index >= 0) {
                params[param_index] = value;
            }
        } else {
            char str[64];
            std::sscanf(line.c_str(), "%s %d", str, &games_played);
        }
    }

    std::cout << "Games played: " << games_played << std::endl;
    for (int i = 0; i < search_params::num_of_params(); i++) {
        std::cout << search_params::get_variable_name(i) << " = " << params[i] << std::endl;
    }

    return games_played;

}


void tuning_utility::tune_search_params(int time, int time_inc, int threads, int num_of_games, std::string output_file, const std::vector<std::string> &params_to_tune, std::string opening_suite)
{
    std::shared_ptr<std::vector<std::string>> openings = nullptr;
    if (opening_suite != "") {
        openings = std::make_shared<std::vector<std::string>>(load_opening_suite(opening_suite));
    }

    search_params default_params;

    std::shared_ptr<std::vector<float>> params = std::make_shared<std::vector<float>>();
    std::shared_ptr<std::atomic<int>> games_left = std::make_shared<std::atomic<int>>(num_of_games);
    for (int i = 0; i < default_params.num_of_params(); i++) {
        params->push_back((float)default_params.data[i]);
    }

    *games_left = num_of_games - read_tuning_file(output_file, *params);

    std::vector<std::unique_ptr<tuner_worker>> workers;
    for (int i = 0; i < threads; i++) {
        workers.push_back(std::make_unique<tuner_worker>(time, time_inc, num_of_games, params, games_left, params_to_tune, openings));
    }

    std::chrono::high_resolution_clock::time_point last_save_time = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point last_speed_time = std::chrono::high_resolution_clock::now();

    double games_per_hour = 0.0f;
    int prev_game_left = *games_left;


    while (*games_left > 0) {
        std::chrono::high_resolution_clock::time_point time_now = std::chrono::high_resolution_clock::now();
        int delta_games = prev_game_left - *games_left;
        if (delta_games > 5) {
            double delta_time = (double)std::chrono::duration_cast<std::chrono::milliseconds>(time_now - last_speed_time).count() / 1000.0f;
            if (games_per_hour == 0.0f) {
                games_per_hour = (delta_games / delta_time)*3600;
            } else {
                games_per_hour = 0.9f*games_per_hour + ((delta_games / delta_time)*3600)*0.1f;
            }
            last_speed_time = time_now;
            prev_game_left = *games_left;
        }

        int hours_left = 0;
        int minutes_left = 0;
        if (games_per_hour > 0.0f) {
            double time_left = (*games_left) / games_per_hour;
            hours_left = time_left;
            minutes_left = (time_left - (double)hours_left)*60;
        }

        std::cout << "\rTuning " << num_of_games - *games_left << "/" << num_of_games << "  Speed: " << (int)games_per_hour << " games/hour   Est: " << hours_left << "h " << minutes_left << "min         ";

        if (std::chrono::duration_cast<std::chrono::milliseconds>(time_now - last_save_time).count() > 10000) {
            last_save_time = time_now;

            std::ofstream file("search_tuning.txt", std::ios::app);
            if (file.is_open()) {
                file << "\n\nPlayed: " << num_of_games - *games_left << "\n";

                int num_of_params = sizeof(search_params) / sizeof(int32_t);
                for (int j = 0; j < num_of_params; j++) {
                    file << search_params::get_variable_name(j) << " = " << (*params)[j] << ";\n";
                }
            }
            file.close();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    workers.clear();

    searcher s0, s1;
    for (int i = 0; i < default_params.num_of_params(); i++) {
        s0.sp.data[i] = (int)std::round((*params)[i]);
    }

    int elo = testing_utility::test(1000, time, time_inc, s0, s1, threads, opening_suite);

    std::ofstream file("search_tuning.txt", std::ios::app);
    if (file.is_open()) {
        file << "\n\nFinal " << elo << " Elo\n";
        int num_of_params = sizeof(search_params) / sizeof(int32_t);
        for (int j = 0; j < num_of_params; j++) {
            file << search_params::get_variable_name(j) << " = " << (int)std::round((*params)[j]) << ";\n";
        }
    }
    file.close();
}
