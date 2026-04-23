#include "wdl_model.hpp"
#include <iostream>
#include <array>
#include <filesystem>

#include "pgn_parser.hpp"
#include "misc.hpp"



typedef std::array<double, 8> win_rate_model_params;

typedef std::tuple<int32_t, int32_t, float> wdl_model_data_point;
typedef std::vector<wdl_model_data_point> wdl_model_dataset;



int32_t eval_scaling_factor = 707;

win_rate_model_params win_rate_params =
{0.20502, -0.0473903, -2.1452, 5.05524,
0.229833, -0.761037, 0.839419, 0.423668};



double get_material(const board_state &state)
{
    int material = 0;

    material += pop_count(state.bitboards[PAWN][0] | state.bitboards[PAWN][1]);
    material += pop_count(state.bitboards[KNIGHT][0] | state.bitboards[KNIGHT][1])*3;
    material += pop_count(state.bitboards[BISHOP][0] | state.bitboards[BISHOP][1])*3;
    material += pop_count(state.bitboards[ROOK][0] | state.bitboards[ROOK][1])*5;
    material += pop_count(state.bitboards[QUEEN][0] | state.bitboards[QUEEN][1])*9;

    return material;
}


double _sigmoid(double x)
{
    if (x > 100.0f) {
        return 1.0f;
    } else if (x < -100.0f) {
        return 0.0f;
    }

    double ex = std::exp(x);

    return (ex/(1.0+ex));
}

double _dsigmoid(double x)
{
    return _sigmoid(x)*(1.0f - _sigmoid(x));
}

std::pair<double, double> win_rate_coefficients(int32_t material, win_rate_model_params &params)
{
    double m = static_cast<double>(std::clamp(material, 18, 78)) / 39.0;

    double m3 = m*m*m;
    double m2 = m*m;

    double a = m3*params[0] + m2*params[1] + m*params[2] + params[3];
    double b = m3*params[4] + m2*params[5] + m*params[6] + params[7];

    return {a, b};
}


float win_rate_model(int32_t material, int32_t cp_eval, win_rate_model_params &params)
{
    double eval = ((double)cp_eval / 100.0f);

    auto [a, b] = win_rate_coefficients(material, params);

    double x = (eval - a) / b;

    return _sigmoid(x);
}


void win_rate_model_backprop(int32_t material, int32_t cp_eval, win_rate_model_params &params, win_rate_model_params &gradient, float error)
{
    double eval = ((double)cp_eval / 100.0f);

    double m = static_cast<double>(std::clamp(material, 18, 78)) / 39.0;

    double m3 = m*m*m;
    double m2 = m*m;

    double a = m3*params[0] + m2*params[1] + m*params[2] + params[3];
    double b = m3*params[4] + m2*params[5] + m*params[6] + params[7];

    double x = (eval - a) / b;

    double ds = _dsigmoid(x)*error;

    double da = -(1 / b) * ds;
    double db = (a/(b*b) - (eval / (b*b))) * ds;

    gradient[0] += da*m3; gradient[1] += da*m2; gradient[2] += da*m; gradient[3] += da;
    gradient[4] += db*m3; gradient[5] += db*m2; gradient[6] += db*m; gradient[7] += db;
}


double log_loss(double truth, double prediction)
{
    prediction = std::clamp(prediction, 0.01, 0.99);

    return -(truth*std::log(prediction) + (1.0 - truth)*std::log(1.0f - prediction));
}

double log_loss_derivate(double truth, double prediction)
{
    prediction = std::clamp(prediction, 0.01, 0.99);

    return ((-truth)/prediction + (1.0-truth)/(1.0-prediction));
}


std::pair<wdl_model_dataset, wdl_model_dataset> load_wdl_model_dataset(std::string dataset_folder, int valid_size)
{
    wdl_model_dataset data_train;
    wdl_model_dataset data_valid;

    srand(123);

    std::vector<selfplay_result> results;

    for (const auto& entry : std::filesystem::directory_iterator(dataset_folder)) {
        if (entry.path().extension().string() == ".txt") {
            std::vector<selfplay_result> file_results;
            load_selfplay_results(file_results, entry.path().string());

            for (auto &r : file_results) {
                results.push_back(r);
            }
        }
    }

    board_state state;

    for (int i = 0; i < valid_size; i++) {
        int r = rand() % results.size();

        state.load_fen(results[r].fen);

        float truth = (((results[r].wdl > 0.75 && state.get_turn() == WHITE) || (results[r].wdl < 0.25 && state.get_turn() == BLACK)) ? 1.0f : 0.0f);

        data_valid.push_back({get_material(state), results[r].eval, truth});

        std::swap(results[results.size()-1], results[r]);
        results.pop_back();
    }

    for (const selfplay_result &spr : results) {
        state.load_fen(spr.fen);

        float truth = (((spr.wdl > 0.75 && state.get_turn() == WHITE) || (spr.wdl < 0.25 && state.get_turn() == BLACK)) ? 1.0f : 0.0f);

        data_train.push_back({get_material(state), spr.eval, truth});
    }

    return {data_train, data_valid};
}


void wdl_model::fit_model(std::string dataset_folder)
{
    constexpr int data_valid_size = 10000;

    auto [data_train, data_valid] = load_wdl_model_dataset(dataset_folder, data_valid_size);

    std::cout << "Train set size: " << data_train.size() << std::endl;
    std::cout << "Valid set size: " << data_valid.size() << std::endl;

    win_rate_model_params params =
                                {0.0, 0.0, 0.0, 1.0,
                                 0.0, 0.0, 0.0, 1.0};


    win_rate_model_params best_params;

    double best_cost = std::numeric_limits<double>::max();
    int best_params_iteration = 0;

    srand(time(NULL));

    for (int i = 0; i < 8; i++) {
        params[i] += (static_cast<float>(rand() & 0xFFFF) / 0xFFFF) * 0.2 - 0.1f;
    }

    double learning_rate = 0.0001f;

    int iterations = 0;
    while (true) {
        for (auto [material, eval, truth] : data_train) {
            if (is_mate_score(eval)) {
                continue;
            }

            win_rate_model_params gradient = {0, 0, 0, 0,
                                              0, 0, 0, 0};

            double prediction = win_rate_model(material, eval, params);

            win_rate_model_backprop(material, eval, params, gradient, log_loss_derivate(truth, prediction));

            for (int i = 0; i < 8 && iterations > 0; i++) {
                params[i] -= learning_rate*gradient[i];
            }
        }


        for (int i = data_train.size() - 1; i > 0; --i) {
            uint32_t r = (rand() << 16) | (rand() & 0xFFFF);
            int j = r % (i+1);

            std::swap(data_train[i], data_train[j]);
        }


        double loss = 0;
        for (auto [material, eval, truth] : data_valid) {
            if (is_mate_score(eval)) {
                continue;
            }

            double prediction = win_rate_model(material, eval, params);

            loss += log_loss(truth, prediction);
        }

        double cost = loss / data_valid.size();


        if (cost < best_cost) {
            best_cost = cost;
            best_params = params;
            best_params_iteration = iterations;

            //std::cout << "New best" << std::endl;
        } else {

            if (std::abs(best_params_iteration - iterations) > 5) {
                params = best_params;
                learning_rate = learning_rate*0.25f;

                //std::cout << "Reducing learning rate" << std::endl;
            }

            if (std::abs(best_params_iteration - iterations) > 20) {
                break;
            }
        }

        std::cout << "Iteration: " << iterations << "  cost: " << cost << std::endl;

        iterations += 1;
    }

    std::cout << "{" << best_params[0] << ", " << best_params[1] << ", " << best_params[2] << ", " << best_params[3] << "," << std::endl;
    std::cout        << best_params[4] << ", " << best_params[5] << ", " << best_params[6] << ", " << best_params[7] << "};" << std::endl;
}


void wdl_model::get_wdl(const board_state &state, int32_t search_score, float &win_p, float &draw_p, float &loss_p)
{
    if (is_mate_score(search_score)) {
        draw_p = 0;
        if (search_score > 0) {
            win_p = 1.0f;
            loss_p = 0.0f;
        } else {
            loss_p = 1.0f;
            win_p = 0.0f;
        }
    }

    int32_t material = get_material(state);

    win_p = win_rate_model(material, search_score, win_rate_params);
    loss_p = win_rate_model(material, -search_score, win_rate_params);
    draw_p = 1.0f - win_p - loss_p;
}


int32_t wdl_model::normalize_score(const board_state &state, int32_t search_score)
{
    if (!is_mate_score(search_score)) {
        return (search_score * 400) / eval_scaling_factor;
    } else {
        return search_score;
    }
}







