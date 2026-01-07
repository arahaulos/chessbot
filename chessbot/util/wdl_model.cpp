#include "wdl_model.hpp"
#include <iostream>
#include <filesystem>
#include <thread>
#include <mutex>
#include <queue>
#include <chrono>
#include <atomic>
#include <array>

#include "../search.hpp"
#include "pgn_parser.hpp"
#include "misc.hpp"


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


float win_rate_model(const board_state &state, int32_t cp_eval, win_rate_model_params &params)
{
    double eval = ((double)cp_eval / 100.0f);

    double m = get_material(state) / 78.0f;

    double m3 = m*m*m;
    double m2 = m*m;

    double a = m3*params[0] + m2*params[1] + m*params[2] + params[3];
    double b = m3*params[4] + m2*params[5] + m*params[6] + params[7];

    double x = (eval - a) / b;

    return _sigmoid(x);
}


void win_rate_model_backprop(const board_state &state, int32_t cp_eval, win_rate_model_params &params, win_rate_model_params &gradient, float error)
{
    double eval = ((double)cp_eval / 100.0f);

    double m = get_material(state) / 78.0f;

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

void wdl_model::fit_model(std::string dataset_file)
{
    constexpr int data_valid_size = 5000;

    std::vector<selfplay_result> data_train;
    std::vector<selfplay_result> data_valid;

    load_selfplay_results(data_train, dataset_file);

    for (int i = 0; i < data_valid_size; i++) {
        int r = rand() % data_train.size();

        data_valid.push_back(data_train[r]);

        std::swap(data_train[data_train.size()-1], data_train[r]);
        data_train.pop_back();
    }

    std::cout << "Train set size: " << data_train.size() << std::endl;
    std::cout << "Valid set size: " << data_valid.size() << std::endl;



    win_rate_model_params params =
{0.882293, -1.04925, -2.83103, 4.74646,
-0.29113, 0.0309023, 0.59633, 0.606723};

    double learning_rate = 0.000001f;

    int iterations = 0;
    while (true) {
        board_state state;
        for (selfplay_result &d : data_train) {
            if (is_mate_score(d.eval)) {
                continue;
            }

            state.load_fen(d.fen);
            double prediction = win_rate_model(state, d.eval, params);
            double truth = (((d.wdl > 0.75 && state.get_turn() == WHITE) || (d.wdl < 0.25 && state.get_turn() == BLACK)) ? 1.0f : 0.0f);

            win_rate_model_params gradient = {0, 0, 0, 0,
                                              0, 0, 0, 0};

            win_rate_model_backprop(state, d.eval, params, gradient, log_loss_derivate(truth, prediction));

            for (int i = 0; i < 8 && iterations > 0; i++) {
                params[i] -= learning_rate*gradient[i];
            }
        }

        for (int i = data_train.size() - 1; i > 0; --i) {
            uint32_t r = (rand() << 16) | (rand() & 0xFFFF);
            int j = r % (i+1);

            std::swap(data_train[i], data_train[j]);
        }

        if ((iterations % 10) == 0) {
            double loss = 0;
            for (selfplay_result &d : data_valid) {
                state.load_fen(d.fen);
                double prediction = win_rate_model(state, d.eval, params);
                double truth = (((d.wdl > 0.75 && state.get_turn() == WHITE) || (d.wdl < 0.25 && state.get_turn() == BLACK)) ? 1.0f : 0.0f);

                loss += log_loss(truth, prediction);
            }

            std::cout << "{" << params[0] << ", " << params[1] << ", " << params[2] << ", " << params[3] << "," << std::endl;
            std::cout        << params[4] << ", " << params[5] << ", " << params[6] << ", " << params[7] << "};" << std::endl;

            double cost = loss / data_valid.size();
            std::cout << cost << std::endl;
        }
        iterations += 1;
    }
}


void wdl_model::get_wdl(const board_state &state, int32_t search_score, float &win_p, float &draw_p, float &loss_p)
{
    win_rate_model_params params =
{0.913711, -1.05117, -2.87007, 4.75986,
-0.284183, 0.0281383, 0.589859, 0.608301};


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

    win_p = win_rate_model(state, search_score, params);
    loss_p = win_rate_model(state, -search_score, params);
    draw_p = 1.0f - win_p - loss_p;
}









