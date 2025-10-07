#include "datagen.hpp"
#include "../search.hpp"
#include "../game.hpp"
#include <memory>
#include <thread>
#include <fstream>
#include <cmath>
#include <atomic>
#include <iomanip>

#include "misc.hpp"


struct datagen_worker
{
    datagen_worker()
    {
        static int worker_counter = 0;
        worker_id = worker_counter;
        worker_counter += 1;
        avg_depth = 0.0f;

        avg_start_pos_abs_eval = 0.0f;
        avg_opening_branching_factor = 0.0f;

        bot = std::make_unique<alphabeta_search>();
    }

    void wait()
    {
        bot->abort_fast_search();
        t.join();
    }

    void start(std::atomic<int> &games, int depth, int nodes, bool random, std::shared_ptr<nnue_weights> weights)
    {
        t = std::thread(&datagen_worker::play, this, std::ref(games), depth, nodes, random, weights);
    }

    std::vector<selfplay_result> results;
    double avg_depth;
    double avg_start_pos_abs_eval;
    double avg_opening_branching_factor;
private:
    void play(std::atomic<int> &games, int d, int n, bool random, std::shared_ptr<nnue_weights> weights);

    std::unique_ptr<alphabeta_search> bot;

    std::thread t;
    int worker_id;
};


void datagen_worker::play(std::atomic<int> &games, int d, int n, bool random, std::shared_ptr<nnue_weights> weights)
{
    std::srand((size_t)time(NULL) + worker_id);

    bot->set_shared_weights(weights);


    game_state game;

    int num_of_first_moves = 0;
    double first_move_abs_eval_sum = 0;

    long total_opening_moves = 0;
    long played_opening_moves = 0;

    while (games > 0) {
        game.reset();
        bot->new_game();

        std::vector<position_evaluation> game_positions;

        game_win_type_t win_status = NO_WIN;
        game_win_type_t mate_found_win = NO_WIN;

        int random_moves = (random ? 8 : 14);

        double total_depth = 0;
        int searched_moves = 0;
        int moves = 0;

        while (true) {
            if (random_moves > 0) {
                std::vector<chess_move> acceptable_moves;
                if (random) {
                    acceptable_moves = game.get_state().get_all_legal_moves(game.get_state().get_turn());
                } else {
                    constexpr int multipv = 5;
                    constexpr int max_cp_loss = 120;

                    if (bot->get_multi_pv() != multipv) {
                        bot->set_multi_pv(multipv);
                    }
                    bot->fast_search(game.get_state(), d/2, n);

                    int32_t best_score = bot->get_evaluation();

                    for (int i = 0; i < multipv; i++) {
                        pv_table pv;
                        bot->get_pv(pv, i);

                        if (pv.num_of_moves > 0 && std::abs(pv.score - best_score) < max_cp_loss) {
                            acceptable_moves.push_back(pv.moves[0]);
                        }
                    }
                }

                if (acceptable_moves.size() == 0) {
                    std::cout << "Error: no legal moves" << std::endl;
                    win_status = DRAW;
                } else {
                    win_status = game.make_move(acceptable_moves[std::rand() % acceptable_moves.size()]);
                    random_moves -= 1;

                    played_opening_moves += 1;
                    total_opening_moves += acceptable_moves.size();
                }

            } else {
                if (bot->get_multi_pv() != 1) {
                    bot->set_multi_pv(1);
                }


                chess_move mov = chess_move::null_move();
                int32_t score;
                int depth;

                mov = bot->fast_search(game.get_state(), d, n);
                score = bot->get_evaluation();
                depth = bot->get_depth();

                if (is_mate_score(score) && mate_found_win == NO_WIN) {
                    if ((game.get_state().get_turn() == BLACK && score < 0) ||
                        (game.get_state().get_turn() == WHITE && score > 0)) {
                        mate_found_win = WHITE_WIN;
                    } else {
                        mate_found_win = BLACK_WIN;
                    }
                }

                if (depth > 1 && !is_mate_score(score)) {
                    game_positions.push_back(position_evaluation(game.get_state(), score, mov));
                }

                if (mov == chess_move::null_move()) {
                    std::cout << "Engine produced nullmove. depth = " << depth << "  score = " << score << "  move = " << mov.to_uci() << std::endl;
                    std::cout << "Position: " << game.get_state().generate_fen() << std::endl;
                }

                if (searched_moves == 0 && !is_mate_score(score)) {
                    first_move_abs_eval_sum += (double)std::abs(score);
                    num_of_first_moves += 1;
                }

                searched_moves += 1;
                total_depth += depth;

                win_status = game.make_move(mov);

                if (mate_found_win != NO_WIN && win_status == NO_WIN) {
                    win_status = mate_found_win;
                }
            }
            moves++;

            if (moves > 512) {
                win_status = DRAW;
            }
            if (win_status != NO_WIN) {
                //std::cout << "Worker " << worker_id << "  Game number: " << games << "  Result: " << win_type_to_str(win_status) << std::endl;
                break;
            }
            if (games <= 0) {
                return;
            }
        }

        if (searched_moves > 0) {
            avg_depth = total_depth / searched_moves;
            avg_start_pos_abs_eval = first_move_abs_eval_sum / num_of_first_moves;
            avg_opening_branching_factor = (double)total_opening_moves / played_opening_moves;
        }

        games--;

        for (auto it = game_positions.begin(); it != game_positions.end(); it++) {
            results.push_back(selfplay_result(*it, win_status));
        }
    }
}


void training_datagen::datagen(std::string output_file, std::string nnue_file, int threads, int games, int depth, int nodes, bool random_opening)
{
    std::cout << std::endl << "Selfplaying\nthreads = " << threads
                           << "\nmin depth = " << depth
                           << "\nmin nodes = " << nodes
                           << "\nrandom opening = " << random_opening
                           << "\noutput = " << output_file
                           << "\nnnue = " << (nnue_file != "" ? nnue_file : "default") << std::endl;

    std::shared_ptr<nnue_weights> weights = std::make_shared<nnue_weights>();

    if (nnue_file != "") {
        std::cout << "Loading shared nnue weights: " << nnue_file << std::endl;
        weights->load(nnue_file);
    }


    std::vector<datagen_worker> workers(threads);

    std::vector<selfplay_result> results;

    std::atomic<int> agames = games;

    for (int i = 0; i < threads; i++) {
        workers[i].start(agames, depth, nodes, random_opening, weights);
    }

    int prev_positions = 0;

    int prev_games_left = agames;
    std::chrono::time_point prev_time_point = std::chrono::high_resolution_clock::now();

    double games_per_hour = 0.0f;
    double positions_per_second = 0.0f;

    double smoothing_frac = 0.9f;

    double first_move_abs_eval = 0;
    double opening_branching_factor = 0;

    while (agames > 0) {

        if (prev_games_left - 10 > agames) {
            std::chrono::time_point t0 = std::chrono::high_resolution_clock::now();

            uint64_t dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>( t0 - prev_time_point ).count();

            double dt_h = (((double)dt_ms / 1000) / 60 / 60);

            if (dt_ms > 0 && dt_h > 0.0f) {
                int positions = 0;
                first_move_abs_eval = 0;
                opening_branching_factor = 0;
                for (int i = 0; i < threads; i++) {
                    positions += workers[i].results.size();
                    first_move_abs_eval += workers[i].avg_start_pos_abs_eval / threads;
                    opening_branching_factor += workers[i].avg_opening_branching_factor / threads;
                }

                int delta_positions = positions - prev_positions;

                double pps = (double)delta_positions / ((double)dt_ms / 1000);
                double gph = (double)(prev_games_left - agames) / dt_h;

                if (games_per_hour == 0.0f) {
                    games_per_hour = gph;
                    positions_per_second = pps;
                } else {
                    positions_per_second = positions_per_second * smoothing_frac + pps * (1.0f - smoothing_frac);
                    games_per_hour       = games_per_hour       * smoothing_frac + gph * (1.0f - smoothing_frac);
                }

                prev_time_point = t0;
                prev_games_left = agames;
                prev_positions = positions;
            }
        }

        double avg_depth = 0;
        for (size_t i = 0; i < workers.size(); i++) {
            avg_depth += workers[i].avg_depth / workers.size();
        }

        int hours_left = 0;
        int minutes_left = 0;
        if (games_per_hour > 0.0f) {
            double time_left = agames / games_per_hour;
            hours_left = time_left;
            minutes_left = (time_left - (double)hours_left)*60;
        }

        std::cout << "\r"
                  << (games - agames) << "/" << games << "  "
                  << std::setw(6) << (int)games_per_hour << " games/h  "
                  << std::setw(5) << (int)positions_per_second << " pos/s   D: "
                  << std::fixed << std::setprecision(2) << avg_depth << "   E: "
                  << std::setprecision(2) << first_move_abs_eval / 100 << "   BF: "
                  << std::setprecision(2) << opening_branching_factor
                  << "  est: " << hours_left << "h " << minutes_left << "min     ";

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << std::endl;

    for (int i = 0; i < threads; i++) {
        workers[i].wait();
        results.insert(results.end(), workers[i].results.begin(), workers[i].results.end());
    }

    std::cout << "Total positions " << results.size() << std::endl;

    save_selfplay_results(results, output_file);
}
