#include "tuning.hpp"
#include "search.hpp"
#include "game.hpp"
#include <memory>
#include <thread>
#include <fstream>
#include <cmath>
#include <atomic>
#include <iomanip>
#include "uci.hpp"

#include "time_managment.hpp"


bool save_selfplay_results(std::vector<selfplay_result> &results, std::string filename)
{
    std::cout << "Saving positions: " << filename << "... ";
    std::ofstream file(filename);
    if (file.is_open()) {
        for (auto it = results.begin(); it != results.end(); it++) {
            file << it->fen << ";" << it->bm.to_uci() << ";" << it->eval << ";" << it->wdl << std::endl;
            //fen;bestmove;score;wdl
        }
    } else {
        std::cout << "error: unable to open file!" << std::endl;
        return false;
    }
    file.close();
    std::cout << "done. " << results.size() << " positions saved!" << std::endl;
    return true;
}

bool load_selfplay_results(std::vector<selfplay_result> &results, std::string filename)
{
    int total = 0;

    std::cout << "Loading positions: " << filename << "... ";
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {

            //fen;bestmove;score;wdl

            std::string parts[8];
            int num_of_parts = 0;
            size_t p = line.find(';');
            while (p != std::string::npos && num_of_parts < 7) {
                parts[num_of_parts++] = line.substr(0, p);
                line.erase(0, p + 1);
                p = line.find(';');
            }
            parts[num_of_parts++] = line;

            if (num_of_parts == 2) {
                results.emplace_back(parts[0], chess_move::null_move().to_uci(), 0, atof(parts[1].c_str()));
                total += 1;
            } else if (num_of_parts == 4) {
                results.emplace_back(parts[0], parts[1], atoi(parts[2].c_str()), atof(parts[3].c_str()));
                total += 1;
            }
        }
    } else {
        std::cout << "error: unable to open file!" << std::endl;
        return false;
    }
    file.close();
    std::cout << "done. " << total << " positions loaded!" << std::endl;
    return true;
}


struct selfplay_worker
{
    selfplay_worker()
    {
        static int worker_counter = 0;
        worker_id = worker_counter;
        worker_counter += 1;
        avg_depth = 0.0f;
    }

    void wait()
    {
        t.join();
    }

    void start(std::atomic<int> &games, int depth, int nodes, bool random, std::shared_ptr<nnue_weights> weights)
    {
        t = std::thread(&selfplay_worker::play, this, std::ref(games), depth, nodes, random, weights);
    }

    std::vector<selfplay_result> results;
    double avg_depth;

private:
    void play(std::atomic<int> &games, int d, int n, bool random, std::shared_ptr<nnue_weights> weights);

    std::thread t;
    int worker_id;
};


void selfplay_worker::play(std::atomic<int> &games, int d, int n, bool random, std::shared_ptr<nnue_weights> weights)
{
    std::srand((size_t)time(NULL) + worker_id);

    std::unique_ptr<alphabeta_search> bot = std::make_unique<alphabeta_search>();

    bot->use_opening_book = !random;
    bot->set_shared_weights(weights);


    game_state game;

    while (games > 0) {
        game.reset();
        bot->new_game();

        std::vector<position_evaluation> game_positions;

        game_win_type_t win_status = NO_WIN;
        game_win_type_t mate_found_win = NO_WIN;

        int random_moves = (random ? 6 + (std::rand() % 8) : 0);

        double total_depth = 0;
        int searched_moves = 0;
        int moves = 0;

        while (true) {
            if (random_moves > 0) {
                std::vector<chess_move> legal_moves = game.get_state().get_all_legal_moves(game.get_state().get_turn());

                if (legal_moves.size() == 0) {
                    std::cout << "Error: no legal moves" << std::endl;
                    win_status = DRAW;
                } else {
                    win_status = game.make_move(legal_moves[std::rand() % legal_moves.size()]);
                    random_moves -= 1;
                }
            } else {
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
        }

        if (searched_moves > 0) {
            avg_depth = total_depth / searched_moves;
        }

        games--;

        for (auto it = game_positions.begin(); it != game_positions.end(); it++) {
            results.push_back(selfplay_result(*it, win_status));
        }
    }
}


void tuning_utility::selfplay(std::string output_file, std::string nnue_file, int threads, int games, int depth, int nodes, bool random_opening)
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


    std::vector<selfplay_worker> workers(threads);

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

    while (agames > 0) {

        if (prev_games_left - 10 > agames) {
            std::chrono::time_point t0 = std::chrono::high_resolution_clock::now();

            uint64_t dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>( t0 - prev_time_point ).count();

            double dt_h = (((double)dt_ms / 1000) / 60 / 60);

            if (dt_ms > 0 && dt_h > 0.0f) {
                int positions = 0;
                for (int i = 0; i < threads; i++) {
                    positions += workers[i].results.size();
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

        std::cout << "\rPlaying " << std::setw(7) << (games - agames) << "/" << games << "  "
                  << ((games - agames)*100)/games << "%  "
                  << std::setw(6) << (int)games_per_hour << " games/hour  "
                  << std::setw(5) << (int)positions_per_second << " moves/sec   Avg depth: "
                  << std::fixed << std::setprecision(2) << avg_depth
                  << "  time left: " << hours_left << "h " << minutes_left << "min     ";

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




struct test_worker
{
    test_worker()
    {
        static int worker_counter = 0;
        worker_id = worker_counter;
        worker_counter += 1;

        wins = 0;
        losses = 0;
        draws = 0;

        bot0_depth = 0;
        bot1_depth = 0;

        bot0_moves = 0;
        bot1_moves = 0;

    }

    void wait()
    {
        t.join();
    }

    void start(std::atomic<int> &games, int time, int time_inc, bool start_with_black, alphabeta_search &bot0, alphabeta_search &bot1, std::string uci_engine_path)
    {
        t = std::thread(&test_worker::play, this, std::ref(games), time, time_inc, start_with_black, std::ref(bot0), std::ref(bot1), uci_engine_path);
    }

    int wins;
    int losses;
    int draws;

    int bot0_depth;
    int bot1_depth;

    int64_t bot0_nps;
    int64_t bot1_nps;

    int bot0_moves;
    int bot1_moves;
private:
    void play(std::atomic<int> &games, int time, int time_inc, bool start_with_black, alphabeta_search &bot00, alphabeta_search &bot10, std::string uci_engine_path);

    std::thread t;
    int worker_id;
};



void test_worker::play(std::atomic<int> &games, int base_time, int time_inc, bool start_with_black, alphabeta_search &bot00, alphabeta_search &bot10, std::string uci_engine_path)
{
    std::srand((size_t)time(NULL) + worker_id);

    wins = 0;
    losses = 0;
    draws = 0;

    bot0_depth = 0;
    bot1_depth = 0;

    bot0_nps = 0;
    bot1_nps = 0;

    bot0_moves = 0;
    bot1_moves = 0;

    std::unique_ptr<alphabeta_search> bot0 = std::make_unique<alphabeta_search>(bot00);
    std::unique_ptr<alphabeta_search> bot1 = nullptr;

    bool bot0_playing_black = (start_with_black ? true : false);

    uci_engine engine;
    bool use_uci_engine = (uci_engine_path.length() > 0);
    if (use_uci_engine) {
        engine.launch(uci_engine_path);
        engine.set_option("Threads", 1);
        engine.set_option("Hash", 64);
    } else {
        bot1 = std::make_unique<alphabeta_search>(bot10);
    }



    std::vector<chess_move> opening_moves;
    bool copy_same_opening = false;

    opening_book book;

    while (games > 0) {
        game_state game;
        game.reset();
        game.get_state().set_initial_state();


        if (copy_same_opening) {
            for (size_t i = 0; i < opening_moves.size(); i++) {
                game.make_move(opening_moves[i]);
            }
        } else {
            opening_moves.clear();
            chess_move mov = book.get_book_move(game.get_state());
            while (mov != chess_move::null_move()) {
                game.make_move(mov);
                opening_moves.push_back(mov);

                mov = book.get_book_move(game.get_state());
            }
        }

        bot0->new_game();
        if (use_uci_engine) {
            engine.new_game();
        } else {
            bot1->new_game();
        }

        int bot0_timeleft = base_time;
        int bot1_timeleft = base_time;

        while (true) {
            chess_move mov;
            if ((bot0_playing_black && game.get_turn() == BLACK) || (!bot0_playing_black && game.get_turn() == WHITE)) {
                int target_time = get_target_search_time(bot0_timeleft, time_inc);
                bot0_timeleft = bot0_timeleft - target_time + time_inc;
                mov = bot0->search_time(game.get_state(), target_time);
                //mov = bot0->fast_search(game.get_state(), 7, 0);

                int d = bot0->get_depth();

                if (d > 1) {
                    bot0_depth += d;
                    bot0_nps += bot0->get_node_count() / (target_time+1);
                    bot0_moves += 1;
                }
            } else {
                if (use_uci_engine) {
                    int wtime, btime;
                    if (bot0_playing_black) {
                        wtime = bot1_timeleft;
                        btime = bot0_timeleft;
                    } else {
                        wtime = bot0_timeleft;
                        btime = bot1_timeleft;
                    }
                    int d, n, t;
                    mov = engine.go_clock(game, wtime, time_inc, btime, time_inc, d, n, t);

                    bot1_timeleft = std::max(10, bot1_timeleft - t) + time_inc;

                    if (d > 1) {
                        bot1_depth += d;
                        bot1_nps += n / (t+1);
                        bot1_moves += 1;
                    }
                } else {
                    int target_time = get_target_search_time(bot1_timeleft, time_inc);
                    bot1_timeleft = bot1_timeleft - target_time + time_inc;
                    mov = bot1->search_time(game.get_state(), target_time);
                    //mov = bot1->fast_search(game.get_state(), 7, 0);

                    int d = bot1->get_depth();
                    if (d > 1) {
                        bot1_depth += d;
                        bot1_nps += bot1->get_node_count() / (target_time+1);
                        bot1_moves += 1;
                    }
                }
            }

            game_win_type_t wstatus = game.make_move(mov);
            if (wstatus != NO_WIN) {
                if (wstatus == DRAW) {
                    draws++;
                } else if ((wstatus == BLACK_WIN && bot0_playing_black) || (wstatus == WHITE_WIN && !bot0_playing_black)) {
                    wins++;
                } else {
                    losses++;
                }
                break;
            }
            if (game.get_last_error() != NO_ERROR && use_uci_engine) {
                std::ofstream file("log.txt");
                if (file.is_open()) {
                    file << engine.comm_log.str();
                }
                file.close();

                return;
            }

            if (games <= 0) {
                return;
            }
        }
        bot0_playing_black = !bot0_playing_black;
        copy_same_opening = !copy_same_opening;

        games--;
    }
}


double score_to_elo(double score)
{
    return -std::log10(1.0/score-1.0)*400.0;
}

double elo_to_score(double elo)
{
   return 1.0f / (std::pow(10.0f, -elo/400.0f)+1.0f);
}


double sprt_llr(double elo0, double elo1, int wins, int draws, int losses)
{
    if (wins == 0 || draws == 0 || losses == 0) {
        return 0.0f;
    }

    int n = wins + draws + losses;

    int results[3] = {wins, draws, losses};

    double dr = (double)draws / n;

    double s0 = elo_to_score(elo0);
    double s1 = elo_to_score(elo1);

    double p0[3] = {s0 - 0.5*dr, dr, 1.0f - s0 - 0.5*dr};
    double p1[3] = {s1 - 0.5*dr, dr, 1.0f - s1 - 0.5*dr};

    double llr = 0;
    for (int i = 0; i < 3; i++) {
        llr += std::log(p1[i] / p0[i]) * results[i];
    }
    return llr;
}

void get_test_workers_results(std::vector<test_worker> &workers, double &bot0_depth, double &bot1_depth, double &bot0_nps, double &bot1_nps, double &elo, int &wins, int &draws, int &losses)
{
    int bot0_moves = 0;
    int bot1_moves = 0;

    bot0_depth = 0;
    bot1_depth = 0;

    wins = 0;
    draws = 0;
    losses = 0;

    for (size_t i = 0; i < workers.size(); i++) {
        wins += workers[i].wins;
        losses += workers[i].losses;
        draws += workers[i].draws;

        bot0_depth += workers[i].bot0_depth;
        bot1_depth += workers[i].bot1_depth;

        bot0_moves += workers[i].bot0_moves;
        bot1_moves += workers[i].bot1_moves;

        bot0_nps += workers[i].bot0_nps;
        bot1_nps += workers[i].bot1_nps;
    }

    if (bot0_moves > 0 && bot1_moves > 0) {
        bot0_depth /= bot0_moves;
        bot1_depth /= bot1_moves;

        bot0_nps /= bot0_moves;
        bot1_nps /= bot1_moves;
    } else {
        bot0_depth = 0;
        bot1_depth = 0;

        bot0_nps = 0;
        bot1_nps = 0;
    }

    if (wins+losses+draws > 1) {
        elo = score_to_elo(0.5f * (double)(2*wins + draws) / (wins+losses+draws));
    } else {
        elo = 0.0f;
    }
}



int tuning_utility::test(int games, int time, int time_inc, alphabeta_search &bot0, alphabeta_search &bot1, int threads, std::string uci_engine, double elo0, double elo1, double alpha, double beta)
{
    std::atomic<int> agames = games;
    std::vector<test_worker> workers(threads);

    for (size_t i = 0; i < workers.size(); i++) {
        workers[i].start(agames, time, time_inc, (i < workers.size()/2), bot0, bot1, uci_engine);
    }

    double bot0_depth, bot1_depth, bot0_nps, bot1_nps, elo;
    int wins, draws, losses;

    double lower = std::log(beta / (1 - alpha));
    double upper = std::log((1 - beta) / alpha);

    std::cout << std::endl;
    std::cout << "Elo0: " << elo0 << std::endl;
    std::cout << "Elo1: " << elo1 << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "Beta: " << beta << std::endl;
    std::cout << "LLR upper limit: " << upper << std::endl;
    std::cout << "LLR lower limit: " << lower << std::endl;
    std::cout << "Playing " << games << " games with " << (float)time/1000.0f << "s + " << time_inc << "ms" << " time control.  Threads: " << threads << std::endl;

    while (agames > 0) {
        get_test_workers_results(workers, bot0_depth, bot1_depth, bot0_nps, bot1_nps, elo, wins, draws, losses);

        double llr = sprt_llr(elo0, elo1, wins, draws, losses);

        std::cout << "\rD: "    << std::left << std::setw(8) << bot0_depth << " " << std::setw(8) << bot1_depth
                  << "  Knps: " << std::setw(8) << bot0_nps << " " << std::setw(8) << bot1_nps
                  << "  G: "    << games - agames << "/" << games
                  << "  R: "    << wins << "/" << draws << "/" << losses
                  << "  Elo: "  << elo
                  << "  LLR: "  << llr << "      ";


        if (llr < lower || llr > upper) {
            std::cout << "\nSPRT terminated!" << std::endl;
            agames = 0;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    for (size_t i = 0; i < workers.size(); i++) {
        workers[i].wait();
    }

    get_test_workers_results(workers, bot0_depth, bot1_depth, bot0_nps, bot1_nps, elo, wins, draws, losses);

    int total = wins + draws + losses;

    std::cout << std::endl << std::endl;

    std::cout << "Wins: " << wins << "(" << (wins*100) / total << "%)" << std::endl;
    std::cout << "Draws: " << draws << "(" << (draws*100) / total << "%)" << std::endl;
    std::cout << "Losses: " << losses << "(" << (losses*100) / total << "%)" << std::endl;

    std::cout << "Elo: " << elo << std::endl;

    std::cout << "Depth0: " << bot0_depth << std::endl;
    std::cout << "Depth1: " << bot1_depth << std::endl;
    std::cout << "Knps0: " << bot0_nps << std::endl;
    std::cout << "Knps1: " << bot1_nps << std::endl;

    return elo;
}


int play_game_pair(alphabeta_search &bot0, alphabeta_search &bot1, opening_book &book, int base_time, int time_inc)
{
    std::vector<chess_move> opening_moves;

    game_state game;
    chess_move mov = book.get_book_move(game.get_state());
    while (mov != chess_move::null_move()) {
        game.make_move(mov);
        opening_moves.push_back(mov);

        mov = book.get_book_move(game.get_state());
    }

    bool bot0_playing_black = std::rand() & 0x1;

    int bot0_time_left = base_time;
    int bot1_time_left = base_time;

    int score = 0;

    for (int i = 0; i < 2; i++) {

        bot0.new_game();
        bot1.new_game();

        while (true) {
            chess_move mov;
            if ((bot0_playing_black && game.get_turn() == BLACK) || (!bot0_playing_black && game.get_turn() == WHITE)) {
                int search_time = get_target_search_time(bot0_time_left, time_inc);
                mov = bot0.search_time(game.get_state(), search_time);
                bot0_time_left = std::max(bot0_time_left - search_time, 0) + time_inc;
            } else {
                int search_time = get_target_search_time(bot1_time_left, time_inc);
                mov = bot1.search_time(game.get_state(), search_time);
                bot1_time_left = std::max(bot1_time_left - search_time, 0) + time_inc;
            }

            game_win_type_t result = game.make_move(mov);
            if (result != NO_WIN) {
                if ((result == BLACK_WIN && bot0_playing_black) || (result == WHITE_WIN && !bot0_playing_black)) {
                    score += 1;
                } else if (result != DRAW) {
                    score -= 1;
                }
                break;
            }
        }
        game.reset();
        for (size_t i = 0; i < opening_moves.size(); i++) {
            game.make_move(opening_moves[i]);
        }
        bot0_playing_black = !bot0_playing_black;
    }

    return score;
}


struct tuner_worker
{
    tuner_worker(int base_time, int time_inc, std::shared_ptr<std::vector<float>> p, std::shared_ptr<std::atomic<int>> g, std::vector<std::string> ptt)
    {
        static int worker_count = 0;
        worker_id = worker_count;
        worker_count++;

        games_left = g;
        params = p;
        params_to_tune = ptt;

        t = std::thread(&tuner_worker::tuner_thread, this, base_time, time_inc);
    }

    ~tuner_worker()
    {
        t.join();
    }

    void tuner_thread(int base_time, int time_inc) {
        std::srand(time(NULL) + worker_id);

        alphabeta_search bot0;
        alphabeta_search bot1;

        bot0.set_threads(1);
        bot1.set_threads(1);
        bot0.use_opening_book = true;
        bot1.use_opening_book = true;

        opening_book book;

        int param_index = rand() % params_to_tune.size();

        float apply_factor = 0.005;

        while (*games_left > 0) {
            search_params local_params;
            for (int i = 0; i < local_params.num_of_params(); i++) {
                local_params.data[i] = (int)std::round((*params)[i]);
            }

            int param = search_params::get_variable_index(params_to_tune[param_index]);
            int min_value, max_value;
            search_params::get_limits(param, min_value, max_value);
            int value = local_params.data[param];
            int range = max_value - min_value;

            int delta = std::max(5, range/16);

            float smaller_value = std::clamp(value - delta, min_value, max_value);
            float bigger_value = std::clamp(value + delta, min_value, max_value);


            bot0.sp = local_params;
            bot1.sp = local_params;

            bot0.sp.data[param] = smaller_value;
            bot1.sp.data[param] = bigger_value;

            int score = play_game_pair(bot0, bot1, book, base_time, time_inc);

            if (score > 0) {
                (*params)[param] -= (float)delta * apply_factor;
            } else if (score < 0) {
                (*params)[param] += (float)delta * apply_factor;
            }


            param_index = (param + 1) % params_to_tune.size();

            (*games_left) -= 2;
        }
    }
    int worker_id;
    std::thread t;

    std::vector<std::string> params_to_tune;
    std::shared_ptr<std::atomic<int>> games_left;
    std::shared_ptr<std::vector<float>> params;
};



void tuning_utility::tune_search_params(int time, int time_inc, int threads, int num_of_games, const std::vector<std::string> &params_to_tune)
{
    search_params default_params;

    std::shared_ptr<std::vector<float>> params = std::make_shared<std::vector<float>>();
    std::shared_ptr<std::atomic<int>> games_left = std::make_shared<std::atomic<int>>(num_of_games);
    for (int i = 0; i < default_params.num_of_params(); i++) {
        params->push_back((float)default_params.data[i]);
    }

    std::vector<std::unique_ptr<tuner_worker>> workers;
    for (int i = 0; i < threads; i++) {
        workers.push_back(std::make_unique<tuner_worker>(time, time_inc, params, games_left, params_to_tune));
    }

    std::chrono::high_resolution_clock::time_point last_save_time = std::chrono::high_resolution_clock::now();
    while (*games_left > 0) {
        std::cout << "\rTuning " << num_of_games - *games_left << "/" << num_of_games << "  ";

        std::chrono::high_resolution_clock::time_point time_now = std::chrono::high_resolution_clock::now();

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

    alphabeta_search bot0, bot1;
    for (int i = 0; i < default_params.num_of_params(); i++) {
        bot0.sp.data[i] = (int)std::round((*params)[i]);
    }

    int elo = test(1000, time, time_inc, bot0, bot1, threads, "");

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























