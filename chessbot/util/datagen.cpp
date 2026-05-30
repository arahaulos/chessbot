#include "datagen.hpp"
#include "../search.hpp"
#include "../game.hpp"
#include <memory>
#include <thread>
#include <cmath>
#include <atomic>
#include <iomanip>
#include "pgn_parser.hpp"
#include "misc.hpp"

#include "../search_manager.hpp"
#include "wdl_model.hpp"



bool adjucate_win(board_state &state, int32_t search_score, int plies_played, datagen_config &config)
{
    if (plies_played < config.adjucate_win_min_plies) {
        return false;
    }

    int32_t normalized_score = wdl_model::normalize_score(state, search_score);

    return (std::abs(normalized_score) > config.adjucate_win_cp_treshold);
}

bool adjucate_draw(board_state &state, int32_t &draw_adjucation_counter, int32_t search_score, int plies_played, datagen_config &config)
{
    if (plies_played < config.adjucate_draw_min_plies) {
        return false;
    }

    int32_t normalized_score = wdl_model::normalize_score(state, search_score);

    if (std::abs(normalized_score) < config.adjucate_draw_cp_treshold) {
        draw_adjucation_counter += 1;
    } else {
        draw_adjucation_counter = 0;
    }

    return (draw_adjucation_counter >= config.adjucate_draw_plies_below_treshold);
}

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

        positions_generated = 0;

        num_of_first_moves = 0;
        first_move_abs_eval_sum = 0;
        total_opening_moves = 0;
        played_opening_moves = 0;

        search = std::make_unique<searcher>();
        sman = std::make_shared<search_manager>();
    }

    void wait()
    {
        sman->stop_search();
        t.join();
    }

    void start(std::atomic<int> &games, std::shared_ptr<nnue_weights> weights, std::shared_ptr<std::vector<std::string>> openings, std::shared_ptr<cache<uint64_t, 64>> startpos_cache, datagen_config &c)
    {
        config = c;

        t = std::thread(&datagen_worker::generate_loop, this, std::ref(games), weights, openings, startpos_cache);
    }

    std::vector<std::string> results;
    double avg_depth;
    double avg_start_pos_abs_eval;
    double avg_opening_branching_factor;
    uint64_t positions_generated;
private:
    void generate_loop(std::atomic<int> &games, std::shared_ptr<nnue_weights> weights, std::shared_ptr<std::vector<std::string>> openings, std::shared_ptr<cache<uint64_t, 64>> startpos_cache);

    std::string generate_game(const std::string &opening_fen, std::shared_ptr<cache<uint64_t, 64>> &startpos_cache);

    std::unique_ptr<searcher> search;
    std::shared_ptr<search_manager> sman;

    game_state game;

    std::thread t;
    int worker_id;

    datagen_config config;

    int num_of_first_moves;
    double first_move_abs_eval_sum;

    long total_opening_moves;
    long played_opening_moves;
};


void datagen_worker::generate_loop(std::atomic<int> &games, std::shared_ptr<nnue_weights> weights, std::shared_ptr<std::vector<std::string>> openings, std::shared_ptr<cache<uint64_t, 64>> startpos_cache)
{
    std::srand((size_t)time(NULL) + worker_id);

    search->set_shared_weights(weights);
    search->forward_pruning = config.forward_pruning;

    while (games > 0) {
        std::string game_pgn = generate_game((*openings)[rand()%openings->size()], startpos_cache);
        if (game_pgn.length() > 0) {
            results.push_back(game_pgn);
            games--;
        }
    }
}


std::string datagen_worker::generate_game(const std::string &s_fen, std::shared_ptr<cache<uint64_t, 64>> &startpos_cache)
{
    game.reset();
    game.get_state().load_fen(s_fen);
    search->new_game();

    game_win_type_t game_result = NO_WIN;
    game_win_type_t adjucated_result = NO_WIN;

    int random_moves = config.opening_moves;
    int draw_adjucation_counter = 0;

    double total_depth = 0;

    std::string opening_fen = "";
    std::vector<chess_move> moves_played;
    std::vector<pgn_comment> comments;

    while (true) {
        if (random_moves > 0) {
            sman->prepare_depth_nodes_search(std::max(config.depth-3, 0), config.nodes);

            std::vector<chess_move> acceptable_moves;
            if (!config.multi_pv_opening) {
                acceptable_moves = game.get_state().get_all_legal_moves(game.get_state().get_turn());
            } else {
                constexpr int multipv = 5;
                constexpr int max_cp_loss = 120;

                if (search->get_multi_pv() != multipv) {
                    search->set_multi_pv(multipv);
                }
                search->search(game.get_state(), sman);

                int32_t best_score = sman->get_evaluation();

                for (int i = 0; i < multipv; i++) {
                    pv_table pv;
                    sman->get_pv(pv, i);

                    if (pv.num_of_moves > 0 && std::abs(pv.score - best_score) < max_cp_loss) {
                        acceptable_moves.push_back(pv.moves[0]);
                    }
                }
            }

            if (acceptable_moves.size() == 0) {
                std::cout << "Error: no legal moves" << std::endl;
                game_result = DRAW;
            } else {
                game_result = game.make_move(acceptable_moves[std::rand() % acceptable_moves.size()]);
                random_moves -= 1;

                played_opening_moves += 1;
                total_opening_moves += acceptable_moves.size();
            }
        } else {
            if (opening_fen == "") {
                if (config.filter_dublicate_openings) {
                    uint64_t zhash = game.get_state().zhash;

                    if ((*startpos_cache)[zhash] == zhash) {
                        break;
                    }
                    (*startpos_cache)[zhash] = zhash;
                }

                opening_fen = game.get_state().generate_fen();
            }

            sman->prepare_depth_nodes_search(config.depth, config.nodes);

            if (search->get_multi_pv() != 1) {
                search->set_multi_pv(1);
            }

            chess_move mov = chess_move::null_move();

            search->search(game.get_state(), sman);

            mov = sman->get_move();
            int32_t score = sman->get_evaluation();
            int depth = sman->get_depth();

            if (is_mate_score(score) || (config.adjucate_wins && adjucate_win(game.get_state(), score, moves_played.size(), config))) {
                if ((game.get_state().get_turn() == BLACK && score < 0) ||
                    (game.get_state().get_turn() == WHITE && score > 0)) {
                    adjucated_result = WHITE_WIN;
                } else {
                    adjucated_result = BLACK_WIN;
                }
            } else if (config.adjucate_draws && adjucate_draw(game.get_state(), draw_adjucation_counter, score, moves_played.size(), config)) {
                adjucated_result = DRAW;
            } else {
                if (moves_played.size() == 0) {
                    first_move_abs_eval_sum += (double)std::abs(score);
                    num_of_first_moves += 1;
                }
                comments.push_back({(int)moves_played.size(), std::to_string(score)});
                moves_played.push_back(mov);
            }

            if (mov == chess_move::null_move()) {
                std::cout << "Engine produced nullmove. depth = " << depth << "  score = " << score << "  move = " << mov.to_uci() << std::endl;
                std::cout << "Position: " << game.get_state().generate_fen() << std::endl;
            }

            total_depth += depth;

            game_result = game.make_move(mov);

            if (adjucated_result != NO_WIN && game_result == NO_WIN) {
                game_result = adjucated_result;
            }
        }
        if (moves_played.size() > 512) {
            game_result = DRAW;
        }
        if (game_result != NO_WIN) {
            break;
        }
    }

    if (moves_played.size() > 0) {
        avg_depth = total_depth / moves_played.size();
        avg_start_pos_abs_eval = first_move_abs_eval_sum / num_of_first_moves;
        avg_opening_branching_factor = (double)total_opening_moves / played_opening_moves;

        positions_generated += moves_played.size();

        std::vector<pgn_tag> tags = {{"FEN", opening_fen}};

        if (game_result == DRAW) {
            tags.push_back({"Result", "1/2-1/2"});
        } else {
            tags.push_back({"Result", (game_result == WHITE_WIN ? "1-0" : "0-1")});
        }

        return pgn_parser::generate_pgn(tags, moves_played, opening_fen, &comments);
    }

    return "";
}


void training_datagen::datagen(std::string output_file, datagen_config &config)
{
    std::cout << std::endl << "Selfplaying\nThreads: " << config.threads
                           << "\nMin depth: " << config.depth
                           << "\nMin nodes: " << config.nodes
                           << "\nOutput file: " << output_file << std::endl;


    auto startpos_cache = std::make_shared<cache<uint64_t, 64>>();

    auto weights = nnue_weights::get_shared_weights(); //Less L3 cache pressure when workers use shared weights

    auto openings = std::make_shared<std::vector<std::string>>();

    //Load opening suite
    if (config.opening_suite != "") {
        *openings = load_opening_suite(config.opening_suite);
    } else {
        openings->push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    std::cout << "Opening suite: " << config.opening_suite << std::endl;
    std::cout << "Opening positions: " << openings->size() << std::endl;
    std::cout << "Multi PV opening: " << config.multi_pv_opening << std::endl;
    std::cout << "Opening moves: " << config.opening_moves << std::endl;

    std::vector<datagen_worker> workers(config.threads);
    std::atomic<int> agames = config.games;

    for (int i = 0; i < config.threads; i++) {
        workers[i].start(agames, weights, openings, startpos_cache, config);
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
                for (int i = 0; i < config.threads; i++) {
                    positions += workers[i].positions_generated;
                    first_move_abs_eval += workers[i].avg_start_pos_abs_eval / config.threads;
                    opening_branching_factor += workers[i].avg_opening_branching_factor / config.threads;
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
                  << (config.games - agames) << "/" << config.games << "  "
                  << std::setw(6) << (int)games_per_hour << " games/h  "
                  << std::setw(5) << (int)positions_per_second << " pos/s   D: "
                  << std::fixed << std::setprecision(2) << avg_depth << "   E: "
                  << std::setprecision(2) << first_move_abs_eval / 100 << "   BF: "
                  << std::setprecision(2) << opening_branching_factor
                  << "  est: " << hours_left << "h " << minutes_left << "min     "
                  << std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << std::endl;


    std::stringstream ss;
    for (int i = 0; i < config.threads; i++) {
        workers[i].wait();

        for (int j = 0; j < workers[i].results.size(); j++) {
            ss << workers[i].results[j] << "\n";
        }
    }

    pgn_parser::write_text_file(output_file, ss.str());
}



























