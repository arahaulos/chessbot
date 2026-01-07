#include "pgn_extract.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>

#include "../search_manager.hpp"
#include "pgn_parser.hpp"


pgn_extractor::pgn_extractor(int num_of_threads, float extract_rate, int nodes_per_position)
{
    running = true;
    total_pgns = 0;

    for (int i = 0; i < num_of_threads; i++) {
        workers.push_back(std::thread(&pgn_extractor::worker_thread, this, extract_rate, nodes_per_position));
    }
}

pgn_extractor::~pgn_extractor() {
    running = false;
    for (auto &t : workers) {
        t.join();
    }
}

void pgn_extractor::add_pgn(const std::string &str)
{
    std::lock_guard<std::mutex> lock(queue_lock);
    total_pgns += 1;
    pgn_queue.push(str);
}

int pgn_extractor::pgns_queued() {
    std::lock_guard<std::mutex> lock(queue_lock);
    return pgn_queue.size();
}

int pgn_extractor::get_total_pgns() {
    return total_pgns;
}

std::vector<selfplay_result> pgn_extractor::get_results()
{
    std::lock_guard<std::mutex> lock(results_lock);
    return results;
}

void pgn_extractor::worker_thread(float extract_rate, int nodes_per_position)
{
    std::shared_ptr<search_manager> search_man = std::make_shared<search_manager>();

    searcher search_instance;

    while (running) {
        std::string pgn_text;
        if (!pop_pgn_queue(pgn_text)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::vector<position_evaluation> evaluations;
        std::vector<pgn_tag> tags = pgn_parser::parse_tags(pgn_text);
        std::string startpos = pgn_parser::get_tag_value(tags, "FEN");
        std::vector<chess_move> moves = pgn_parser::parse_moves(pgn_text, startpos);


        game_win_type_t game_result = NO_WIN;
        std::string game_result_str = pgn_parser::get_tag_value(tags, "Result");

        if (game_result_str == "1-0") {
            game_result = WHITE_WIN;
        } else if (game_result_str == "0-1") {
            game_result = BLACK_WIN;
        } else {
            game_result = DRAW;
        }

        board_state state;
        if (startpos != "") {
            state.load_fen(startpos);
        } else {
            state.set_initial_state();
        }

        search_instance.new_game();
        for (chess_move mov : moves) {
            if (((float)(rand() % 1000) / 1000) > extract_rate) {
                state.make_move(mov);
                continue;
            }

            search_man->prepare_depth_nodes_search(10, nodes_per_position);
            search_instance.search(state, search_man);

            chess_move bm = search_man->get_move();
            int32_t eval = search_man->get_evaluation();

            evaluations.emplace_back(state, eval, bm);

            state.make_move(mov);
        }

        push_evaluations(evaluations, game_result);
    }
}

bool pgn_extractor::pop_pgn_queue(std::string &pgn)
{
    std::lock_guard<std::mutex> lock(queue_lock);
    if (pgn_queue.empty()) {
        return false;
    }
    pgn = pgn_queue.front();
    pgn_queue.pop();
    return true;
}

void pgn_extractor::push_evaluations(const std::vector<position_evaluation> &evals, game_win_type_t result) {
    std::lock_guard<std::mutex> lock(results_lock);

    for (int i = 0; i < evals.size(); i++) {
        results.emplace_back(evals[i], result);
    }
}



bool filter_game(const std::vector<pgn_tag> &tags, float max_elo_diff, float min_elo)
{
    int white_elo = atoi(pgn_parser::get_tag_value(tags, "WhiteElo").c_str());
    int black_elo = atoi(pgn_parser::get_tag_value(tags, "BlackElo").c_str());

    if (white_elo < min_elo || black_elo < min_elo) {
        return true;
    }

    if (std::abs(white_elo - black_elo) > max_elo_diff) {
        return true;
    }

    return false;
}

void pgn_extract::create_dataset(const std::string &pgn_folder, std::string output_file, float max_elo_diff, float min_elo, float extract_rate, int nodes)
{

    std::cout << "Reading PGNs... ";

    pgn_extractor extractor(16, extract_rate, nodes);

    for (const auto& entry : std::filesystem::directory_iterator(pgn_folder)) {
        std::string filename = entry.path().string();
        std::string extension = entry.path().extension().string();

        if (extension == ".pgn") {
            std::string pgn_text = pgn_parser::read_text_file(filename);

            std::vector<std::string> games = pgn_parser::parse_games(pgn_text);

            for (auto game : games) {
                std::vector<pgn_tag> tags = pgn_parser::parse_tags(game);

                if (filter_game(tags, max_elo_diff, min_elo)) {
                    continue;
                }

                extractor.add_pgn(game);
            }
        }
    }

    std::cout << "done." << std::endl;

    int queued_pgns = extractor.pgns_queued();
    int total_pgns = extractor.get_total_pgns();
    while (queued_pgns > 0) {
        std::cout << "\rAnalyzing games " << (total_pgns-queued_pgns) << "/" << total_pgns << "   ";

        queued_pgns = extractor.pgns_queued();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "All games analyzed!" << std::endl;

    std::vector<selfplay_result> results = extractor.get_results();

    std::cout << "Positions extracted and scored: " << results.size() << std::endl;

    save_selfplay_results(results, output_file);

}
