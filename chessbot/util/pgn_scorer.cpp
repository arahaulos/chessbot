#include "pgn_scorer.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <sstream>

#include "../search_manager.hpp"
#include "pgn_parser.hpp"


pgn_scorer_pool::pgn_scorer_pool(int num_of_threads, int nodes_per_position)
{
    running = true;
    total_pgns = 0;

    for (int i = 0; i < num_of_threads; i++) {
        workers.push_back(std::thread(&pgn_scorer_pool::worker_thread, this, nodes_per_position));
    }
}

pgn_scorer_pool::~pgn_scorer_pool() {
    running = false;
    for (auto &t : workers) {
        t.join();
    }
}

void pgn_scorer_pool::add_pgn(const std::string &str)
{
    std::lock_guard<std::mutex> lock(queue_lock);
    total_pgns += 1;
    pgn_queue.push(str);
}

int pgn_scorer_pool::pgns_queued() {
    std::lock_guard<std::mutex> lock(queue_lock);
    return pgn_queue.size();
}

int pgn_scorer_pool::get_total_pgns() {
    return total_pgns;
}

std::string pgn_scorer_pool::get_result()
{
    std::lock_guard<std::mutex> lock(results_lock);

    std::stringstream ss;

    for (const std::string &r : results) {
        ss << r << "\n\n";
    }

    return ss.str();
}

void pgn_scorer_pool::worker_thread(int nodes_per_position)
{
    std::shared_ptr<search_manager> search_man = std::make_shared<search_manager>();

    searcher search_instance;

    while (running) {
        std::string pgn_text;
        if (!pop_pgn_queue(pgn_text)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::vector<pgn_tag> tags = pgn_parser::parse_tags(pgn_text);
        std::string startpos = pgn_parser::get_tag_value(tags, "FEN");
        std::vector<chess_move> moves = pgn_parser::parse_moves(pgn_text, startpos);

        std::vector<pgn_comment> comments;

        board_state state;
        if (startpos != "") {
            state.load_fen(startpos);
        } else {
            state.set_initial_state();
        }

        search_instance.new_game();
        for (chess_move mov : moves) {
            search_man->prepare_depth_nodes_search(10, nodes_per_position);
            search_instance.search(state, search_man);

            int32_t eval = search_man->get_evaluation();

            comments.push_back({(int)comments.size(), std::to_string(eval)});

            state.make_move(mov);
        }


        {
            std::lock_guard<std::mutex> lock(results_lock);
            results.push_back(pgn_parser::generate_pgn(tags, moves, startpos, &comments));
        }
    }
}

bool pgn_scorer_pool::pop_pgn_queue(std::string &pgn)
{
    std::lock_guard<std::mutex> lock(queue_lock);
    if (pgn_queue.empty()) {
        return false;
    }
    pgn = pgn_queue.front();
    pgn_queue.pop();
    return true;
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

void pgn_scorer::create_dataset(const std::string &pgn_folder, std::string output_file, float max_elo_diff, float min_elo, int nodes)
{

    std::cout << "Reading PGNs... ";

    pgn_scorer_pool scorer(16, nodes);

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

                scorer.add_pgn(game);
            }
        }
    }

    std::cout << "done." << std::endl;

    int queued_pgns = scorer.pgns_queued();
    int total_pgns = scorer.get_total_pgns();
    while (queued_pgns > 0) {
        std::cout << "\rAnalyzing games " << (total_pgns-queued_pgns) << "/" << total_pgns << "   " << std::flush;

        queued_pgns = scorer.pgns_queued();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "All games analyzed!" << std::endl;


    pgn_parser::write_text_file(output_file, scorer.get_result());
}
