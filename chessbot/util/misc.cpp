#include "misc.hpp"
#include <fstream>
#include <cmath>

int play_game_pair(alphabeta_search &bot0, alphabeta_search &bot1, const std::string &opening_fen, int base_time, int time_inc, bool bot0_playing_black, match_stats &stats)
{
    game_state game;

    int score = 0;

    for (int i = 0; i < 2; i++) {
        game.reset();
        game.get_state().load_fen(opening_fen);

        bot0.new_game();
        bot1.new_game();

        int bot0_time_left = base_time;
        int bot1_time_left = base_time;

        while (true) {
            chess_move mov;
            if ((bot0_playing_black && game.get_turn() == BLACK) || (!bot0_playing_black && game.get_turn() == WHITE)) {
                int search_time = get_target_search_time(bot0_time_left, time_inc);
                mov = bot0.search_time(game.get_state(), search_time, nullptr);
                bot0_time_left = std::max(bot0_time_left - search_time, 0) + time_inc;

                stats.bot0_depth += bot0.get_depth();
                stats.bot0_nps += bot0.get_nps() / 1000;
                stats.bot0_moves += 1;
            } else {
                int search_time = get_target_search_time(bot1_time_left, time_inc);
                mov = bot1.search_time(game.get_state(), search_time, nullptr);
                bot1_time_left = std::max(bot1_time_left - search_time, 0) + time_inc;

                stats.bot1_depth += bot1.get_depth();
                stats.bot1_nps += bot1.get_nps() / 1000;
                stats.bot1_moves += 1;
            }

            game_win_type_t result = game.make_move(mov);
            if (result != NO_WIN) {
                if ((result == BLACK_WIN && bot0_playing_black) || (result == WHITE_WIN && !bot0_playing_black)) {
                    score += 1;

                    stats.wins += 1;
                } else if (result != DRAW) {
                    score -= 1;

                    stats.losses += 1;
                } else {
                    stats.draws += 1;
                }
                break;
            }
        }
        bot0_playing_black = !bot0_playing_black;
    }

    return score;
}


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


std::vector<std::string> load_opening_suite(std::string filename)
{
    std::vector<std::string> openings;

    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            openings.push_back(line);
        }
    } else {
        std::cout << "Error: can't open opening suite: " << filename << std::endl;
    }
    file.close();

    return openings;
}

