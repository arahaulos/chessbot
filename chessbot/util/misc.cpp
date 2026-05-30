#include "misc.hpp"
#include <fstream>
#include <cmath>
#include "pgn_parser.hpp"
#include "../search_manager.hpp"

int play_game_pair(searcher &s0, searcher &s1, const std::string &opening_fen, int base_time, int time_inc, bool s0_playing_black, match_stats &stats)
{
    game_state game;

    int score = 0;

    std::shared_ptr<time_manager> tman = std::make_shared<time_manager>();
    std::shared_ptr<search_manager> sman = std::make_shared<search_manager>();

    for (int i = 0; i < 2; i++) {
        game.reset();
        game.get_state().load_fen(opening_fen);

        s0.new_game();
        s1.new_game();

        int s0_time_left = base_time;
        int s1_time_left = base_time;

        int ply = 0;

        while (true) {
            chess_move mov;
            if ((s0_playing_black && game.get_turn() == BLACK) || (!s0_playing_black && game.get_turn() == WHITE)) {
                tman->test_flag = s0.test_flag;
                tman->init(s0_time_left, time_inc, ply);
                sman->prepare_clock_search(tman);

                s0.search(game.get_state(), sman);
                s0_time_left = std::max(s0_time_left - tman->get_time_used(), 0) + time_inc;

                mov = sman->get_move();
                stats.s0_depth += sman->get_depth();
                stats.s0_nps += sman->get_nps() / 1000;
                stats.s0_moves += 1;
            } else {
                tman->test_flag = s1.test_flag;
                tman->init(s1_time_left, time_inc, ply);
                sman->prepare_clock_search(tman);

                s1.search(game.get_state(), sman);
                s1_time_left = std::max(s1_time_left - tman->get_time_used(), 0) + time_inc;

                mov = sman->get_move();
                stats.s1_depth += sman->get_depth();
                stats.s1_nps += sman->get_nps() / 1000;
                stats.s1_moves += 1;
            }

            game_win_type_t result = game.make_move(mov);
            if (result != NO_WIN) {
                if ((result == BLACK_WIN && s0_playing_black) || (result == WHITE_WIN && !s0_playing_black)) {
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

            ply++;
        }
        s0_playing_black = !s0_playing_black;
    }

    return score;
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



void iterate_pgn_positions(const std::string &pgn_text, std::function<void(const board_state &state, chess_move next_mov, game_win_type_t game_result, std::string *comment)> callback)
{
    std::vector<std::string> games = pgn_parser::parse_games(pgn_text);

    board_state state;

    for (const std::string &game : games) {

        std::vector<pgn_tag> tags = pgn_parser::parse_tags(game);
        std::vector<pgn_comment> comments;

        std::string startpos = pgn_parser::get_tag_value(tags, "FEN");
        std::string result = pgn_parser::get_tag_value(tags, "Result");

        std::vector<chess_move> moves = pgn_parser::parse_moves(game, startpos, &comments);

        if (startpos == "") {
            state.set_initial_state();
        } else {
            state.load_fen(startpos);
        }

        game_win_type_t game_result = DRAW;
        if (result == "1-0") {
            game_result = WHITE_WIN;
        } else if (result == "0-1") {
            game_result = BLACK_WIN;
        }

        int comment_index = 0;

        for (size_t i = 0; i < moves.size(); i++) {
            chess_move mov = moves[i];

            if (comment_index < comments.size() && comments[comment_index].pos == i) {

                callback(state, mov, game_result, &comments[comment_index].comment);

                comment_index++;
            } else {
                callback(state, mov, game_result, nullptr);
            }

            state.make_move(mov);
        }
    }
}







