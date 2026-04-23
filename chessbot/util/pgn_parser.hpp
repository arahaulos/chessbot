#pragma once

#include <vector>
#include <string>
#include <tuple>

#include "../state.hpp"


typedef std::pair<std::string, std::string> pgn_tag;

struct pgn_parser
{
    static chess_move parse_san(const board_state &state, std::string san);


    static std::vector<std::string> parse_games(std::string pgn_text);
    static std::vector<pgn_tag> parse_tags(std::string pgn_text);
    static std::vector<chess_move> parse_moves(std::string pgn_text, std::string startpos = "");
    static std::string get_tag_value(const std::vector<pgn_tag> &tags, std::string tag_name);
    static void set_tag_value(std::vector<pgn_tag> &tags, std::string tag_name, std::string tag_value);

    static std::string generate_san(board_state &state, chess_move mov);
    static std::string generate_pgn(const std::vector<pgn_tag> &tags, const std::vector<chess_move> &moves, std::string startpos = "");

    static std::string read_text_file(std::string path);
    static int write_text_file(std::string path, const std::string &text);
};
