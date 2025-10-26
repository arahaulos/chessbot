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

    static std::string read_text_file(std::string path);
};
