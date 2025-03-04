#pragma once

#include <vector>
#include <string>

#include "state.hpp"

struct pgn_parser
{
    static chess_move parse_san(const board_state &state, std::string san);
    static std::vector<chess_move> parse_pgn(std::string pgn_text);
    static std::string read_text_file(std::string path);
};
