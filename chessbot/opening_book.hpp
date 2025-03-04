#pragma once

#include "state.hpp"
#include "zobrist.hpp"

#include <vector>
#include <list>
#include <map>

struct book_move
{
    book_move(chess_move m) {
        mov = m;
        lines = 1;
    }

    chess_move mov;
    int lines;
};


struct opening_book
{
    opening_book();
    ~opening_book();

    chess_move get_book_move(const board_state &state);

    std::vector<book_move> get_book_moves(const board_state &state);
private:
    std::map<uint64_t, std::vector<book_move>> hash_moves_map;

    void load_line(std::string uci_list);
};
