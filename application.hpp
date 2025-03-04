#pragma once

#include <memory>

#include "chessbot/game.hpp"
#include "chessbot/search.hpp"
#include "chessbot/uci.hpp"




class application
{
public:
    void run();
private:
    void initialize();
    void main_loop();

    std::shared_ptr<game_state> game;
    std::shared_ptr<alphabeta_search> alphabeta;
    std::shared_ptr<uci_interface> uci;
};
