#pragma once


#include "state.hpp"
#include "movepicker.hpp"
#include "movegen.hpp"
#include "cache.hpp"
#include "transposition.hpp"
#include "search.hpp"

#include "nnue/nnue.hpp"

struct perft
{
    perft() {
        cache_age = 0;
    }

    int run_perft(board_state &state, int depth, bool debug);
private:
    uint64_t performance_perft(board_state &state, int depth);
    uint64_t debug_perft(board_state &state, int depth, int ply, search_context &sc);

    cache<tt_bucket, 1> test_perft_tt;

    int cache_age;
};
