#include <memory>
#include <iostream>
#include <chrono>
#include "perft.hpp"


int perft::run_perft(board_state &state, int depth, bool debug)
{
    if (debug) {
        for (size_t i = 0; i < test_perft_tt.get_size(); i++) {
            test_perft_tt[i].clear();
        }

        cache_age += 2;
        if (cache_age > 500) {
            cache_age = 0;
            for (uint64_t i = 0; i < test_perft_tt.get_size(); i++) {
                test_perft_tt[i].clear();
            }
        }

        std::shared_ptr<nnue_weights> weights = std::make_shared<nnue_weights>();

        state.nnue = std::make_shared<nnue_network>(weights);
        state.nnue->evaluate(state);

        int32_t eval_sum = 0;


        search_context *sc = new search_context(nullptr);
        sc->history.conthist_stack = sc->conthist;
        sc->history.move_stack = sc->moves;


        std::cout << "Running debug perft " << depth << "...";

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        uint64_t nodes = debug_perft(state, depth, 0, *sc, eval_sum);

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 );

        std::cout << " done in " << ms.count() << "ms" << std::endl;

        std::cout << "Nodes: " << nodes << std::endl;
        if (ms.count() > 0) {
            std::cout << "Performance: " << (float)(nodes / ms.count()) / 1000 << " million moves per second" << std::endl;
        }
        std::cout << "TT hits: " << sc->stats.cache_hits << std::endl;
        std::cout << "Eval sum: " << eval_sum << std::endl;

        delete sc;

        state.nnue = nullptr;

        return nodes;

    } else {
        std::cout << "Running perft " << depth << "...";

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        uint64_t nodes = performance_perft(state, depth);

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 );

        std::cout << " done in " << ms.count() << "ms" << std::endl;

        std::cout << "Nodes: " << nodes << std::endl;
        if (ms.count() > 0) {
            std::cout << "Performance: " << (float)(nodes / ms.count()) / 1000 << " million moves per second" << std::endl;
        }

        return nodes;
    }
}

uint64_t perft::performance_perft(board_state &state, int depth)
{
    if (depth == 0) {
        return 1;
    }

    move_generator movegen;
    chess_move buffer[255];
    int move_count = movegen.generate_all_pseudo_legal_moves(state, state.get_turn(), buffer);


    uint64_t c = 0;

    for (int i = 0; i < move_count; i++) {
        chess_move m = buffer[i];

        player_type_t turn = state.get_turn();

        unmove_data restore = state.make_move(m);

        if (!state.in_check(turn)) {
            c += performance_perft(state, depth-1);
        }

        state.unmake_move(m, restore);

    }

    return c;
}

uint64_t perft::debug_perft(board_state &state, int depth, int ply, search_context &sc, int32_t &eval_sum)
{
    /*if (state.zhash != hashgen.get_hash(state, state.get_turn())) {
        std::cout << "Hash mismatch" << std::endl;
    }*/

    int32_t tt_score;
    int tt_depth;
    chess_move tt_move = chess_move::null_move();
    int tt_node_type;
    int32_t tt_static_eval;
    bool tt_hit = test_perft_tt[state.zhash].read(state.zhash, tt_move, tt_depth, tt_node_type, tt_score, tt_static_eval, 0);

    if (tt_hit && tt_depth == depth) {
        sc.stats.cache_hits += 1;
        return tt_score;
    }


    if (depth == 0) {
        //eval_sum += state.nnue->evaluate(state.get_turn());
        return 1;
    }

    move_picker picker;
    picker.init(state, ply, tt_move, chess_move::null_move(), sc.history, true);

    chess_move best_move = chess_move::null_move();
    uint64_t c = 0;
    uint64_t bc = 0;
    uint64_t total = 0;
    chess_move m;

    while (picker.pick(m) != MOVES_END) {
        sc.conthist[ply] = sc.history.get_continuation_history_table(state, m);
        sc.moves[ply] = m;

        c = 0;

        uint64_t new_zhash = hashgen.update_hash(state.zhash, state, m);

        test_perft_tt.prefetch(new_zhash);

        unmove_data restore = state.make_move(m, new_zhash);

        c = debug_perft(state, depth-1, ply+1, sc, eval_sum);

        state.unmake_move(m, restore);

        if (c > bc) {
            bc = c;
            best_move = m;
        }
        total += c;
    }

    sc.history.update(state, best_move, depth, ply, picker.picked_quiet_moves, picker.picked_quiet_count, picker.picked_captures, picker.picked_capture_count);

    if (total < 32000) {
        test_perft_tt[state.zhash].write(state.zhash, best_move, depth, ALL_NODE, total, total, 0, cache_age);
    }

    return total;
}



