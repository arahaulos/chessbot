#include <algorithm>
#include <iostream>
#include <sstream>
#include "application.hpp"
#include "chessbot/tuning.hpp"
#include "chessbot/pgn_parser.hpp"
#include "chessbot/perft.hpp"
#include "chessbot/search.hpp"


application::application()
{
    game = std::make_shared<game_state>();

    alphabeta = std::make_shared<alphabeta_search>();
    alphabeta->use_opening_book = false;

    uci = std::make_shared<uci_interface>(alphabeta, game);

    game->get_state().set_initial_state();
}


void application::selfplay(std::string folder, std::string nnue_file, int threads, int depth, int nodes, bool random, int games_per_file, int max_files)
{
    int filenum = 0;
    while (filenum < max_files) {
        filenum++;

        std::stringstream ss;
        ss << folder << "\\nodes" << nodes / 1000 << "k_" << games_per_file << "_" << filenum << ".txt";

        std::string filename = ss.str();

        std::ifstream file(filename);
        if (file.is_open()) {
            file.close();
            continue;
        }
        tuning_utility::selfplay(filename, nnue_file, threads, games_per_file, depth, nodes, random);
    }
}



int application::perft_test(std::string position_fen, std::vector<int> expected_results)
{
    perft p;

    int fails = 0;

    for (int i = 1; i < expected_results.size(); i++) {
        game->get_state().load_fen(position_fen);

        std::cout << "Running perft " << i << " on " << position_fen << "  Expected result: " << expected_results[i] << std::endl;

        int result0 = p.run_perft(game->get_state(), i, false);
        int result1 = p.run_perft(game->get_state(), i, true);

        if (result0 != expected_results[i] || result1 != expected_results[i]) {
            std::cout << "\n\nFailed! Expected: " << expected_results[i] << std::endl;
            std::cout << "Movegenerator: " << result0 << std::endl;
            std::cout << "Movepicker: " << result1 << std::endl;
            std::cout << "Position: " << position_fen << "  depth " << i << std::endl;
            std::cout << "---------------------------------------------\n" << std::endl;
            fails++;
        } else {
            std::cout << "Passed" << std::endl;
        }
    }
    return fails;
}

int application::test_incremental_updates()
{
    std::cout << "Testing incremental updates" << std::endl;

    bool failed = false;

    std::shared_ptr<nnue_weights> weights = std::make_shared<nnue_weights>();
    nnue_network net(weights);

    game->get_state().nnue = std::make_shared<nnue_network>(weights);
    game->get_state().nnue->evaluate(game->get_state());


    int moves_left = 1000000;
    int max_depth = MAX_DEPTH;
    int depth = 0;

    while (moves_left-- > 0) {
        std::vector<chess_move> legal_moves = game->get_state().get_all_legal_moves(game->get_state().get_turn());

        chess_move mov = legal_moves[rand() % legal_moves.size()];

        uint32_t pawn_hash = game->get_state().pawn_hash();
        uint32_t minor_hash = game->get_state().minor_hash();
        uint32_t major_hash = game->get_state().major_hash();
        uint32_t material_hash = game->get_state().material_hash();

        piece_type_t moving = mov.get_moving_piece().get_type();
        piece_type_t captured = mov.get_captured_piece().get_type();

        bool pawn_hash_should_change = (moving == PAWN || captured == PAWN);
        bool major_hash_should_change = (moving == ROOK || moving == QUEEN || moving == KING ||
                                         captured == ROOK || captured == QUEEN ||
                                         (moving == PAWN && (mov.promotion == ROOK || mov.promotion == QUEEN)));

        bool minor_hash_should_change = (moving == BISHOP || moving == KNIGHT || moving == KING ||
                                         captured == BISHOP || captured == KNIGHT ||
                                         (moving == PAWN && (mov.promotion == BISHOP || mov.promotion == KNIGHT)));

        bool material_hash_should_change = (captured != EMPTY || mov.promotion != EMPTY);

        game_win_type_t result = game->make_move(mov);

        if ((game->get_state().pawn_hash() != pawn_hash) != pawn_hash_should_change) {
            std::cout << "Pawn hash failed!" << std::endl;
            failed = true;
            break;
        }
        if ((game->get_state().material_hash() != material_hash) != material_hash_should_change) {
            std::cout << "Material hash failed!" << std::endl;
            failed = true;
            break;
        }
        if ((game->get_state().minor_hash() != minor_hash) != minor_hash_should_change) {
            std::cout << "Minor hash failed!" << std::endl;
            failed = true;
            break;
        }
        if ((game->get_state().major_hash() != major_hash) != major_hash_should_change) {
            std::cout << "Major hash failed!" << std::endl;
            failed = true;
            break;
        }


        depth += 1;

        if (result != NO_WIN || depth >= max_depth) {

            int unmake = (rand() % (depth-1))+1;
            for (int i = 0; i < unmake; i++) {
                game->unmove();
                depth--;
            }
        }

        uint64_t zhash0 = game->get_state().zhash;
        uint64_t zhash1 = hashgen.get_hash(game->get_state(), game->get_state().get_turn());

        uint64_t shash0 = game->get_state().structure_hash;
        uint64_t shash1 = hashgen.get_structure_hash(game->get_state(), game->get_state().get_turn());

        int16_t eval0 = net.evaluate(game->get_state());
        int16_t eval1 = game->get_state().nnue->evaluate(game->get_state().get_turn());
        if (eval0 != eval1) {
            std::cout << "Evaluation failed!!! " << eval0 << " != " << eval1 << std::endl;
            failed = true;
            break;
        }

        if (zhash0 != zhash1) {
            std::cout << "Zobrist hashing failed!!! " << zhash0 << " != " << zhash1 << std::endl;
            failed = true;
            break;
        }

        if (shash0 != shash1) {
            std::cout << "Zobrist structure hashing failed!!! " << shash0 << " != " << shash1 << std::endl;
            failed = true;
            break;
        }
    }

    game->get_state().nnue = nullptr;

    if (failed == 0) {
        std::cout << "Passed" << std::endl;
    }

    return failed;
}



void application::run_tests()
{
    int fails = 0;

    fails += test_incremental_updates();

    fails += perft_test("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ",                       {0, 20, 400,  8902,  197281,   4865609});
    fails += perft_test("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ",               {0, 48, 2039, 97862, 4085603,  193690690});
    fails += perft_test("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ",                                          {0, 14, 191,  2812,  43238,    674624,   11030083});
    fails += perft_test("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",                {0, 6,  264,  9467,  422333,   15833292});
    fails += perft_test("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ",                     {0, 44, 1486, 62379, 2103487,  89941194});
    fails += perft_test("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ",       {0, 46, 2079, 89890, 3894594,  164075551});

    if (fails == 0) {
        std::cout << "\nAll tests passed." << std::endl;
    } else {
        std::cout << "\n" << fails << " tests failed!!!" << std::endl;
    }
}



uint64_t application::bench_position(std::string position_fen, int depth)
{
    std::cout << "Searching position " << position_fen << " depth " << depth << std::endl;
    game->get_state().load_fen(position_fen);
    alphabeta->fast_search(game->get_state(), depth, 0);
    //alphabeta->test_flag = true;

    std::cout << "bestmove " << alphabeta->get_move().to_uci() << std::endl;

    return alphabeta->get_node_count();
}


void application::run_benchmark()
{
    uint64_t total_nodes = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; i++) {

        total_nodes += bench_position("2rq1r1k/pp3ppp/3n4/n2p4/1Q6/2PBBP1P/P4P2/2KR2R1 w - - 0 19", 20);

        total_nodes += bench_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 20);
        total_nodes += bench_position("5rk1/1b2bpp1/p3pn1p/1p1q4/3B4/1P1NPP1P/P1rNQ1P1/R1R3K1 b - -", 20);
        total_nodes += bench_position("8/4b1k1/8/p4Q2/1p2N1Pp/1P1K3P/P5q1/8 w - -", 20);
        total_nodes += bench_position("3r2k1/5pp1/r7/P7/1nNBp2p/1P2P2P/5PP1/3R2K1 w - -", 20);
        total_nodes += bench_position("8/6R1/3k3p/1prp1p2/p1n2P2/1B2P2P/P4PK1/8 w - -", 20);
        total_nodes += bench_position("r3k2r/pp1q1pb1/2npb1p1/2pN3p/2PnPB2/2NP3P/PP2B1P1/R2Q1RK1 b kq -", 20);
        total_nodes += bench_position("3rr2k/6p1/2nb3p/p1pq4/3p2Q1/PP2P1PP/4RP2/BN1R2K1 w - -", 20);
        total_nodes += bench_position("r7/8/3N2k1/3P2p1/3B1p2/1b3P2/6PK/8 b - -", 20);
        total_nodes += bench_position("r1b2rk1/4np1p/1p2pnp1/p5N1/PqPN4/1P1R3P/3Q1PP1/4RBK1 w - -", 20);

        total_nodes += bench_position("8/3r2pk/1Q4np/1Nn3q1/P1P5/2B2b2/6PP/4RBK1 b - -", 20);
        total_nodes += bench_position("2b1r1k1/Q4pp1/1pq2n1p/4p3/1PP2b2/5N1P/P3BPP1/3R1RK1 b - -", 20);
        total_nodes += bench_position("2b3r1/2P4k/p3N1p1/7p/2N1R3/2Pr1P2/1R3bPK/8 b - -", 20);
        total_nodes += bench_position("r1bq1rk1/1pp2pp1/p2p3p/3Bp3/3NPPPb/P1NP4/1PPQ3P/R2K2R1 b - -", 20);
        total_nodes += bench_position("r5k1/1pq1rp2/2p3pp/3bN1n1/p2P4/P1Q1RPP1/1P5P/4RBK1 b - -", 20);
        total_nodes += bench_position("8/1p3pk1/6p1/7p/8/PR4P1/1P2K2P/2r5 b - -", 20);
        total_nodes += bench_position("3r4/5pk1/R4p2/1R6/1r5p/1bN4P/1P3PP1/6K1 b - -", 20);
        total_nodes += bench_position("r2q1rk1/1b2bppp/1p2pn2/pPnP4/3N4/P2BP3/1B1N1PPP/R2Q1RK1 w - -", 20);
        total_nodes += bench_position("3r4/1RR2p1p/4ppk1/p7/P7/1P2PPK1/1r4PP/8 b - -", 20);
        total_nodes += bench_position("r4rk1/ppp1nppp/2nbbq2/8/Q2pP3/3P1N2/PP1NBPPP/R1B2RK1 w - -", 20);

    }

    auto end_time = std::chrono::high_resolution_clock::now();

    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Time: " << ms << "ms   Nodes: " << total_nodes << "  Speed: " << total_nodes / ms << "Knps" << std::endl;

}



void application::run()
{
    uci->start();

    while (!uci->exit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}













