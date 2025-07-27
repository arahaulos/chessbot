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

    //alphabeta->load_nnue_net("qnetkb16d2_1.nnue");

    uci = std::make_shared<uci_interface>(alphabeta, game);

    game->get_state().set_initial_state();

    //game->get_state().load_fen("2rq1r1k/pp3ppp/3n4/n2p4/1Q6/2PBBP1P/P4P2/2KR2R1 w - - 0 19");
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

        game_win_type_t result = game->make_move(legal_moves[rand() % legal_moves.size()]);

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

        int16_t eval0 = net.evaluate(game->get_state());
        int16_t eval1 = game->get_state().nnue->evaluate(game->get_state().get_turn(), encode_output_bucket(game->get_state().get_non_pawn_pieces_bb()));
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

    fails += perft_test("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ",                       {0, 20, 400,  8902,  197281,   4865609});
    fails += perft_test("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ",               {0, 48, 2039, 97862, 4085603,  193690690});
    fails += perft_test("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ",                                          {0, 14, 191,  2812,  43238,    674624,   11030083});
    fails += perft_test("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",                {0, 6,  264,  9467,  422333,   15833292});
    fails += perft_test("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  ",                     {0, 44, 1486, 62379, 2103487,  89941194});
    fails += perft_test("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ",       {0, 46, 2079, 89890, 3894594,  164075551});

    fails += test_incremental_updates();

    if (fails == 0) {
        std::cout << "\nAll tests passed." << std::endl;
    } else {
        std::cout << "\n" << fails << " tests failed!!!" << std::endl;
    }
}



void application::run()
{
    uci->start();

    while (!uci->exit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}













