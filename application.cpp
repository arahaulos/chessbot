#include "application.hpp"
#include "chessbot/search.hpp"
#include <algorithm>
#include "chessbot/tuning.hpp"
#include "chessbot/pgn_parser.hpp"
#include <iostream>


void application::run()
{
    initialize();
    main_loop();
}


void application::initialize()
{
    game = std::make_shared<game_state>();

    alphabeta = std::make_shared<alphabeta_search>();
    alphabeta->print_search_info = false;
    alphabeta->use_opening_book = false;

    //alphabeta->set_multi_pv(8);

    //alphabeta->set_threads(16);
    //alphabeta->set_transposition_table_size_MB(4096);

    //alphabeta->load_nnue_net("qnetkb16_2_5.nnue");

    uci = std::make_shared<uci_interface>(alphabeta, game);

    game->get_state().set_initial_state();


    //game->get_state().set_square(square_index(3, 0), piece(WHITE, EMPTY));

    //game->get_state().load_fen("r2r2k1/1p1n1pp1/4pnp1/8/PpBRqP2/1Q2B1P1/1P5P/R5K1 b - - ");
    //game->get_state().load_fen("5r1k/4Qpq1/4p3/1p1p2P1/2p2P2/1p2P3/3P4/BK6 b - -");


    //game->get_state().load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - "); //peft 5 193690690


    /*chess_move mov = alphabeta->fast_search(game->get_state(), 25, 0);
    pv_table pv;
    alphabeta->get_pv(pv);
    uint64_t search_time_ms = alphabeta->get_search_time_ms();

    std::cout << "Returned move: " << mov.to_uci() << std::endl;
    std::cout << "PV: "; pv.print();
    std::cout << "Score: " << pv.score << std::endl;
    std::cout << "Time: " << search_time_ms << "ms" << std::endl;*/


   /* alphabeta->run_perft(game->get_state(), 5, false);
    alphabeta->run_perft(game->get_state(), 5, true);
    return;*/

    //void run_perft(board_state &state, int depth, bool validation);


    //game->get_state().load_fen("3r2rk/2p2p2/bp1p1q1p/p2Bp1p1/4PB2/P4P1Q/1PPR2PP/3R3K b - - 0 28");

    //game->get_state().load_fen("1rb2r2/p1qp1nkp/2p2pp1/8/1PQ4P/P1NB2P1/K1P5/3R3R b - - 0 22"); //blunder


    //alphabeta->run_perft(game->get_state(), 5, true);

    //game->get_state().load_fen("5k2/2p2p2/1p3b1p/p2rp3/P7/1PP3B1/8/4R1K1 w - -");

    //game->get_state().load_fen("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - -");

    //game->get_state().load_fen("2rq1r1k/pp3ppp/3n4/n2p4/1Q6/2PBBP1P/P4P2/2KR2R1 w - - 0 19");
    //game->get_state().load_fen("r1bq1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1 b - - 1 7");
    //game->get_state().load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ");

    //game->get_state().load_fen("6k1/1r1q4/p1p2p1p/1p1b1np1/PB1P2Q1/3B3P/P4PP1/4R1K1 b - - 5 37");

    //game->get_state().load_fen("6rk/1p1R3p/2n3p1/p7/8/P3Q3/1qP2PPP/4R1K1 w - - 0 1");
    //game->get_state().load_fen("1k6/1Pp5/2P3p1/p1P1p1P1/2PpPp2/3p1p2/3P1P2/5K2 w - - 0 1");
    //game->get_state().load_fen("8/2p5/1kpq4/p7/P7/8/3Q4/6K1 w - - 0 1");

    //game->get_state().load_fen("r4r1k/pp5p/2pp1q2/3n1b2/6Q1/P2B1NP1/3R1PP1/5RK1 w - - 0 24");

    //game->get_state().load_fen("2r1kb1r/1p2nppp/1qb1p3/1B2P3/8/2N5/PP2QPPP/2RR2K1 w - - 0 2");

    //game->get_state().load_fen("5r1k/2p3pp/p1p5/3p4/5nq1/2P1QNP1/PP3P1P/R5K1 b - - 0 1");
    //game->get_state().load_fen("rn1qk2r/4ppbp/2p2np1/pp2Nb2/3P4/1P4P1/1Q2PPBP/RNB2RK1 b kq - 0 1");

    //game->get_state().load_fen("8/pp2pp2/kpp5/4P3/KP6/4P3/8/8 w - -");


    //5rk1/n4ppp/p1rp1P2/q1N1p1P1/1p2P3/1P5B/P1P1Q2P/1K1RR3 b - - 0 24 blunder

    uci->start();
}


void application::main_loop()
{
    while (!uci->exit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}













