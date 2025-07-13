#include <iostream>
#include "application.hpp"
#include "chessbot/nnue/training/training.hpp"

#include "chessbot/pgn_parser.hpp"
#include <chrono>
#include <thread>


void global_init()
{
    std::srand(time(NULL));
}


int main()
{
    global_init();

    //training_data_utility::convert_training_data({"tuning/selfplays_nnue15"}, "tuning/data10", 1024);

    //nnue_trainer::train("netkb16d3_9.nnue", "qnetkb16d3_9.nnue", {"tuning/data9", "tuning/data10"});

    //tuning_utility::tune_search_params(1000, 100, 16, 120000, {"rfmargin_base", "rfmargin_mult", "rfmargin_improving_modifier", "fmargin_base", "fmargin_mult"});

    /*std::vector<selfplay_result> positions;
    load_selfplay_results(positions, "tuning\\selfplays_nnue8\\nodes6k_32000_1.txt");
    nnue_trainer::test_nets("netkb16d3_8.nnue", "qnetkb16d3_8.nnue", positions);//*/

    /*std::unique_ptr<alphabeta_search> bot0 = std::make_unique<alphabeta_search>();
    std::unique_ptr<alphabeta_search> bot1 = std::make_unique<alphabeta_search>();

    //bot0->load_nnue_net("qnetkb16d3_9.nnue");
    //bot1->load_nnue_net("embedded_weights.nnue");

    bot0->set_threads(1);
    bot1->set_threads(1);
    bot0->use_opening_book = true;
    bot1->use_opening_book = true;
    bot0->experimental_features = true;
    bot1->experimental_features = false;


    tuning_utility::test(20000, 1000, 100, *bot0, *bot1, 16, "", 0.0f, 5.0f);
    //tuning_utility::test(20000, 16000, 1000, *bot0, *bot1, 16, "builds/1_4_25/chessbot.exe", 0.0f, 5.0f);
    //*/

    std::unique_ptr<application> app = std::make_unique<application>();
    app->run();
    //app->run_tests();
    //app->selfplay("tuning\\selfplays_nnue15", "", 28, 8, 6000, false, 32000, 1000);

    return 0;
}
