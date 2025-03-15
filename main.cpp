#include <iostream>
#include "application.hpp"
#include "chessbot/nnue/training/training.hpp"

void global_init()
{
    std::srand(time(NULL));
}


int main()
{
    global_init();

    //nnue_trainer::train("netkb16d2_1.nnue", "qnetkb16d2_1.nnue", {"tuning\\data2", "tuning\\data3"});

    //tuning_utility::tune_search_params(1000, 50, 16, 60000, {"rfmargin_base", "rfmargin_mult", "rfmargin_improving_modifier", "fmargin_base", "fmargin_mult", "razoring_margin"});

    /*std::vector<selfplay_result> positions;
    load_selfplay_results(positions, "tuning\\selfplays_nnue8\\nodes6k_32000_1.txt");
    nnue_trainer::test_nets("netkb16d2_1.nnue", "qnetkb16d2_1.nnue", positions);//*/

    /*std::unique_ptr<alphabeta_search> bot0 = std::make_unique<alphabeta_search>();
    std::unique_ptr<alphabeta_search> bot1 = std::make_unique<alphabeta_search>();

    //bot0->load_nnue_net("qnetkb16d2_1.nnue");
    //bot1->load_nnue_net("embedded_weights.nnue");
    //bot1->load_nnue_net("qnetkb16d_1.nnue");

    bot0->set_threads(1);
    bot1->set_threads(1);
    bot0->use_opening_book = true;
    bot1->use_opening_book = true;
    bot0->experimental_features = true;
    bot1->experimental_features = true;

    tuning_utility::test(20000, 1000, 100, *bot0, *bot1, 16, "", 0.0f, 5.0f);
    //tuning_utility::test(20000, 16000, 1000, *bot0, *bot1, 16, "builds/11_3_25/chessbot.exe", 0.0f, 5.0f);
    //*/

    std::unique_ptr<application> app = std::make_unique<application>();
    app->run();
    //app->run_tests();
    //app->selfplay("tuning\\selfplays_nnue8", "embedded_weights.nnue", 28, 8, 6000, 32000, 1000);

    return 0;
}
