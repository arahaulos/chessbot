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


    //training_data_utility::convert_training_data({"tuning\\selfplays_nnue8"}, "tuning\\data3", 1024);

    //nnue_trainer::train("netkb16_2_6.nnue", "qnetkb16_2_6.nnue", {"tuning\\data_d8", "tuning\\data2", "tuning\\data3"});

    //tuning_utility::tune_search_params(1000, 50, 16, 60000, {"rfmargin_base", "rfmargin_mult", "rfmargin_improving_modifier", "fmargin_base", "fmargin_mult", "hmargin_mult"});

    /*std::unique_ptr<alphabeta_search> bot0 = std::make_unique<alphabeta_search>();
    std::unique_ptr<alphabeta_search> bot1 = std::make_unique<alphabeta_search>();

    bot0->set_threads(1);
    bot1->set_threads(1);
    bot0->use_opening_book = true;
    bot1->use_opening_book = true;
    bot0->experimental_features = true;
    bot1->experimental_features = false;

    //tuning_utility::test(20000, 1000, 50, *bot0, *bot1, 16, "", 0.0f, 5.0f);
    tuning_utility::test(20000, 16000, 1000, *bot0, *bot1, 16, "builds/4_3_25/chessbot_x64.exe", 0.0f, 5.0f);
    //*/

    std::unique_ptr<application> app = std::make_unique<application>();
    app->run();
    //app->run_tests();
    //app->selfplay("tuning\\selfplays_nnue8", "embedded_weights.nnue", 28, 8, 6000, 32000, 1000);

    return 0;
}
