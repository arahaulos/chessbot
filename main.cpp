#include <iostream>
#include "application.hpp"
#include "chessbot/nnue/training/training.hpp"
#include "chessbot/util/testing.hpp"
#include "chessbot/util/tuning.hpp"
#include "chessbot/util/datagen.hpp"

void global_init()
{
    std::srand(time(NULL));
}

void print_info()
{
    std::cout << "Version: " << __DATE__ << "   " << (USE_AVX2 ? "AVX2" : "SSE4.1") << " " << (USE_PEXT ? "BMI2" : "POPCNT") << std::endl;
}


int main()
{
    global_init();
    print_info();

    /*std::vector<selfplay_result> positions;
    load_selfplay_results(positions, "tuning/selfplays_nnue20/nodes5k_32000_1.txt");
    nnue_trainer::find_scaling_factor_for_net("qnetkb16_512x2-(16-32-1)x8.nnue", positions);
    nnue_trainer::test_nets("netkb16_512x2-(16-32-1)x8.nnue", "qnetkb16_512x2-(16-32-1)x8.nnue", positions);//*/

    //training_data_utility::convert_training_data({"tuning/selfplays_nnue21"}, "tuning/data16", 1024);

    //nnue_trainer::train("netkb16_512x2-(16-32-1)x8.nnue", "qnetkb16_512x2-(16-32-1)x8.nnue", {"tuning/data8", "tuning/data9", "tuning/data10", "tuning/data11", "tuning/data12", "tuning/data13", "tuning/data14", "tuning/data15", "tuning/data16"});

    /*tuning_utility::tune_search_params(1000, 25, 16, 100000, "search_tuning.txt", {"rfmargin_mult", "rfmargin_improving_modifier",
                                                                                   "fmargin_mult", "fmargin_base",
                                                                                   "hmargin_mult", "razoring_margin",
                                                                                   "probcut_margin", "good_quiet_treshold"}, "tuning/UHO_4060_v4.epd");//*/

    /*std::unique_ptr<alphabeta_search> bot0 = std::make_unique<alphabeta_search>();
    std::unique_ptr<alphabeta_search> bot1 = std::make_unique<alphabeta_search>();

    bot0->load_nnue_net("qnetkb16_512x2-(16-32-1)x8.nnue");
    bot1->load_nnue_net("embedded_weights.nnue");
    bot0->set_threads(1);
    bot1->set_threads(1);
    bot0->test_flag = true;
    bot1->test_flag = true;

    testing_utility::test(100000, 1000, 100, *bot0, *bot1, 16, "tuning/UHO_4060_v4.epd", 0.0, 5.0f);
    //*/

    std::unique_ptr<application> app = std::make_unique<application>();
    //app->run_benchmark();
    app->run();
    //app->run_tests();
    //app->datagen("tuning\\selfplays_nnue21", "", 28, 8, 5000, false, 32000, 1000);

    return 0;
}
