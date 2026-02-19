#include <iostream>
#include "application.hpp"
#include "chessbot/nnue/training/training.hpp"
#include "chessbot/util/testing.hpp"
#include "chessbot/util/tuning.hpp"

void global_init()
{
    nnue_weights::get_shared_weights();
    std::srand(time(NULL));
}

void print_info()
{
    std::cout << "Built: " << __DATE__ << "   " << (USE_AVX2 ? "AVX2" : "SSE4.1") << " " << (USE_PEXT ? "BMI2" : "POPCNT") << std::endl;
}


void test_search()
{
    std::unique_ptr<searcher> s0 = std::make_unique<searcher>();
    std::unique_ptr<searcher> s1 = std::make_unique<searcher>();

    std::shared_ptr<nnue_weights> w0 = std::make_shared<nnue_weights>();
    std::shared_ptr<nnue_weights> w1 = std::make_shared<nnue_weights>();
    w0->load("qnn776x2-pwm-psqt-8ls.nnue");
    w1->load("embedded_weights.nnue");
    s0->set_shared_weights(w0);
    s1->set_shared_weights(w1);

    s0->set_threads(1);
    s1->set_threads(1);
    s0->test_flag = true;
    s1->test_flag = true;

    testing_utility::test(100000, 8000, 100, *s0, *s1, 16, "tuning/UHO_4060_v4.epd", 0.0, 5.0f);
}

void train()
{
    //training_data_utility::convert_training_data({"tuning/selfplays_S14M_2"}, "tuning/data_S14M_2", 1024);
    nnue_trainer::train("nn776x2-pwm-psqt-8ls.nnue", "qnn776x2-pwm-psqt-8ls.nnue",   {//"tuning/data8", "tuning/data9", "tuning/data10", "tuning/data11",
                                                                                      //"tuning/data12", "tuning/data13",
                                                                                      //"tuning/data14", "tuning/data15",
                                                                                      "tuning/data16",  "tuning/data17",
                                                                                      "tuning/data_UHO_1", "tuning/data_UHO_2", "tuning/data_UHO_3", "tuning/data_UHO_4", "tuning/data_UHO_5",
                                                                                      "tuning/data_N4M_1",
                                                                                      "tuning/data_S14M_1", "tuning/data_S14M_2"});//*/

}

void tune_search()
{
    std::vector<std::string> params =
    {
        "rfmargin_base",
        "rfmargin_mult",
        "rfmargin_improving_modifier",
        "fmargin_base",
        "fmargin_mult",
        "hmargin_mult",
        "razoring_margin",
        "probcut_margin",
        "good_quiet_treshold"
    };

    tuning_utility::tune_search_params(2000, 50, 16, 200000, "tuning.txt", params, "tuning/UHO_4060_v4.epd");
}

void test_net()
{
    std::vector<selfplay_result> data;
    load_selfplay_results(data, "tuning/selfplays_UHO_6/nodes6k_32000_1.txt");
    data = get_random_batch(data, 100000);
    nnue_trainer::test_nets("nn776x2-pwm-psqt-8ls.nnue", "qnn776x2-pwm-psqt-8ls.nnue", data);
}

int main()
{
    global_init();
    print_info();

    //tune_search();
    //train();
    //test_search();
    //test_net();

    std::unique_ptr<application> app = std::make_unique<application>();
    //app->run_benchmark();
    app->run();
    //app->run_tests();
    //app->datagen("tuning/selfplays_UHO_6", "", "tuning/UHO_Lichess_4852_v1.epd", 28, 8, 6000, true, 6, 32000, 1000);

    return 0;
}
