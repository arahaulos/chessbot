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

void export_output_weights(std::string output_file)
{
    std::shared_ptr<nnue_weights> weights = std::make_shared<nnue_weights>();
    weights->save(output_file);
}

void test_search()
{
    std::unique_ptr<searcher> s0 = std::make_unique<searcher>();
    std::unique_ptr<searcher> s1 = std::make_unique<searcher>();

    std::shared_ptr<nnue_weights> w0 = std::make_shared<nnue_weights>();
    std::shared_ptr<nnue_weights> w1 = std::make_shared<nnue_weights>();
    w0->load("qnetkb16_512x2-(16-32-1)x8.nnue");
    w1->load("embedded_weights.nnue");
    s0->set_shared_weights(w0);
    s1->set_shared_weights(w1);

    s0->set_threads(1);
    s1->set_threads(1);
    s0->test_flag = true;
    s1->test_flag = true;

    testing_utility::test(100000, 2000, 50, *s0, *s1, 16, "tuning/UHO_Lichess_4852_v1.epd", 0.0, 3.0f);
}

void train()
{
    //training_data_utility::convert_training_data({"tuning/selfplays_HQ_1"}, "tuning/data_HQ_1", 1024);
    nnue_trainer::train("netkb16_512x2-(16-32-1)x8.nnue", "qnetkb16_512x2-(16-32-1)x8.nnue", {//"tuning/data4", "tuning/data5", "tuning/data6", "tuning/data7",
                                                                                              //"tuning/data8", "tuning/data9", "tuning/data10", "tuning/data11",
                                                                                              //"tuning/data12", "tuning/data13", "tuning/data14", "tuning/data15",
                                                                                              "tuning/data16",  "tuning/data17",
                                                                                              "tuning/data_UHO_1", "tuning/data_UHO_2", "tuning/data_UHO_3", "tuning/data_UHO_4", "tuning/data_UHO_5", "tuning/data_N4M_1"});//*/

    //nnue_trainer::train("netkb16_512x2-(16-32-1)x8.nnue", "qnetkb16_512x2-(16-32-1)x8.nnue", {"tuning/data_HQ_1"});
}

int main()
{
    global_init();
    print_info();


    //train();
    //test_search();

    std::unique_ptr<application> app = std::make_unique<application>();
    //app->run_benchmark();
    app->run();
    //app->run_tests();

    return 0;
}
