#include <iostream>
#include "application.hpp"
#include "chessbot/nnue/training/training.hpp"
#include "chessbot/search.hpp"
#include "chessbot/util/datagen.hpp"
#include "chessbot/util/testing.hpp"
#include "chessbot/util/tuning.hpp"
#include "chessbot/util/pgn_parser.hpp"

void global_init()
{
    //nnue_weights::get_shared_weights();
    std::srand(time(NULL));
}

void print_info()
{
    std::cout << "Built: " << __DATE__ << "   "
              << (USE_AVX2 ? "AVX2" : "SSE4.1") << " "
              << (USE_PEXT ? "BMI2" : "POPCNT") << " "
              << (USE_HUGEPAGES ? "hugepages" : "") << std::endl;
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
    //training_data_utility::convert_training_data({"tuning/raw_data/selfplays_UHO_9"}, "tuning/bin_data/data_UHO_9", 1024);

    nnue_trainer::train("nn776x2-pwm-psqt-8ls.nnue", "qnn776x2-pwm-psqt-8ls.nnue",   {//"tuning/data8", "tuning/data9", "tuning/data10", "tuning/data11",
                                                                                       //"tuning/data12", "tuning/data13",
                                                                                       //"tuning/data14", "tuning/data15",
                                                                                       //"tuning/bin_data/data16",  "tuning/bin_data/data17",
                                                                                       "tuning/bin_data/data_UHO_1", "tuning/bin_data/data_UHO_2", "tuning/bin_data/data_UHO_3", "tuning/bin_data/data_UHO_4", "tuning/bin_data/data_UHO_5",
                                                                                       "tuning/bin_data/data_N4M_1", "tuning/bin_data/data_UHO_6", "tuning/bin_data/data_UHO_7" , "tuning/bin_data/data_UHO_8", "tuning/bin_data/data_UHO_9",
                                                                                       "tuning/bin_data/data_S14M_1", "tuning/bin_data/data_S14M_2",
                                                                                       "tuning/bin_data/data_S12M_1", "tuning/bin_data/data_S12M_2",
                                                                                       "tuning/bin_data/data_R8_1", "tuning/bin_data/data_R8_2", "tuning/bin_data/selfplays_UHO_NOFP_1"});//*/
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

    tuning_utility::tune_search_params(2000, 50, 16, 100000, 4, "tuning.txt", params, "tuning/UHO_4060_v4.epd");
}

void test_net()
{
    nnue_trainer::test_nets("nn1032x2-pwm-psqt-8ls.nnue", "qnn1032x2-pwm-psqt-8ls.nnue", pgn_parser::read_text_file("tuning/net_testset.pgn"));
}


void datagen()
{
    std::string folder = "tuning/raw_data/selfplays_UHO_9";

    datagen_config config;

    config.depth = 8;
    config.nodes = 6000;
    config.forward_pruning = true;
    config.multi_pv_opening = true;
    config.opening_moves = 8;
    config.games = 32000;
    config.opening_suite = "tuning/UHO_4060_v4.epd";
    config.threads = 28;
    config.filter_dublicate_openings = true;
    config.adjucate_wins = true;
    config.adjucate_win_cp_treshold = 1200;
    config.adjucate_win_min_plies = 40;

    config.adjucate_draws = true;
    config.adjucate_draw_min_plies = 80;
    config.adjucate_draw_plies_below_treshold = 10;
    config.adjucate_draw_cp_treshold = 50;

    int filenum = 0;
    while (filenum < 1000) {
        filenum++;

        std::stringstream ss;
        ss << folder << "/nodes" << config.nodes / 1000 << "k_" << config.games << "_" << filenum << ".pgn";

        std::string filename = ss.str();

        std::ifstream file(filename);
        if (file.is_open()) {
            file.close();
            continue;
        }
        training_datagen::datagen(filename, config);
    }
}


int main()
{
    global_init();
    print_info();

    //tune_search();
    train();
    //test_search();
    //test_net();
    //datagen();

    std::unique_ptr<application> app = std::make_unique<application>();
    app->run();

    return 0;
}
