#include <iostream>
#include "application.hpp"
#include "chessbot/bitboard.hpp"
#include "chessbot/tuning.hpp"
#include "chessbot/nnue/training/training.hpp"
#include "chessbot/pgn_parser.hpp"
#include <iomanip>

#include <sstream>
#include <random>

using namespace std;


void global_init()
{
    //global_eval_params.load_params("params.params");

    std::srand(time(NULL));
}


void self_play(std::string folder, std::string nnue_file, int threads, int depth, int nodes, int games_per_file)
{
    int filenum = 0;
    while (filenum < 1000) {
        filenum++;

        std::stringstream ss;
        ss << folder << "\\nodes" << nodes / 1000 << "k_" << games_per_file << "_" << filenum << ".txt";

        std::string filename = ss.str();

        std::ifstream file(filename);
        if (file.is_open()) {
            continue;
        }
        file.close();

        tuning_utility::selfplay(filename, nnue_file, threads, games_per_file, depth, nodes, true);
    }
}


int main()
{
    global_init();


    //training_data_utility::convert_training_data({"tuning\\selfplays_nnue8"}, "tuning\\data3", 1024);

    //self_play("tuning\\selfplays_nnue8", "embedded_weights.nnue", 28, 8, 6000, 32000);

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

    //tuning_utility::test(20000, 8000, 100, *bot0, *bot1, 16, "", 0.0f, 5.0f);*/

    std::unique_ptr<application> app = std::make_unique<application>();
    app->run();
    return 0;
}
