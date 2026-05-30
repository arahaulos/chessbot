#pragma once


#include "training_data.hpp"

struct selfplay_result;

struct nnue_trainer {
    static void train(std::string net_file, std::string qnet_file, std::vector<std::string> dataset_directories);

    static void test_nets(std::string training_net_file, std::string quantized_net_file, const std::string &pgn_dataset);

    static void quantize_net(std::string net_file, std::string qnet_file);

    static float find_scaling_factor_for_net(std::string qnet_file, const std::string &pgn_dataset);
};
