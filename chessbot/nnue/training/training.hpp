#pragma once


#include "training_data.hpp"

struct selfplay_result;

struct nnue_trainer {
    static void train(std::string net_file, std::string qnet_file, std::vector<std::string> selfplay_directories);

    static void test_nets(std::string training_net_file, std::string quantized_net_file, std::vector<selfplay_result> &test_set);

    static void quantize_net(std::string net_file, std::string qnet_file);
};
