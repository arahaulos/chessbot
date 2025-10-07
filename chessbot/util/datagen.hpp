#pragma once

#include <string>

struct training_datagen
{
    static void datagen(std::string output_file, std::string nnue_file, int threads, int games, int depth, int nodes, bool random_opening);
};
