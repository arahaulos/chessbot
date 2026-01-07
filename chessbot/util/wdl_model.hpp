#pragma once

#include <string>
#include "../state.hpp"

typedef std::array<double, 8> win_rate_model_params;

namespace wdl_model
{
    void fit_model(std::string dataset_file);

    void get_wdl(const board_state &state, int32_t search_score, float &win_p, float &draw_p, float &loss_p);
}
