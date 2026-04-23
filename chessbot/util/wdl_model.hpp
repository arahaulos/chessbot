#pragma once

#include <string>
#include "../state.hpp"

namespace wdl_model
{
    void fit_model(std::string dataset_folder);

    void get_wdl(const board_state &state, int32_t search_score, float &win_p, float &draw_p, float &loss_p);

    int32_t normalize_score(const board_state &state, int32_t search_score);
}
