#pragma once

#include <stdint.h>
#include "../../defs.hpp"
#include "../../bitboard.hpp"

enum training_position_flags {WHITE_TURN = 0x1, RESULT_STM_WIN = 0x1 << 1, RESULT_DRAW = 0x1 << 2};


class board_state;
struct selfplay_result;

struct training_position
{
    training_position() {};

    training_position(const selfplay_result &spr);
    training_position(const board_state &state, const selfplay_result &spr);


    uint8_t iterate_pieces(uint64_t &occ, int &index, int &sq_index) const
    {
        uint8_t p;
        if ((index & 0x1) == 0) {
            p = packed_pieces[index/2] & 0xF;
        } else {
            p = (packed_pieces[index/2] >> 4) & 0xF;
        }

        sq_index = bit_scan_forward_clear(occ);

        index += 1;

        return p;
    }

    int32_t count_pieces() const {
        return pop_count(occupation);
    }


    player_type_t get_turn() const {
        if ((flags & WHITE_TURN) != 0) {
            return WHITE;
        } else {
            return BLACK;
        }
    }

    float get_wdl_relative_to_white() const {
        if (get_turn() == BLACK) {
            return 1.0f - get_wdl_relative_to_stm();
        } else {
            return get_wdl_relative_to_stm();
        }
    }
    float get_wdl_relative_to_stm() const {
        if ((flags & RESULT_STM_WIN) != 0) {
            return 1.0f;
        } else if ((flags & RESULT_DRAW) != 0) {
            return 0.5f;
        } else {
            return 0.0f;
        }
    }
    int32_t get_eval() const {
        return eval;
    }

    bool operator == (const training_position &other) const {
        if (occupation != other.occupation ||
            eval != other.eval ||
            flags != other.flags) {
            return false;
        }

        for (int i = 0; i < 16; i++) {
            if (packed_pieces[i] != other.packed_pieces[i]) {
                return false;
            }
        }
        return true;
    }

    bool operator != (const training_position &other) const {
        return !(*this == other);
    }

    uint64_t occupation;
    int16_t eval;
    uint8_t flags;
    uint8_t packed_pieces[16];

private:
    void init(const board_state &state, float wdl, int32_t score);
};
