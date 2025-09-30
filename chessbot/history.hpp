#pragma once

#include "state.hpp"

constexpr int32_t correction_history_grain = 512;
constexpr int32_t correction_pawn_table_size = 16384;
constexpr int32_t correction_material_table_size = 2048;
constexpr int32_t correction_clamp = 150;

struct piece_square_history
{
    int16_t history[16][64];
    int32_t get(chess_move &mov)
    {
        return history[mov.get_moving_piece().d][mov.to.index];
    }

    void update(chess_move &mov, int32_t bonus)
    {
        constexpr int32_t max_history = 16384;

        int32_t clamped_bonus = std::clamp(bonus, -max_history, max_history);

        int32_t old_value = history[mov.get_moving_piece().d][mov.to.index];
        int32_t new_value = old_value + clamped_bonus - (old_value * std::abs(clamped_bonus) / max_history);

        history[mov.get_moving_piece().d][mov.to.index] = new_value;
    }
};


struct history_heurestic_table
{
    history_heurestic_table()
    {
        reset();
    }

    void reset() {
        memset((void*)this, 0, sizeof(*this));
    }

    void update_continuation_history(chess_move m, int ply, int effect)
    {
        if (ply >= 1 && conthist_stack[ply-1] != nullptr) {
            conthist_stack[ply-1]->update(m, effect);
        }
        if (ply >= 2 && conthist_stack[ply-2] != nullptr) {
            conthist_stack[ply-2]->update(m, effect);
        }
        if (ply >= 4 && conthist_stack[ply-4] != nullptr) {
            conthist_stack[ply-4]->update(m, effect);
        }
        if (ply >= 6 && conthist_stack[ply-6] != nullptr) {
            conthist_stack[ply-6]->update(m, effect);
        }

        if (ply < 4) {
            low_ply_history.update(m, effect);
        }
    }

    int32_t get_continuation_history(chess_move m, int ply)
    {
        int score = 0;
        if (ply >= 1 && conthist_stack[ply-1] != nullptr) {
            score += conthist_stack[ply-1]->get(m);
        }
        if (ply >= 2 && conthist_stack[ply-2] != nullptr) {
            score += conthist_stack[ply-2]->get(m);
        }
        if (ply >= 4 && conthist_stack[ply-4] != nullptr) {
            score += conthist_stack[ply-4]->get(m);
        }

        if (ply < 4) {
            score += low_ply_history.get(m);
        }

        return score;
    }

    void update(chess_move bm, int depth, int ply, chess_move *searched_quiets, int searched_quiet_count, chess_move *searched_captures, int searched_capture_count) {
        if (!bm.is_capture()) {
            int b = KILLER_MOVE_SLOTS-1;
            for (int i = 0; i < KILLER_MOVE_SLOTS; i++) {
                if (killer_moves[ply][i] == bm) {
                    b = i;
                    break;
                }
            }
            for (int i = b; i >= 1; i--) {
                killer_moves[ply][i] = killer_moves[ply][i-1];
            }
            killer_moves[ply][0] = bm;
        }

        if (depth < 3) {
            return;
        }

        int32_t weight = std::min(4 * depth * depth, 1024);

        if (bm.is_capture()) {
            if (searched_capture_count > 1) {
                int bonus = std::min(weight*(searched_capture_count-1), 4096);
                int malus = -bonus/(searched_capture_count-1);

                capture_history[bm.get_captured_piece().get_type()].update(bm, bonus);

                for (int i = 0; i < searched_capture_count; i++) {
                    chess_move cap = searched_captures[i];
                    if (cap != bm) {
                        capture_history[cap.get_captured_piece().get_type()].update(cap, malus);
                    }
                }
            }
        } else {
            if (searched_quiet_count > 1) {
                int scaling = std::min(searched_quiet_count-1, 8);
                int bonus = std::min(weight*scaling, 4096);
                int malus = -bonus/scaling;

                history.update(bm, bonus);
                update_continuation_history(bm, ply, bonus);

                for (int i = 0; i < searched_quiet_count; i++) {
                    chess_move m = searched_quiets[i];
                    if (m != bm) {
                        history.update(m, malus);
                        update_continuation_history(m, ply, malus);
                    }
                }
            }

            if (ply > 0) {
                chess_move previous_move = move_stack[ply-1];
                if (previous_move != chess_move::null_move()) {
                    counter_moves[previous_move.get_moving_piece().d][previous_move.to.index] = bm;
                }
            }
        }
    }


    int32_t get_capture_history(chess_move cap)
    {
        return capture_history[cap.get_captured_piece().get_type()].get(cap);
    }

    int32_t get_quiet_history(int ply, chess_move m) {
        return get_continuation_history(m, ply) + history.get(m);
    }

    int get_killers(chess_move *buffer, int ply)
    {
        int index = 0;

        chess_move previous_move = move_stack[ply-1];

        if (previous_move != chess_move::null_move()) {
            buffer[index++] = counter_moves[previous_move.get_moving_piece().d][previous_move.to.index];
        }

        for (int i = 0; i < KILLER_MOVE_SLOTS; i++) {
            buffer[index++] = killer_moves[ply][i];
            if (ply >= 2) {
                buffer[index++] = killer_moves[ply-2][i];
            }
        }

        return index;
    }

    piece_square_history *get_continuation_history_table(chess_move m)
    {
        return &continuation_history[m.is_capture()][m.get_moving_piece().d][m.to.index];
    }

    piece_square_history *get_continuation_history_table_null(player_type_t turn)
    {
        return &continuation_history[0][turn][0];
    }

    int32_t get_correction_history(const board_state &state)
    {
        int player_index = (state.get_turn() == WHITE ? 0 : 1);

        int32_t pawn_correction     = pawn_correction_history[player_index][state.pawn_hash() % correction_pawn_table_size];
        int32_t minor_correction    = minor_correction_history[player_index][state.minor_hash() % correction_pawn_table_size];
        int32_t major_correction    = major_correction_history[player_index][state.major_hash() % correction_pawn_table_size];
        int32_t material_correction = material_correction_history[player_index][state.material_hash() % correction_material_table_size];

        return (pawn_correction + material_correction + minor_correction + major_correction) / (4*correction_history_grain);
    }

    void update_correction_history(const board_state &state, int32_t correction, int depth)
    {
        int player_index = (state.get_turn() == WHITE ? 0 : 1);

        int32_t &pawn_table_entry     = pawn_correction_history[player_index][state.pawn_hash() % correction_pawn_table_size];
        int32_t &minor_table_entry    = minor_correction_history[player_index][state.minor_hash() % correction_pawn_table_size];
        int32_t &major_table_entry    = major_correction_history[player_index][state.major_hash() % correction_pawn_table_size];
        int32_t &material_table_entry = material_correction_history[player_index][state.material_hash() % correction_material_table_size];

        int32_t correction_adjust = std::clamp(correction, -correction_clamp, correction_clamp) * correction_history_grain;

        int32_t weight = std::min(depth, 16);

        pawn_table_entry     = ((correction_adjust * weight) + (pawn_table_entry     * (256 - weight))) / 256;
        minor_table_entry    = ((correction_adjust * weight) + (minor_table_entry    * (256 - weight))) / 256;
        major_table_entry    = ((correction_adjust * weight) + (major_table_entry    * (256 - weight))) / 256;
        material_table_entry = ((correction_adjust * weight) + (material_table_entry * (256 - weight))) / 256;
    }

    void reset_killers(int ply)
    {
        if (ply >= MAX_DEPTH) {
            return;
        }

        for (int i = 0; i < KILLER_MOVE_SLOTS; i++) {
            killer_moves[ply][i] = chess_move::null_move();
        }
    }

    void reset_killers()
    {
        memset(&counter_moves, 0, sizeof(counter_moves));
        memset(&killer_moves, 0, sizeof(killer_moves));
    }

    chess_move *move_stack;
    piece_square_history **conthist_stack;

    piece_square_history history;
    piece_square_history capture_history[8];
    piece_square_history continuation_history[2][16][BOARD_HEIGHT*BOARD_WIDTH];
    piece_square_history low_ply_history;

    int32_t pawn_correction_history[2][correction_pawn_table_size];
    int32_t minor_correction_history[2][correction_pawn_table_size];
    int32_t major_correction_history[2][correction_pawn_table_size];
    int32_t material_correction_history[2][correction_material_table_size];


    chess_move counter_moves[16][BOARD_HEIGHT*BOARD_WIDTH];
    chess_move killer_moves[MAX_DEPTH][KILLER_MOVE_SLOTS];

    bool test_flag;
};
