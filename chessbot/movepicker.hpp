#pragma once

#include "history.hpp"
#include "see.hpp"
#include "movegen.hpp"

enum picked_move_types {MOVES_END = 0,
                        TT_MOVE = 1,
                        GOOD_CAPTURE_MOVE = 2,
                        PROMOTION_MOVE = 3,
                        KILLER_MOVE = 4,
                        GOOD_QUIET_MOVE = 5,
                        BAD_CAPTURE_MOVE = 6,
                        BAD_QUIET_MOVE = 7};

constexpr int GOOD_CAPTURE_ARRAY_SIZE = 20;
constexpr int GOOD_NON_KILLER_ARRAY_SIZE = 40;
constexpr int KILLER_ARRAY_SIZE = 10;

constexpr int MAX_CAPTURE_MOVES = 80;
constexpr int MAX_QUIET_MOVES = 240;
constexpr int MAX_PROMOTION_MOVES = 24;


struct scored_move
{
    chess_move mov;
    int32_t score;
};

template <int N>
struct scored_move_array
{
    scored_move_array() {
        num_of_moves = 0;
        moves_left = 0;
    }

    void clear()
    {
        num_of_moves = 0;
        moves_left = 0;
    }

    bool is_full() const {
        return (num_of_moves >= N);
    }

    bool contains(const chess_move &mov) {
        for (int i = 0; i < num_of_moves; i++) {
            if (mov == moves[i].mov) {
                return true;
            }
        }
        return false;
    }

    bool pick(chess_move &mov, int32_t &score) {
        if (moves_left <= 0) {
            return false;
        }
        int32_t best_score = moves[0].score;
        int best_index = 0;
        for (int i = 1; i < moves_left; i++) {
            if (moves[i].score > best_score) {
                best_index = i;
                best_score = moves[i].score;
            }
        }

        mov = moves[best_index].mov;
        score = best_score;

        std::swap(moves[best_index], moves[moves_left-1]);
        moves_left--;
        return true;
    }

    bool pick(chess_move &mov) {
        int32_t tmp;
        return pick(mov, tmp);
    }

    void add(const chess_move &mov, const int32_t &score) {
        moves[num_of_moves].mov = mov;
        moves[num_of_moves].score = score;
        num_of_moves++;
        moves_left++;
    }

    scored_move moves[N];
    int num_of_moves;
    int moves_left;
};


struct move_picker
{
    void init(board_state &state_a, int ply_a, chess_move tt_move_a, chess_move threat_move_a, history_heurestic_table &history_table_a, bool test_flag_a) {
        history_table = &history_table_a;
        state = &state_a;
        ply = ply_a;

        test_flag = test_flag_a;

        threat_move = threat_move_a;
        tt_move = tt_move_a;

        tt_move_picked = false;
        tt_move_valid = false;

        if (tt_move.valid()) {
            tt_move_valid = move_generator::is_move_valid(*state, tt_move_a);
            if (!tt_move_valid) {
                std::cout << "TT move not valid!" << std::endl;
            }
        }

        captures_generated = false;
        quiets_generated = false;
        promotions_generated = false;
        killers_generated = false;

        picked_quiet_count = 0;
        picked_capture_count = 0;

        legal_moves = 0;

        skip_quiets_flag = false;

        stage = TT_MOVE;

        winning_captures.clear();
        losing_captures.clear();
        killers.clear();
        good_non_killers.clear();
        bad_non_killers.clear();
        promotions.clear();
    }

    void skip_quiets()
    {
        skip_quiets_flag = true;
    }

    int32_t get_threats_score(chess_move mov)
    {
        constexpr int32_t minor_threat_value = 6000;
        constexpr int32_t major_threat_value = 12000;

        constexpr int32_t null_move_escape_value = 8000;

        constexpr int32_t king_attack_bonus = 16000;
        constexpr int32_t attack_bonus = 8000;

        piece moving = state->get_square(mov.from);

        int32_t score = 0;

        if (moving.get_type() == BISHOP || moving.get_type() == KNIGHT) {
            if (state->is_square_threatened_by_pawn(mov.from, moving.get_player())) {
                score += minor_threat_value;
            }
            if (state->is_square_threatened_by_pawn(mov.to, moving.get_player())) {
                score -= minor_threat_value;
            }
        } else if (moving.get_type() == ROOK || moving.get_type() == QUEEN) {
            if (state->is_square_threatened_by_pawn(mov.from, moving.get_player()) ||
                state->is_square_threatened_by_minor(mov.from, moving.get_player())) {
                score += major_threat_value;
            }
            if (state->is_square_threatened_by_pawn(mov.to, moving.get_player()) ||
                state->is_square_threatened_by_minor(mov.to, moving.get_player())) {
                score -= major_threat_value;
            }
        }


        if (score >= 0) {
            uint_fast8_t opponent_color = (moving.get_player() == WHITE);

            bitboard attack = move_generator::piece_attack(*state, moving, mov.to);
            bitboard king_sq = (uint64_t)0x1 << (moving.get_player() == WHITE ? state->black_king_square.index : state->white_king_square.index);

            //Attacking greater or equal value pieces without getting en prise
            bitboard targets = 0;
            if (moving.get_type() == PAWN) {
                targets = state->bitboards[KNIGHT][opponent_color] | state->bitboards[ROOK][opponent_color];
            } else if (moving.get_type() == KNIGHT) {
                targets = state->bitboards[BISHOP][opponent_color] | state->bitboards[ROOK][opponent_color] | state->bitboards[QUEEN][opponent_color];
            } else if (moving.get_type() == BISHOP) {
                targets = state->bitboards[KNIGHT][opponent_color] | state->bitboards[ROOK][opponent_color];
            }
            if ((attack & targets) != 0 && !state->is_square_threatened(mov.to, moving.get_player())) {
                //Extra bonus for forks
                score += pop_count(attack & (targets | king_sq))*attack_bonus;
            }

            //Attacking king
            if ((attack & king_sq) != 0) {
                score += king_attack_bonus;
            }
        }
        if (threat_move.valid()) {
            if (mov.from == threat_move.to) {
                //Escaping capture which caused cut-off in nullmove pruning search
                score += null_move_escape_value;
            }
        }

        return score;
    }

    void add_capture_moves(chess_move *moves, int move_count) {
        for (int i = 0; i < move_count; i++) {
            chess_move m = moves[i];

            if (m == tt_move) {
                continue;
            }

            int32_t see = static_exchange_evaluation(*state, m);
            int32_t score = see + (history_table->get_capture_history(m) / 32);

            if (score >= 0 && !winning_captures.is_full()) {
                winning_captures.add(m, score);
            } else {
                losing_captures.add(m, score);
            }
        }
    }

    void add_killer_moves(chess_move *moves, int num_of_moves)
    {
        for (int i = 0; i < num_of_moves; i++) {
            chess_move m = moves[i];
            if (m == tt_move) {
                continue;
            }
            int32_t score = history_table->get_quiet_history(ply, m) + get_threats_score(m);

            killers.add(m, score);
        }
    }

    void add_quiet_moves(chess_move *moves, int num_of_moves) {
        constexpr int32_t good_quiet_treshold = 3000;

        for (int i = 0; i < num_of_moves; i++) {
            chess_move m = moves[i];
            if (m == tt_move || killers.contains(m)) {
                continue;
            }

            int32_t score = history_table->get_quiet_history(ply, m) + get_threats_score(m);

            if (score > good_quiet_treshold && !good_non_killers.is_full()) {
                good_non_killers.add(m, score);
            } else {
                bad_non_killers.add(m, score);
            }
        }
    }

    void add_promotion_moves(chess_move *moves, int num_of_moves) {
        for (int i = 0; i < num_of_moves; i++) {
            chess_move m = moves[i];
            if (m == tt_move) {
                continue;
            }
            int32_t score = history_table->get_quiet_history(ply, m);

            promotions.add(m, score);
        }
    }



    void previus_pick_was_skipped(const chess_move &skipped_move)
    {
        if (skipped_move.is_capture()) {
            picked_capture_count--;
        } else {
            picked_quiet_count--;
        }
    }

    int pick_pseudo_legal(chess_move &m, int32_t &score) {
        switch (stage) {
            case TT_MOVE:

                if (!tt_move_picked && tt_move_valid) {
                    score = 32000;
                    tt_move_picked = true;
                    m = tt_move;
                    return TT_MOVE;
                }
                stage = GOOD_CAPTURE_MOVE;

            case GOOD_CAPTURE_MOVE:

                if (!captures_generated) {
                    chess_move movelist[80];
                    int num_of_moves = move_generator::generate_capture_moves(*state, state->get_turn(), movelist);
                    add_capture_moves(movelist, num_of_moves);

                    captures_generated = true;
                }
                if (winning_captures.pick(m, score)) {
                    return GOOD_CAPTURE_MOVE;
                }
                stage = PROMOTION_MOVE;

            case PROMOTION_MOVE:

                if (!promotions_generated) {
                    chess_move movelist[16];
                    int num_of_moves = move_generator::generate_promotion_moves(*state, state->get_turn(), movelist);
                    add_promotion_moves(movelist, num_of_moves);

                    promotions_generated = true;
                }
                if (promotions.pick(m, score)) {
                    return PROMOTION_MOVE;
                }
                stage = KILLER_MOVE;

            case KILLER_MOVE:

                if (!killers_generated) {
                    chess_move movelist[16];
                    int num_of_moves = move_generator::generate_killer_moves(*state, *history_table, ply, movelist);
                    add_killer_moves(movelist, num_of_moves);

                    killers_generated = true;
                }
                if (killers.pick(m, score)) {
                    return KILLER_MOVE;
                }
                stage = GOOD_QUIET_MOVE;

            case GOOD_QUIET_MOVE:
                if (!quiets_generated && !skip_quiets_flag) {
                    chess_move movelist[240];
                    int num_of_moves = move_generator::generate_quiet_moves(*state, state->get_turn(), movelist);
                    add_quiet_moves(movelist, num_of_moves);

                    quiets_generated = true;
                }
                if (good_non_killers.pick(m, score) && !skip_quiets_flag) {
                    return GOOD_QUIET_MOVE;
                }

                stage = BAD_CAPTURE_MOVE;
            case BAD_CAPTURE_MOVE:
                if (losing_captures.pick(m, score)) {
                    return BAD_CAPTURE_MOVE;
                }

                stage = BAD_QUIET_MOVE;

            case BAD_QUIET_MOVE:
                if (bad_non_killers.pick(m, score) && !skip_quiets_flag) {
                    return BAD_QUIET_MOVE;
                }
        }

        return MOVES_END;
    }

    int pick(chess_move &m) {
        while (true) {
            int type = pick_pseudo_legal(m, previus_pick_score);
            if (type == MOVES_END) {
                return type;
            }
            if (!state->causes_check(m, state->get_turn())) {
                if (m.is_capture()) {
                    picked_captures[picked_capture_count] = m;
                    picked_capture_count++;
                } else {
                    picked_quiet_moves[picked_quiet_count] = m;
                    picked_quiet_count++;
                }
                legal_moves++;
                return type;
            }
        }
    }

    chess_move picked_quiet_moves[MAX_QUIET_MOVES];
    chess_move picked_captures[MAX_CAPTURE_MOVES];

    int picked_quiet_count;
    int picked_capture_count;

    int legal_moves;

    int32_t previus_pick_score;

private:
    chess_move threat_move;

    int stage;

    board_state *state;
    history_heurestic_table *history_table;
    int ply;

    bool captures_generated;
    bool quiets_generated;
    bool promotions_generated;
    bool killers_generated;

    chess_move tt_move;

    scored_move_array<GOOD_CAPTURE_ARRAY_SIZE>    winning_captures;
    scored_move_array<MAX_CAPTURE_MOVES>          losing_captures;

    scored_move_array<KILLER_ARRAY_SIZE>          killers;
    scored_move_array<GOOD_NON_KILLER_ARRAY_SIZE> good_non_killers;
    scored_move_array<MAX_QUIET_MOVES>            bad_non_killers;
    scored_move_array<MAX_PROMOTION_MOVES>        promotions;

    bool tt_move_valid;
    bool tt_move_picked;

    bool skip_quiets_flag;

    bool test_flag;
};
