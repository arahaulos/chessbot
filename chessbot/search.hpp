#pragma once

#include "game.hpp"

#include "zobrist.hpp"
#include "cache.hpp"

#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

#include "state.hpp"
#include "transposition.hpp"
#include "defs.hpp"
#include "history.hpp"
#include "movepicker.hpp"

struct search_manager;

struct search_params
{
    search_params()
    {
        rfmargin_base = 0;
        rfmargin_mult = 60;
        rfmargin_improving_modifier = 88;
        fmargin_base = 95;
        fmargin_mult = 70;
        hmargin_mult = 3893;
        lmr_hist_adjust = 7079;
        cap_see_margin_mult = 120;
        quiet_see_margin_mult = 24;
        lmr_modifier = 17;
        razoring_margin = 200;
        probcut_margin = 180;

        good_quiet_treshold = -1000;
    }


    union {
        struct {
            int rfmargin_base;
            int rfmargin_mult;
            int rfmargin_improving_modifier;

            int fmargin_base;
            int fmargin_mult;

            int hmargin_mult;

            int lmr_hist_adjust;
            int cap_see_margin_mult;
            int quiet_see_margin_mult;
            int lmr_modifier;

            int razoring_margin;

            int probcut_margin;

            int good_quiet_treshold;
        };
        int data[13];
    };

    static int get_variable_index(std::string name) {
        for (int i = 0; i < 13; i++) {
            if (name == get_variable_name(i)) {
                return i;
            }
        }
        return -1;
    }

    static std::string get_variable_name(int index)
    {
        const char *str[13] = {
            "rfmargin_base",
            "rfmargin_mult",
            "rfmargin_improving_modifier",
            "fmargin_base",
            "fmargin_mult",
            "hmargin_mult",
            "lmr_hist_adjust",
            "cap_see_margin_mult",
            "quiet_see_margin_mult",
            "lmr_modifier",
            "razoring_margin",
            "probcut_margin",
            "good_quiet_treshold"
        };

        if (index >= 0 && index < 13) {
            return std::string(str[index]);
        }
        return std::string("check that fucking index");
    }

    static void get_limits(int index, int &minv, int &maxv) {
        int limits[26] = {
            0, 100,
            40, 140,
            20, 100,
            30, 150,
            30, 150,
            2000, 10000,
            4000, 10000,
            80, 200,
            10, 50,
            10, 30,
            150, 300,
            100, 300,
            -3000, 3000
        };

        if (index >= 0 && index < 13) {
            minv = limits[index*2 + 0];
            maxv = limits[index*2 + 1];
        }
    }

    static int num_of_params() {
        return sizeof(search_params) / sizeof(int);
    }
};

struct pv_table
{
    pv_table()
    {
        score = MIN_EVAL;
        num_of_moves = 0;
    }

    int32_t score;
    chess_move moves[MAX_DEPTH];
    int num_of_moves;

    void print() {
        for (int i = 0; i < num_of_moves; i++) {
            std::cout << moves[i].to_uci() << " ";
        }
        std::cout << std::endl;
    }

    void first(chess_move m) {
        moves[0] = m;
        num_of_moves = 1;
    }

    void collect(const pv_table &other) {
        if (other.num_of_moves >= MAX_DEPTH) {
            std::cout << "Error: can't collect PV. Max depth reached!" << std::endl;
            return;
        }
        if (num_of_moves == 0) {
            *this = other;
        } else {
            for (int i = 0; i < other.num_of_moves; i++) {
                moves[i+1] = other.moves[i];
            }
            num_of_moves = other.num_of_moves + 1;
        }
    }
};

struct search_statistics
{
    search_statistics() {
        reset();
    }

    void reset() {
        memset((void*)this, 0, sizeof(*this));
    }

    void add(const search_statistics &other) {
        nodes += other.nodes;
        cache_hits += other.cache_hits;
        cache_misses += other.cache_misses;

        fail_high_index += other.fail_high_index;
        fail_highs += other.fail_highs;

        eval_cache_hits += other.eval_cache_hits;
        eval_cache_misses += other.eval_cache_misses;

        max_distance_to_root = std::max(max_distance_to_root, other.max_distance_to_root);
    }

    int64_t nodes;
    int64_t cache_hits;
    int64_t cache_misses;

    int64_t eval_cache_hits;
    int64_t eval_cache_misses;

    int64_t fail_high_index;
    int64_t fail_highs;

    int max_distance_to_root;
};


struct search_context
{
    search_context(std::shared_ptr<nnue_weights> shared_weights) {
        min_nmp_ply = 0;
        for (int i = 0; i < MAX_DEPTH; i++) {
            conthist[i] = nullptr;
            moves[i] = chess_move::null_move();
            static_eval[i] = 0;
        }
        history.conthist_stack = conthist;
        history.move_stack = moves;

        if (shared_weights) {
            nnue = std::make_shared<nnue_network>(shared_weights);
        }
    }

    void presearch(const board_state &state_to_search) {
        //history.reset();
        history.reset_killers();
        stats.reset();
        min_nmp_ply = 0;

        state = state_to_search;
        if (nnue) {
            state.nnue = nnue;

            nnue->reset_acculumator_stack();
            nnue->evaluate(state);
        } else {
            state.nnue = nullptr;
        }

        history.conthist_stack = conthist;
        history.move_stack = moves;
    }

    bool main_thread;

    board_state state;
    search_statistics stats;

    piece_square_history *conthist[MAX_DEPTH];

    int min_nmp_ply;

    chess_move moves[MAX_DEPTH];
    int reduction[MAX_DEPTH];
    int32_t static_eval[MAX_DEPTH];

    history_heurestic_table history;

    move_picker move_pickers[MAX_DEPTH];

    std::shared_ptr<nnue_network> nnue;
};


class searcher
{
public:
    searcher();
    searcher(const searcher &other);

    void search(const board_state &state, std::shared_ptr<search_manager> m);

    void set_shared_weights(std::shared_ptr<nnue_weights> weights) {
        shared_nnue_weights = weights;
        for (size_t i = 0; i < thread_datas.size(); i++) {
            thread_datas[i]->nnue = std::make_shared<nnue_network>(shared_nnue_weights);
        }
    }

    void new_game();

    void set_multi_pv(int num) {
        num_of_pvs = num;
    }
    int get_multi_pv() {
        return num_of_pvs;
    }

    void clear_transposition_table();
    void clear_evaluation_cache();
    void clear_history();

    int32_t get_transpostion_table_usage_permill();

    bool test_flag;

    void set_threads(int num_of_threads);
    void set_transposition_table_size_MB(int size_MB);

    search_params sp;
private:
    uint64_t get_total_node_count();

    int32_t static_evaluation(const board_state &state, player_type_t player, search_statistics &stats);
    void start_helper_threads(int32_t window_alpha, int32_t window_beta, int depth);
    void stop_helper_threads();
    void aspirated_search(int depth);
    void search_root(int32_t window_alpha, int32_t window_beta, int depth, int thread_id);

    int32_t alphabeta(board_state &state, int32_t alpha, int32_t beta, int depth, int ply, search_context &sc, pv_table &pv, chess_move *skip_move, node_type_t expected_node_type);
    int32_t quisearch(board_state &state, int32_t alpha, int32_t beta, int depth, int ply, search_context &sc, bool is_pv);

    int number_of_helper_threads;
    int nominal_search_depth;
    uint32_t current_cache_age;

    cache<tt_bucket, TT_SIZE> transposition_table;
    cache<eval_cache_bucket, EVAL_CACHE_SIZE> eval_cache;

    std::vector<std::thread> helper_threads;
    std::vector<std::unique_ptr<search_context>> thread_datas;

    int num_of_pvs;
    pv_table root_search_pv[MAX_MULTI_PV];
    std::mutex root_search_lock;

    std::atomic<bool> searching_flag;
    std::atomic<bool> alphabeta_abort_flag;

    search_statistics all_threads_stats;

    std::vector<std::pair<chess_move, int32_t>> root_moves;

    std::shared_ptr<nnue_weights> shared_nnue_weights;
    std::shared_ptr<search_manager> manager;
};
