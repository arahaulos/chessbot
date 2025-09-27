#pragma once


#include "game.hpp"

#include "zobrist.hpp"
#include "cache.hpp"

#include <memory>
#include <thread>
#include <mutex>
#include <map>
#include <atomic>
#include <chrono>
#include "state.hpp"
#include "opening_book.hpp"
#include "transposition.hpp"
#include "defs.hpp"
#include "history.hpp"
#include "movepicker.hpp"
#include "time_managment.hpp"

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
        see_margin_mult = 120;
        lmr_modifier = 17;

        razoring_margin = 200;

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
            int see_margin_mult;
            int lmr_modifier;

            int razoring_margin;

            int good_quiet_treshold;
        };
        int data[11];
    };

    static int get_variable_index(std::string name) {
        for (int i = 0; i < 11; i++) {
            if (name == get_variable_name(i)) {
                return i;
            }
        }
        return -1;
    }

    static std::string get_variable_name(int index)
    {
        const char *str[11] = {
            "rfmargin_base",
            "rfmargin_mult",
            "rfmargin_improving_modifier",
            "fmargin_base",
            "fmargin_mult",
            "hmargin_mult",
            "lmr_hist_adjust",
            "see_margin_mult",
            "lmr_modifier",
            "razoring_margin",
            "good_quiet_treshold"
        };

        if (index >= 0 && index < 11) {
            return std::string(str[index]);
        }
        return std::string("check that fucking index");
    }

    static void get_limits(int index, int &minv, int &maxv) {
        int limits[22] = {
            0, 100,
            40, 140,
            20, 100,
            30, 150,
            30, 150,
            2000, 10000,
            4000, 10000,
            80, 200,
            10, 30,
            150, 300,
            -3000, 3000
        };

        if (index >= 0 && index < 11) {
            minv = limits[index*2 + 0];
            maxv = limits[index*2 + 1];
        }
    }

    static int num_of_params() {
        return sizeof(search_params) / sizeof(int);
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

        history_score_samples += other.history_score_samples;

        for (int i = 0; i < 32; i++) {
            history_buckets[i] += other.history_buckets[i];
        }

        max_distance_to_root = std::max(max_distance_to_root, other.max_distance_to_root);
    }

    void add_history_sample(int64_t score)
    {
        int min_hist = -16000;
        int max_hist = 16000;

        int step = (max_hist - min_hist)/32;

        int treshold = min_hist + step;
        for (int i = 0; i < 32; i++) {
            if (score < treshold) {
                history_buckets[i] += 1;
                return;
            }
            treshold = treshold + step;
        }
        history_buckets[31] += 1;
    }

    void print_history_distribution()
    {
        int min_hist = -16000;
        int max_hist = 16000;

        int step = (max_hist - min_hist)/32;

        std::cout << step << std::endl;

        int treshold = min_hist + step;

        int64_t total_samples = 0;
        for (int i = 0; i < 32; i++) {
            total_samples += history_buckets[i];
        }

        for (int i = 0; i < 32; i++) {
            std::cout << " < " << treshold << ": " << (double)history_buckets[i]*100 / total_samples << "%" << std::endl;
            treshold = treshold + step;
        }
    }


    int64_t nodes;
    int64_t cache_hits;
    int64_t cache_misses;

    int64_t eval_cache_hits;
    int64_t eval_cache_misses;

    int64_t fail_high_index;
    int64_t fail_highs;

    int64_t history_score_samples;
    int64_t history_buckets[32];

    int max_distance_to_root;
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

struct search_info
{
    int depth;
    int seldepth;

    std::vector<std::shared_ptr<pv_table>> lines;
    uint64_t time_ms;
    uint64_t nodes;
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

    void presearch(board_state &state_to_search) {
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


class alphabeta_search
{
public:
    alphabeta_search();
    ~alphabeta_search();
    alphabeta_search(const alphabeta_search &other);

    void load_nnue_net(std::string path)
    {
        shared_nnue_weights->load(path);
    }

    void set_shared_weights(std::shared_ptr<nnue_weights> weights) {
        shared_nnue_weights = weights;
        for (size_t i = 0; i < thread_datas.size(); i++) {
            thread_datas[i]->nnue = std::make_shared<nnue_network>(shared_nnue_weights);
        }
    }

    void set_nnue_rescaling(int f0, int f1) {
        shared_nnue_weights->rescale_factor0 = f0;
        shared_nnue_weights->rescale_factor1 = f1;
    }

    uint64_t get_search_time_ms();


    void new_game();


    void begin_search(const board_state &state, std::shared_ptr<time_manager> tman);

    void begin_search(const board_state &state) {
        begin_search(state, nullptr);
    }

    void stop_search();
    void set_multi_pv(int num) {
        num_of_pvs = num;
    }
    int get_multi_pv() {
        return num_of_pvs;
    }

    void clear_transposition_table();
    void clear_evaluation_cache();

    int32_t get_transpostion_table_usage_permill();

    uint64_t get_node_count() {
        int n = 0;
        info_lock.lock();
        if (search_infos.size() > 0) {
            n = search_infos.back().nodes;
        }
        info_lock.unlock();
        return n;
    }

    int get_max_depth_reached() {
        int sd = 0;
        info_lock.lock();
        if (search_infos.size() > 0) {
            sd = search_infos.back().seldepth;
        }
        info_lock.unlock();
        return sd;
    }

    chess_move get_move(int pv_num)
    {
        chess_move p = chess_move::null_move();
        info_lock.lock();
        if (search_infos.size() > 0) {
            p = search_infos.back().lines[pv_num]->moves[0];
        }
        info_lock.unlock();
        return p;
    }

    int32_t get_evaluation(int pv_num)
    {
        int32_t e = 0;
        info_lock.lock();
        if (search_infos.size() > 0) {
            e = search_infos.back().lines[pv_num]->score;
        }
        info_lock.unlock();
        return e;
    }

    void get_pv(pv_table &pv_out, int pv_num)
    {
        info_lock.lock();
        if (search_infos.size() > 0) {
            pv_out = *search_infos.back().lines[pv_num];
        }
        info_lock.unlock();
    }

    int32_t get_depth()
    {
        info_lock.lock();
        int depth = 0;
        if (search_infos.size() > 0) {
            depth = search_infos.back().depth;
        }
        info_lock.unlock();
        return depth;
    }

    void limit_search(int depth, int nodes)
    {
        min_depth = depth;
        limit_nodes = nodes;
    }


    chess_move get_move() {
        return get_move(0);
    }
    int32_t get_evaluation() {
        return get_evaluation(0);
    }
    void get_pv(pv_table &pv_out)  {
        get_pv(pv_out, 0);
    }

    uint64_t get_nps() {
        return nps;
    }

    bool is_searching() {
        return searching_flag;
    }
    bool is_ready_to_stop() {
        return ready_flag;
    }

    search_info get_search_info(int depth)
    {
        search_info info;
        info_lock.lock();
        info = search_infos[depth-1];
        info_lock.unlock();
        return info;
    }

    bool test_flag;
    bool use_opening_book;

    chess_move fast_search(board_state &state, int depth, int max_nodes);
    void abort_fast_search()  {
        searching_flag = false;
        alphabeta_abort_flag = true;
    }
    chess_move search_time(board_state &state, uint64_t max_time_ms, std::shared_ptr<time_manager> tman);

    void set_threads(int num_of_threads);
    void set_transposition_table_size_MB(int size_MB);

    search_params sp;
private:
    int32_t static_evaluation(const board_state &state, player_type_t player, search_statistics &stats);

    void iterative_search(int max_depth);
    void start_helper_threads(int32_t window_alpha, int32_t window_beta, int depth);
    void stop_helper_threads();
    void aspirated_search(int depth);
    void search_root(int32_t window_alpha, int32_t window_beta, int depth, int thread_id);

    int32_t alphabeta(board_state &state, int32_t alpha, int32_t beta, int depth, int ply, search_context &sc, pv_table &pv, chess_move *skip_move, node_type_t expected_node_type);
    int32_t quisearch(board_state &state, int32_t alpha, int32_t beta, int depth, int ply, search_context &sc, bool is_pv);


    int number_of_helper_threads;
    int nominal_search_depth;
    uint32_t current_cache_age;

    uint64_t nps;

    std::chrono::high_resolution_clock::time_point search_begin_time;

    cache<tt_bucket, TT_SIZE> transposition_table;
    cache<eval_cache_bucket, EVAL_CACHE_SIZE> eval_cache;

    opening_book book;
    board_state state_to_search;


    std::thread search_main_thread;
    std::vector<std::thread> helper_threads;
    std::vector<std::unique_ptr<search_context>> thread_datas;

    int num_of_pvs;
    pv_table root_search_pv[MAX_MULTI_PV];
    std::mutex root_search_lock;

    std::atomic<bool> searching_flag;
    std::atomic<bool> alphabeta_abort_flag;
    std::atomic<bool> ready_flag;

    search_statistics all_threads_stats;

    std::vector<std::pair<chess_move, int32_t>> root_moves;

    std::shared_ptr<nnue_weights> shared_nnue_weights;
    std::shared_ptr<time_manager> time_man;

    int limit_nodes;
    int min_depth;

    std::mutex info_lock;
    std::vector<search_info> search_infos;
};


std::string eval_to_str(int32_t score);

