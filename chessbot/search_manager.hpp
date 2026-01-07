#pragma once

#include <vector>
#include <memory>
#include <mutex>

#include "search.hpp"
#include "time_manager.hpp"

struct uci_interface;

struct search_result
{
    int depth;
    int seldepth;

    std::vector<std::shared_ptr<pv_table>> lines;
    uint64_t time_ms;
    uint64_t nodes;
};

enum search_limit_type_t {LIMIT_CLOCK, LIMIT_DEPTH, LIMIT_NODES, LIMIT_TIME, LIMIT_INFINITE, LIMIT_DEPTH_NODES};

struct search_manager
{
    void set_uci(uci_interface *ui)
    {
        uci = ui;
    }

    void prepare_clock_search(std::shared_ptr<time_manager> tman);
    void prepare_fixed_depth_search(int d);
    void prepare_fixed_nodes_search(int n);
    void prepare_depth_nodes_search(int d, int n);
    void prepare_fixed_time_search(int time_ms);
    void prepare_infinite_search();

    void start_clock(std::shared_ptr<time_manager> tman);

    void stop_search();

    bool on_end_of_iteration(int depth, int seldepth, uint64_t nodes, pv_table *pv, int num_of_pvs);
    bool on_search_stop_control(uint64_t nodes);

    uint64_t get_node_count() {
        std::lock_guard<std::mutex> guard(results_lock);

        if (results.size() > 0) {
            return results.back().nodes;
        }
        return 0;
    }

    int get_max_depth_reached() {
        std::lock_guard<std::mutex> guard(results_lock);

        if (results.size() > 0) {
            return results.back().seldepth;
        }
        return 0;
    }

    chess_move get_move(int pv_num)
    {
        std::lock_guard<std::mutex> guard(results_lock);

        if (results.size() > 0) {
            return results.back().lines[pv_num]->moves[0];
        }
        return chess_move::null_move();
    }

    int32_t get_evaluation(int pv_num)
    {
        std::lock_guard<std::mutex> guard(results_lock);

        if (results.size() > 0) {
            return results.back().lines[pv_num]->score;
        }
        return 0;
    }

    void get_pv(pv_table &pv_out, int pv_num)
    {
        std::lock_guard<std::mutex> guard(results_lock);

        if (results.size() > 0) {
            pv_out = *results.back().lines[pv_num];
        }
    }

    int32_t get_depth()
    {
        std::lock_guard<std::mutex> guard(results_lock);

        if (results.size() > 0) {
            return results.back().depth;
        }
        return 0;
    }

    search_result get_search_result(int depth)
    {
        std::lock_guard<std::mutex> guard(results_lock);

        return results[depth-1];
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

    uint64_t get_nps()
    {
        std::lock_guard<std::mutex> guard(results_lock);

        uint64_t ptime = 0;
        uint64_t pnodes = 0;
        if (results.size() > 1) {
            ptime = results[results.size()-2].time_ms;
            pnodes = results[results.size()-2].nodes;
        }

        uint64_t dtime = results.back().time_ms - ptime;
        uint64_t dnodes = results.back().nodes - pnodes;

        if (dtime > 0) {
            return (dnodes*1000)/dtime;
        } else {
            return dnodes*1000;
        }
    }

    std::shared_ptr<time_manager> tm;
private:
    void prepare_common();
    uint64_t get_timestamp();

    uint64_t start_time;

    std::atomic<bool> ready_flag;
    std::vector<search_result> results;
    std::mutex results_lock;

    search_limit_type_t limit;

    int nodes_limit;
    int depth_limit;
    int time_limit;

    int depth;
    uint64_t nodes;
    uint64_t clock_start_time;

    uci_interface *uci;
};
