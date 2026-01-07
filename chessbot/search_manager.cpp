#include "search_manager.hpp"
#include "uci.hpp"

void search_manager::prepare_clock_search(std::shared_ptr<time_manager> tman)
{
    prepare_common();
    tm = tman;
    limit = LIMIT_CLOCK;

    clock_start_time = start_time;
}


void search_manager::prepare_fixed_depth_search(int d)
{
    prepare_common();
    depth_limit = d;
    limit = LIMIT_DEPTH;
}


void search_manager::prepare_fixed_nodes_search(int n)
{
    prepare_common();
    nodes_limit = n;
    limit = LIMIT_NODES;
}

void search_manager::prepare_depth_nodes_search(int d, int n)
{
    prepare_common();
    nodes_limit = n;
    depth_limit = d;
    limit = LIMIT_DEPTH_NODES;
}


void search_manager::prepare_fixed_time_search(int time_ms)
{
    prepare_common();
    time_limit = time_ms;
    limit = LIMIT_TIME;
}

void search_manager::prepare_infinite_search()
{
    prepare_common();
    limit = LIMIT_INFINITE;
}

void search_manager::start_clock(std::shared_ptr<time_manager> tman)
{
    tm = tman;
    clock_start_time = get_timestamp();
    limit = LIMIT_CLOCK;

    if (get_depth() == MAX_DEPTH) {
        stop_search();
    }
}

void search_manager::stop_search()
{
    if (uci && !ready_flag) {
        uci->search_end();
    }
    ready_flag = true;
}


bool search_manager::on_end_of_iteration(int d, int seldepth, uint64_t n, pv_table *pv, int num_of_pvs)
{
    depth = d;
    nodes = n;

    uint64_t timestamp = get_timestamp();

    search_result r;
    r.nodes = nodes;
    r.time_ms = timestamp - start_time;
    r.depth = depth;
    r.seldepth = seldepth;
    for (int i = 0; i < num_of_pvs; i++) {
        r.lines.push_back(std::make_shared<pv_table>(pv[i]));
    }
    results_lock.lock();
    results.push_back(r);
    results_lock.unlock();

    if (uci) {
        uci->iteration_end();
    }

    if (limit != LIMIT_INFINITE && depth == MAX_DEPTH && !ready_flag) {
        stop_search();
        return ready_flag;
    }

    if (limit == LIMIT_CLOCK) {
        if (tm->end_of_iteration(depth, pv[0].moves[0], pv[0].score, timestamp - clock_start_time)) {
            stop_search();
        }
        return ready_flag;
    } else {
        return on_search_stop_control(nodes);
    }
}


bool search_manager::on_search_stop_control(uint64_t n)
{
    nodes = n;

    switch (limit) {
        case LIMIT_CLOCK:
            if (tm->should_stop_search(get_timestamp() - clock_start_time)) {
                stop_search();
            }
            break;
        case LIMIT_DEPTH:
            if (depth >= depth_limit) {
                stop_search();
            }
            break;
        case LIMIT_DEPTH_NODES:
            if (nodes >= nodes_limit && depth >= depth_limit) {
                stop_search();
            }
            break;
        case LIMIT_INFINITE:
            break;
        case LIMIT_NODES:
            if (nodes >= nodes_limit) {
                stop_search();
            }
            break;
        case LIMIT_TIME:
            if (get_timestamp() - start_time >= time_limit) {
                stop_search();
            }
            break;
    };

    return ready_flag;
}

void search_manager::prepare_common()
{
    start_time = get_timestamp();
    ready_flag = false;
    results.clear();
    nodes = 0;
    depth = 0;
}

uint64_t search_manager::get_timestamp()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
}
