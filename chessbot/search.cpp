#include "search.hpp"
#include <algorithm>
#include <float.h>
#include <iostream>
#include <memory>
#include <thread>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "movepicker.hpp"
#include "movegen.hpp"
#include "sstream"

#include <array>

#include "zobrist.hpp"

std::string eval_to_str(int32_t score)
{
    std::stringstream ss;
    if (is_mate_score(score)) {
        if (score < 0) {
            ss << "-";
        }
        ss << "M";
        ss << get_mate_distance(score);
    } else {
        ss << ((float)score)/100;
    }
    return ss.str();
}

alphabeta_search::alphabeta_search()
{
    clear_transposition_table();
    clear_evaluation_cache();

    searching_flag = false;
    alphabeta_abort_flag = false;

    test_flag = true;
    use_opening_book = true;

    current_cache_age = 0;
    limit_nodes = 0;
    num_of_pvs = 1;

    shared_nnue_weights = std::make_shared<nnue_weights>();

    set_threads(DEFAULT_THREADS);
}

alphabeta_search::alphabeta_search(const alphabeta_search &other): transposition_table(other.transposition_table), eval_cache(other.eval_cache)
{
    shared_nnue_weights = other.shared_nnue_weights;

    searching_flag = false;
    alphabeta_abort_flag = false;
    test_flag = other.test_flag;
    use_opening_book = other.use_opening_book;
    current_cache_age = other.current_cache_age;
    limit_nodes = 0;
    num_of_pvs = other.num_of_pvs;

    sp = other.sp;

    set_threads(other.number_of_helper_threads + 1);

    for (int i = 0; i < number_of_helper_threads+1; i++) {
        thread_datas[i]->history = other.thread_datas[i]->history;
    }
}


int32_t alphabeta_search::get_transpostion_table_usage_permill()
{
    uint64_t slots_used = 0;
    for (uint64_t i = 0; i < transposition_table.get_size(); i += 10) {
        slots_used += transposition_table[i].used(current_cache_age);
    }
    return (slots_used*10000) / (transposition_table.get_size()*TT_ENTRIES_IN_BUCKET);
}

alphabeta_search::~alphabeta_search()
{
    if (searching_flag) {
        stop_search();
    }
}

void alphabeta_search::clear_transposition_table()
{
    for (uint64_t i = 0; i < transposition_table.get_size(); i++) {
        transposition_table[i].clear();
    }
}

void alphabeta_search::clear_evaluation_cache()
{
    for (uint64_t i = 0; i < eval_cache.get_size(); i++) {
        eval_cache[i].clear();
    }
}

void alphabeta_search::set_threads(int num_of_threads)
{
    if (!is_searching()) {
        number_of_helper_threads = std::clamp(num_of_threads-1, 0, MAX_THREADS);

        thread_datas.clear();
        for (int i = 0; i < number_of_helper_threads+1; i++) {
            thread_datas.push_back(std::make_unique<search_context>(shared_nnue_weights));
        }
    }
}

void alphabeta_search::set_transposition_table_size_MB(int size_MB)
{
    if (!is_searching()) {
        transposition_table.resize(size_MB);
        clear_transposition_table();
    }
}


uint64_t alphabeta_search::get_search_time_ms()
{
    if (!searching_flag) {
        return 0;
    }
    std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( t - search_begin_time );

    return ms.count();
}

void alphabeta_search::new_game()
{
    current_cache_age += 4;

    for (size_t i = 0; i < thread_datas.size(); i++) {
        thread_datas[i]->history.reset();
    }
}


void alphabeta_search::stop_search()
{
    searching_flag = false;
    alphabeta_abort_flag = true;
    search_main_thread.join();
    time_man = nullptr;

    limit_nodes = 0;
    min_depth = 0;
}

void alphabeta_search::begin_search(const board_state &state, std::shared_ptr<time_manager> tman)
{
    if (searching_flag) {
        return;
    }

    time_man = tman;

    state_to_search = state;
    searching_flag = true;
    alphabeta_abort_flag = false;
    ready_flag = false;

    search_infos.clear();

    search_begin_time = std::chrono::high_resolution_clock::now();

    search_main_thread = std::thread(&alphabeta_search::iterative_search, this, MAX_DEPTH);

}



chess_move alphabeta_search::fast_search(board_state &state, int depth, int nodes)
{
    state_to_search = state;
    searching_flag = true;

    if (nodes == 0) {
        nodes = 1;
    }

    //search_begin_time = std::chrono::high_resolution_clock::now();

    int restore_helper_threads = number_of_helper_threads;
    number_of_helper_threads = 0;

    int restore_limit_nodes = limit_nodes;
    int restore_min_depth = min_depth;

    limit_nodes = nodes;
    min_depth = depth;

    iterative_search(MAX_DEPTH);
    number_of_helper_threads = restore_helper_threads;

    limit_nodes = restore_limit_nodes;
    min_depth = restore_min_depth;
    searching_flag = false;

    return search_infos.back().lines[0]->moves[0];
}


chess_move alphabeta_search::search_time(board_state &state, uint64_t time_ms, std::shared_ptr<time_manager> tman)
{
    begin_search(state, tman);

    while ((get_search_time_ms() < time_ms || get_depth() < 3) && !is_ready_to_stop()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stop_search();

    return get_move();
}


void alphabeta_search::iterative_search(int max_depth)
{
    for (int i = 0; i < number_of_helper_threads+1; i++) {
        thread_datas[i]->presearch(state_to_search);
    }

    if (time_man) {
        time_man->test_flag = test_flag;
    }

    nps = 0;

    search_infos.clear();

    current_cache_age += 1;
    if (current_cache_age > 500) {
        clear_transposition_table();
        current_cache_age = 0;
    }

    std::vector<chess_move> legal_moves = state_to_search.get_all_legal_moves(state_to_search.get_turn());

    chess_move book_move = book.get_book_move(state_to_search);

    if ((book_move.valid() && use_opening_book) /*|| legal_moves.size() == 1*/) {
        info_lock.lock();

        search_info info;
        info.depth = 1;
        info.seldepth = 1;
        info.nodes = 1;
        info.time_ms = get_search_time_ms();
        for (int i = 0; i < num_of_pvs; i++) {
            info.lines.push_back(std::make_shared<pv_table>());
        }
        info.lines[0]->first((book_move.valid() ? book_move : legal_moves[0]));

        search_infos.push_back(info);

        info_lock.unlock();

        ready_flag = true;

        return;
    } else if (legal_moves.size() == 0) {
        std::cout << "No legal moves!!" << std::endl;
        searching_flag = false;
        return;
    }

    root_moves.clear();
    for (auto it = legal_moves.begin(); it != legal_moves.end(); it++) {
        root_moves.push_back(std::pair(*it, 0));
    }

    uint64_t total_nodes_searched = 0;
    int depth = 1;
    while (depth <= max_depth) {
        all_threads_stats.reset();
        all_threads_stats.nodes = total_nodes_searched;

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        aspirated_search(depth);

        std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( t3 - t2 );

        uint64_t nodes = all_threads_stats.nodes - total_nodes_searched;

        nps = (nodes*1000) / (ms.count()+1);

        total_nodes_searched = all_threads_stats.nodes;


        if (!searching_flag || (limit_nodes != 0 && all_threads_stats.nodes > limit_nodes && depth >= min_depth)) {
            break;
        } else {
            if (time_man) {
                if (time_man->end_of_iteration(depth, get_move(), get_evaluation(), get_search_time_ms()) || root_moves.size() == 1) {
                    ready_flag = true;
                }
            }

            depth += 1;
        }
    }

    ready_flag = true;

    /*if (all_threads_stats.tempo_samples > 0) {
        std::cout << all_threads_stats.avg_tempo / all_threads_stats.tempo_samples << std::endl;
    }*/
}

void alphabeta_search::start_helper_threads(int32_t window_alpha, int32_t window_beta, int depth)
{
    alphabeta_abort_flag = false;
    for (int i = 0; i < number_of_helper_threads; i++) {
        helper_threads.push_back(std::thread(&alphabeta_search::search_root, this, window_alpha, window_beta, depth, i + 1));
    }
}

void alphabeta_search::stop_helper_threads()
{
    alphabeta_abort_flag = true;
    for (int i = 0; i < number_of_helper_threads; i++) {
        helper_threads[i].join();
    }
    helper_threads.clear();
}



void sort_root_moves(std::vector<std::pair<chess_move, int32_t>> &root_moves, chess_move prev_iteration_bm)
{
    for (auto it = root_moves.begin(); it < root_moves.end(); it++) {
        if (it->first == prev_iteration_bm) {
            int distance_from_end = root_moves.size() - std::distance(root_moves.begin(), it) - 1;
            std::rotate(root_moves.rbegin() + distance_from_end, root_moves.rbegin() + distance_from_end + 1, root_moves.rend());
            break;
        }
    }

    std::stable_sort(root_moves.begin() + 1, root_moves.end(), [] (auto a, auto b) {return (a.second > b.second); });
}



void alphabeta_search::aspirated_search(int depth)
{
    nominal_search_depth = depth;

    if (depth > 4) {
        int64_t window_beta_limit = MAX_EVAL-1;
        int64_t window_alpha_limit = MIN_EVAL+1;
        int64_t window_alpha = root_search_pv[num_of_pvs-1].score - 25;
        int64_t window_beta = root_search_pv[0].score + 25;
        int64_t window_expansion = 50;

        while (searching_flag) {
            sort_root_moves(root_moves, root_search_pv[0].moves[0]);

            start_helper_threads(window_alpha, window_beta, depth);
            search_root(window_alpha, window_beta, depth, 0);
            stop_helper_threads();

            if (!searching_flag) {
                return;
            }

            //Adjust window when search failed high or low
            if (root_search_pv[0].score >= window_beta) {
                window_beta = std::min((int64_t)root_search_pv[0].score + window_expansion, window_beta_limit);
                window_expansion *= 2.0f;
            } else if (root_search_pv[num_of_pvs-1].score <= window_alpha) {
                window_alpha = std::max((int64_t)root_search_pv[num_of_pvs-1].score - window_expansion, window_alpha_limit);
                window_expansion *= 2.0f;
            } else {
                break;
            }

            if (limit_nodes != 0 && all_threads_stats.nodes > limit_nodes && depth > min_depth) {
                return;
            }
        }
    } else {
        alphabeta_abort_flag = false;
        search_root(MIN_EVAL, MAX_EVAL, depth, 0);
        sort_root_moves(root_moves, root_search_pv[0].moves[0]);
    }

    info_lock.lock();

    search_info info;
    info.nodes = all_threads_stats.nodes;
    info.time_ms = get_search_time_ms();
    info.depth = depth;
    info.seldepth = all_threads_stats.max_distance_to_root;

    for (int i = 0; i < num_of_pvs; i++) {
        info.lines.push_back(std::make_shared<pv_table>(root_search_pv[i]));
    }
    search_infos.push_back(info);

    info_lock.unlock();
}



void alphabeta_search::search_root(int32_t window_alpha, int32_t window_beta, int depth, int thread_id)
{
    search_context &sc = *thread_datas[thread_id];

    int32_t alpha = window_alpha;
    int32_t beta = window_beta;

    sc.stats.reset();
    sc.history.test_flag = test_flag;
    sc.static_eval[0] = static_evaluation(sc.state, sc.state.get_turn(), sc.stats);
    sc.main_thread = (thread_id == 0);

    pv_table root_pv[MAX_MULTI_PV];
    pv_table line;

    for (auto it = root_moves.begin(); it != root_moves.end(); it++) {
        chess_move root_move = it->first;

        sc.moves[0] = root_move;
        sc.conthist[0] = sc.history.get_continuation_history_table(root_move);
        sc.stats.nodes += 1;

        line.first(root_move);

        unmove_data restore = sc.state.make_move(root_move);

        int32_t pv_score = MAX_EVAL;
        for (int i = 0; i < num_of_pvs; i++) {
            pv_score = std::min(pv_score, root_pv[i].score);
        }

        alpha = std::max(window_alpha, pv_score);

        int32_t score;
        if (pv_score == MIN_EVAL) {
            score = -alphabeta(sc.state, -beta, -alpha, depth - 1, 1, sc, line, nullptr, PV_NODE);
        } else {
            score = -alphabeta(sc.state, -alpha-1, -alpha, depth - 1, 1, sc, line, nullptr, CUT_NODE);

            if (score > alpha) {
                line.first(root_move);
                score = -alphabeta(sc.state, -beta, -alpha, depth - 1, 1, sc, line, nullptr, PV_NODE);
            }
        }

        sc.state.unmake_move(root_move, restore);

        if (alphabeta_abort_flag) {
            break;
        }

        if (is_score_valid(score)) {
            it->second = score;

            int worst_pv = 0;
            int worst_pv_score = root_pv[0].score;
            for (int i = 1; i < num_of_pvs; i++) {
                if (root_pv[i].score < worst_pv_score) {
                    worst_pv_score = root_pv[i].score;
                    worst_pv = i;
                }
            }

            if (score > worst_pv_score) {
                root_pv[worst_pv] = line;
                root_pv[worst_pv].score = score;

                if (score >= beta) {
                    break;
                }
            }
        }
    }

    root_search_lock.lock();
    if (!alphabeta_abort_flag) { //First to finish aborts others
        alphabeta_abort_flag = true;

        //Copy and order pvs
        for (int i = 0; i < num_of_pvs; i++) {
            int32_t pv_index = 0;
            int32_t pv_score = root_pv[0].score;
            for (int j = 1; j < num_of_pvs; j++) {
                if (root_pv[j].score > pv_score) {
                    pv_index = j;
                    pv_score = root_pv[j].score;
                }
            }

            root_search_pv[i] = root_pv[pv_index];
            root_pv[pv_index].score = MIN_EVAL;

        }
    }
    all_threads_stats.add(sc.stats);

    root_search_lock.unlock();
}


int32_t alphabeta_search::static_evaluation(const board_state &state, player_type_t player, search_statistics &stats)
{
    int32_t score;
    if (eval_cache[state.zhash].read(state.zhash, score)) {
        stats.eval_cache_hits++;
    } else {
        stats.eval_cache_misses++;

        score = state.nnue->evaluate(player, encode_output_bucket(state.get_non_pawn_pieces_bb()));

        eval_cache[state.zhash].write(state.zhash, score);
    }
    return score;
}



int32_t alphabeta_search::alphabeta(board_state &state, int32_t alpha, int32_t beta, int depth, int ply, search_context &sc, pv_table &pv, chess_move *skip_move, node_type_t expected_node_type)
{
    sc.stats.max_distance_to_root = std::max(sc.stats.max_distance_to_root, ply);

    //These are just flags for expected node type.
    //Node type is not known until all move have been searched or cutoff occurs
    bool is_cut = (expected_node_type == CUT_NODE);
    bool is_pv = (expected_node_type == PV_NODE);

    uint64_t zhash = (skip_move == nullptr ? state.zhash : hashgen.get_singular_search_hash(state.zhash, *skip_move));

    if (alphabeta_abort_flag) {
        return INVALID_EVAL;
    }

    if (sc.main_thread && time_man && (sc.stats.nodes % 2048) == 0) {
        if (time_man->should_stop_search(get_search_time_ms())) {
            ready_flag = true;
        }
    }

    if (state.check_repetition() >= 2 || state.half_move_clock >= 100 || state.is_insufficient_material()) {
        return 0;
    }

    //Mate distance pruning
    alpha = std::max(alpha, MIN_EVAL + ply);
    beta = std::min(beta, MAX_EVAL - ply);

    if (alpha >= beta) {
        return alpha;
    }

    if (depth <= 0 || ply >= MAX_DEPTH-2) {
        return quisearch(state, alpha, beta, ply, sc, is_pv);
    }

    chess_move tt_move = chess_move::null_move();
    int tt_depth = 0;
    int tt_node_type = ALL_NODE;
    int32_t tt_score = 0;
    int32_t tt_static_eval = 0;

    bool tt_hit = transposition_table[zhash].probe(state, zhash, tt_move, tt_depth, tt_node_type, tt_score, tt_static_eval, ply);

    sc.stats.cache_hits += tt_hit;
    sc.stats.cache_misses += (!tt_hit);

    //If we have tt hit with proper depth, we might be able go get cutoff
    //We dont use tt cutoffs in PV nodes, because 1) PV line must be collected 2) Search anomalies
    if (tt_hit &&
        !is_pv &&
        tt_depth >= depth &&
        state.half_move_clock < 90)
    {
        sc.static_eval[ply] = tt_static_eval;
        sc.moves[ply] = tt_move; //if previous move was null move, move which caused cut-off is threat move

        if (tt_node_type == CUT_NODE && tt_score >= beta) {
            return tt_score;
        } else if (tt_node_type == ALL_NODE && tt_score <= alpha) {
            return tt_score;
        }
    }

    int32_t raw_eval = (tt_hit ? tt_static_eval : static_evaluation(state, state.get_turn(), sc.stats));
    int32_t static_eval = raw_eval + sc.history.get_correction_history(state);
    int32_t eval = static_eval;

    //If we have tt hit, we might be able to use tt score as more accurate evaluation
    if (tt_hit && tt_score != 0 && !is_mate_score(tt_score)) {
        if ((tt_node_type == CUT_NODE && static_eval < tt_score) ||
            (tt_node_type == ALL_NODE && static_eval > tt_score)) {
            eval = tt_score;
        }
    }

    bool improving = false;

    if (ply >= 2) {
        if (sc.static_eval[ply-2] == MIN_EVAL) {
            if (ply >= 4 && sc.static_eval[ply-4] != MIN_EVAL) {
                improving = (static_eval > sc.static_eval[ply-4]);
            }
        } else {
            improving = (static_eval > sc.static_eval[ply-2]);
        }
    }

    bool in_check = state.in_check(state.get_turn());

    sc.static_eval[ply] = (in_check ? MIN_EVAL : static_eval); //NNUE is not trained check positions
    sc.history.reset_killers(ply+2);
    pv_table line;

    //Reverse futility pruning
    //If evaluation is good enough at shallow depth, we prune
    int32_t rfmargin = std::max(0, sp.rfmargin_base + sp.rfmargin_mult*depth - (improving ? sp.rfmargin_improving_modifier : 0));
    if (is_cut &&
        !in_check &&
        depth < 6 &&
        ply > 2 &&
        skip_move == nullptr &&
        eval - rfmargin > beta &&
        beta < 32000 && beta > -32000)
    {
        return (eval + beta) / 2;
    }

    //Null move pruning
    chess_move threat_move = chess_move::null_move();
    int non_pawn_pieces = state.count_non_pawn_pieces(state.get_turn());
    if (!is_pv &&
        !in_check &&
        (state.flags & NULL_MOVE) == 0 &&
        depth > 3 &&
        non_pawn_pieces > 2 &&
        ply >= sc.min_nmp_ply &&
        skip_move == nullptr &&
        eval >= beta)
    {
        uint64_t next_hash = hashgen.next_turn_hash(state.zhash, state);

        eval_cache.prefetch(next_hash);
        transposition_table.prefetch(next_hash);

        int nmp_reduction = 3 + std::clamp((eval - beta) / 100, 0, 4) + (depth / 3);

        int nmp_depth = std::max(depth - nmp_reduction, 1);

        sc.moves[ply+1] = chess_move::null_move();
        sc.moves[ply] = chess_move::null_move();
        sc.conthist[ply] = sc.history.get_continuation_history_table_null(state.get_turn());
        sc.stats.nodes += 1;

        unmove_data restore = state.make_null_move();

        int32_t null_score = -alphabeta(state, -beta, 1-beta, nmp_depth, ply+1, sc, line, nullptr, CUT_NODE);

        state.unmake_null_move(restore);

        if (null_score >= beta) {
            if (depth > 8) {
                //NMP vertification search
                int restore_min_nmp_ply = sc.min_nmp_ply;

                sc.min_nmp_ply = ply + ((nmp_depth * 3) / 4) + 1;

                int32_t score = alphabeta(state, beta-1, beta, nmp_depth, ply, sc, line, nullptr, ALL_NODE);

                sc.min_nmp_ply = restore_min_nmp_ply;

                if (score >= beta) {
                    return std::min(null_score, score);
                }
            } else {
                //If null move search returns mate score, return beta because line where mate score is found is not legal
                return (is_mate_score(null_score) ? beta : null_score);
            }
        } else {
            //Move which refutes null move is fetched for improving move ordering
            threat_move = sc.moves[ply+1];
        }
    }

    int32_t razoring_margin = sp.razoring_margin*depth;
    if (!is_pv && !in_check && depth < 5 && eval + razoring_margin < alpha) {
        int score = quisearch(state, alpha, beta, ply, sc, false);
        if (score < alpha) {
            return score;
        }
    }

    //IIR
    //If this unexplored part of tree proves to be important, there will be tt hits at next search
    if ((is_pv || is_cut) && depth >= 3 && !tt_hit) {
        depth -= 1;
    }

    int tt_move_extensions = 0;
    if (tt_hit &&
        (tt_node_type == CUT_NODE || tt_node_type == PV_NODE) &&
        tt_depth >= (depth-3) &&
        depth > 5 &&
        skip_move == nullptr &&
        !is_mate_score(tt_score) &&
        ply + depth < 2*nominal_search_depth)
    {
        int32_t singular_beta = tt_score - (2*depth);
        int32_t score = alphabeta(state, singular_beta-1, singular_beta, depth / 2, ply, sc, line, &tt_move, (is_cut ? CUT_NODE : ALL_NODE));

        if (is_pv) {
            if (score < singular_beta) {
                //Singular extension
                //TT move is better than rest of the moves. This node is singular and should be searcher with more carefully
                tt_move_extensions += 1;
            } else if (singular_beta > alpha && tt_score < beta) {
                //There is other moves that might beat alpha. Lets extend them (by reducing tt move)
                tt_move_extensions -= 1;
                if (ply+depth < nominal_search_depth-3) {
                    depth += 1; //Depth shouldn't get too much below nominal search depth
                }
            }
        } else {
            if (score < singular_beta) {
                tt_move_extensions += 1 + (score + 10*depth < tt_score);
            } else if (singular_beta >= beta && !is_mate_score(score)) {
                //Multicut
                //There is multiple moves that fails-high in this node
                //Changes that deeper search doesn't fail high is low
                return score;
            } else if (tt_score >= beta) {
                tt_move_extensions -= 2;
            } else if (is_cut) {
                tt_move_extensions -= 1;
            }
        }
    }

    move_picker &mpicker = sc.move_pickers[ply];
    mpicker.init(state, ply, tt_move, threat_move, sc.history, test_flag);

    chess_move best_move = chess_move::null_move();
    int32_t best_score = MIN_EVAL;
    int node_type = ALL_NODE;
    chess_move mov;
    int move_type = MOVES_END;

    while ((move_type = mpicker.pick(mov)) != MOVES_END) {
        if (skip_move && mov == *skip_move) {
            mpicker.previus_pick_was_skipped(mov);
            continue;
        }
        bool causes_check = state.causes_check(mov, next_turn(state.get_turn()));
        bool is_capture = mov.is_capture();
        bool is_promotion = (mov.promotion != 0);


        int extensions = (move_type == TT_MOVE ? tt_move_extensions : 0);
        int reductions = (int)(std::log2(depth) * std::log2(std::max(1, mpicker.legal_moves - is_pv)) * ((float)sp.lmr_modifier) / 100.0f);

        int32_t history_score = (is_capture ? sc.history.get_capture_history(mov) : sc.history.get_quiet_history(ply, mov));

        reductions -= history_score / sp.lmr_hist_adjust;

        //Forward pruning at low depth
        if (best_score != MIN_EVAL && !is_mate_score(best_score) && ply > 2) {
            int d = depth;
            int rd = std::clamp(depth - reductions, (d+1)/2, d);

            if ((mpicker.legal_moves >= (d*d+6)   && improving) ||
                (mpicker.legal_moves >= (d*d+6)/2 && !improving)) {
                mpicker.skip_quiets(); //Late move pruning. Currently does not prune killers
            }


            bool prune = false;
            if (!is_capture) {
                //If move is tactical, it should not be pruned
                //Or if we are in PV node or in check
                bool can_be_pruned = (!in_check && !is_promotion && !causes_check && !is_pv);

                //History pruning. If move have bad quiet history, skip it
                int32_t history_treshold = -d * sp.hmargin_mult;
                if (can_be_pruned &&
                    move_type != KILLER_MOVE &&
                    history_score < history_treshold &&
                    d < 6) {
                    prune = true;
                }

                //Futility pruning. If static evaluation is way belove alpha, skip move
                //Futility margin gets smaller for later moves
                int32_t fmargin = sp.fmargin_base + sp.fmargin_mult*rd;
                if (can_be_pruned &&
                    eval + fmargin < alpha &&
                    rd < 8) {
                    prune = true;
                }
            } else {
                //SEE pruning. If capture SEE is below treshold, skip capture
                int32_t see_treshold = -d * sp.see_margin_mult;
                if (!is_pv && d < 6 && static_exchange_evaluation(state, mov) < see_treshold) {
                    prune = true;
                }
            }
            if (prune) {
                //Notify move picker that move was skipped so that it dont get history penalty for not causing cut-off.
                mpicker.previus_pick_was_skipped(mov);
                continue;
            }
        }

        uint64_t next_hash = hashgen.update_hash(state.zhash, state, mov);

        transposition_table.prefetch(next_hash);
        eval_cache.prefetch(next_hash);

        int new_ply = ply + 1;
        int new_depth = depth + extensions - 1;
        int reduced_depth = new_depth;

        //Late move reduction
        //Non PV nodes starts reducing after first move, PV nodes start reducing after second move
        //Captures are not reduced at PV nodes
        if (new_depth >= 3 &&
            mpicker.legal_moves >= 2 &&
            !(is_pv && is_capture) &&
            !(is_pv && mpicker.legal_moves < 3))
        {
            if (!(is_pv || tt_node_type == PV_NODE)) {
                reductions += 2;
            }
            if (!improving) {
                reductions += 1;
            }

            if (causes_check || in_check) {
                reductions -= 1;
            }
            if (move_type == KILLER_MOVE) {
                reductions -= 1;
            }

            if (is_capture) {
                reductions /= 2;
            }

            reductions = std::clamp(reductions, 1, new_depth/2);

            reduced_depth = new_depth - reductions;
        }

        sc.moves[ply] = mov;
        sc.conthist[ply] = sc.history.get_continuation_history_table(mov);
        sc.stats.nodes += 1;

        unmove_data restore = state.make_move(mov, next_hash);

        int32_t score = MIN_EVAL;
        if (is_pv) {
            if (best_score != MIN_EVAL) {
                //If we have score for this node, we can search with zero window
                score = -alphabeta(state, -alpha-1, -alpha, reduced_depth, new_ply, sc, line, nullptr, CUT_NODE);

                //LMR research with full depth
                if (score > alpha && reduced_depth < new_depth) {
                    score = -alphabeta(state, -alpha-1, -alpha, new_depth, new_ply, sc, line, nullptr, CUT_NODE);
                }

                //If zero window search fails high, lets search with full window to get exact score
                if (score > alpha) {
                    line.first(mov);
                    score = -alphabeta(state, -beta, -alpha, new_depth, new_ply, sc, line, nullptr, PV_NODE);
                }
            } else {
                //If we dont have score for this node (first move), we search with full window
                line.first(mov);
                score = -alphabeta(state, -beta, -alpha, new_depth, new_ply, sc, line, nullptr, PV_NODE);
            }
        } else {
            //In all node, child nodes are expected to fail high (none of moves cause cutoff in this node).
            //In cut node only the first move is expected to cause cutoff.
            node_type_t child_type = ((mpicker.legal_moves > 1 || expected_node_type == ALL_NODE) ? CUT_NODE : ALL_NODE);

            score = -alphabeta(state, -alpha-1, -alpha, reduced_depth, new_ply, sc, line, nullptr, child_type);

            if (score > alpha && reduced_depth < new_depth) {
                score = -alphabeta(state, -alpha-1, -alpha, new_depth, new_ply, sc, line, nullptr, child_type);
            }
        }

        state.unmake_move(mov, restore);

        if (!is_score_valid(score)) {
            return INVALID_EVAL;
        }

        if (score > best_score) {
            best_score = score;
            best_move = mov;

            if (score >= beta) {
                node_type = CUT_NODE;

                if (depth > 5) {
                    sc.stats.fail_high_index += (mpicker.legal_moves == 1);
                    sc.stats.fail_highs += 1;
                }

                break;
            }

            if (score > alpha) {
                alpha = score;
                node_type = PV_NODE;

                pv.collect(line);
            }
        }
    }

    if (mpicker.legal_moves == 0) {
        if (!in_check) {
            best_score = 0;
        } else {
            best_score = (MIN_EVAL + ply);
        }
    }

    if (is_score_valid(best_score) && best_move.valid()) {
        //Update static evaluation correction history
        if (!is_mate_score(best_score) && !best_move.is_capture() && !in_check && skip_move == nullptr) {

            if ((node_type == CUT_NODE && raw_eval < best_score) ||
                (node_type == ALL_NODE && raw_eval > best_score) ||
                 node_type == PV_NODE) {
                sc.history.update_correction_history(state, best_score - raw_eval, depth);
            }
        }
        //Update move ordering heurestics. Move picker stores moves that were tried.
        if (node_type == CUT_NODE) {
            sc.history.update(best_move, depth, ply, mpicker.picked_quiet_moves, mpicker.picked_quiet_count, mpicker.picked_captures, mpicker.picked_capture_count);
        }

        //If we have tt move and all moves failed low,
        //we don't want overwrite move found from tt because nothing proves that our "best move" is actually better than alternatives
        //If node has previously be pv node or cut node, move in tt is probably better
        if (node_type == ALL_NODE && tt_hit) {
            best_move = tt_move;
        }

        transposition_table[zhash].store(zhash, best_move, depth, node_type, best_score, raw_eval, ply, current_cache_age);
    }

    return best_score;
}



int32_t alphabeta_search::quisearch(board_state &state, int32_t alpha, int32_t beta, int ply, search_context &sc, bool is_pv)
{
    if (ply >= MAX_DEPTH) {
        return static_evaluation(state, state.get_turn(), sc.stats);
    }

    sc.stats.max_distance_to_root = std::max(sc.stats.max_distance_to_root, ply);

    bool in_check = state.in_check(state.get_turn());

    chess_move tt_move = chess_move::null_move();
    int tt_depth = 0;
    int tt_node_type = ALL_NODE;
    int32_t tt_score = 0;
    int32_t tt_static_eval = 0;
    bool tt_hit = transposition_table[state.zhash].probe(state, state.zhash, tt_move, tt_depth, tt_node_type, tt_score, tt_static_eval, ply);

    if (!is_pv &&
        tt_hit &&
        tt_score != 0 &&
        !is_mate_score(tt_score))
    {
        if (tt_node_type == CUT_NODE && tt_score >= beta) {
            return tt_score;
        } else if (tt_node_type == ALL_NODE && tt_score <= alpha) {
            return tt_score;
        }
    }

    int32_t raw_eval = (tt_hit ? tt_static_eval : static_evaluation(state, state.get_turn(), sc.stats));
    int32_t static_eval = raw_eval + sc.history.get_correction_history(state);
    int32_t eval = static_eval;

    if (tt_hit && tt_score != 0 && !is_mate_score(tt_score)) {
        if ((tt_node_type == CUT_NODE && static_eval < tt_score) ||
            (tt_node_type == ALL_NODE && static_eval > tt_score)) {
            eval = tt_score;
        }
    }

    chess_move best_move = chess_move::null_move();
    int32_t best_score = MIN_EVAL;
    node_type_t node_type = ALL_NODE;
    int legal_moves = 0;

    scored_move_array<240> moves;

    if (!in_check) {
        //Static evaluation works as lower bound of the score quiscence search can return when we are not in check
        //Based on assumption that there is quiet move, which doesn't make position worse
        //Assumption doesn't hold in many cases, but purpose of quiscence search is to make sure that position where evaluation happens is quiet (opponent have change to recapture)
        if (eval >= beta) {
            return eval;
        }
        if (alpha < eval) {
            alpha = eval;
        }

        chess_move captures[80];
        int num_of_captures = move_generator::generate_capture_moves(state, state.get_turn(), captures);
        num_of_captures += move_generator::generate_promotion_moves(state, state.get_turn(), &captures[num_of_captures]);
        if (num_of_captures == 0) {
            //No captures of promotions, position is quiet
            return eval;
        }
        //Captures are ordered with SEE
        for (int i = 0; i < num_of_captures; i++) {
            moves.add(captures[i], (captures[i] == tt_move ? 32000 : static_exchange_evaluation(state, captures[i])));
        }

        best_score = eval;
    } else {
        //When in check, we search all move
        chess_move all_moves[240];
        int num_of_moves = move_generator::generate_all_pseudo_legal_moves(state, state.get_turn(), all_moves);

        for (int i = 0; i < num_of_moves; i++) {
            moves.add(all_moves[i], (all_moves[i] == tt_move ? 32000 : static_exchange_evaluation(state, all_moves[i])));
        }
    }

    chess_move mov;
    int32_t see;

    while (moves.pick(mov, see)) {
        if (state.causes_check(mov, state.get_turn())) {
            continue;
        }
        legal_moves++;

        //SEE pruning
        //Captures with negative SEE have very little changes to increase evaluation
        if (see < 0 && !in_check) {
            break;
        }

        uint64_t next_hash = hashgen.update_hash(state.zhash, state, mov);

        eval_cache.prefetch(next_hash);
        transposition_table.prefetch(next_hash);

        sc.stats.nodes += 1;

        unmove_data restore = state.make_move(mov, next_hash);

        int32_t score = -quisearch(state, -beta, -alpha, ply+1, sc, is_pv);

        state.unmake_move(mov, restore);

        if (score > best_score) {
            best_score = score;
            best_move = mov;
            if (score >= beta) {
                node_type = CUT_NODE;
                break;
            }
            if (score > alpha) {
                alpha = score;
                node_type = PV_NODE;
            }
        }
    }

    if (in_check && legal_moves == 0) {
        return (MIN_EVAL + ply);
    }

    //Only store node to TT if we found move which increased evaluation
    if (is_score_valid(best_score) && best_move.valid()) {
        if (node_type == ALL_NODE && tt_hit) { //Only possible in check
            best_move = tt_move;
        }
        transposition_table[state.zhash].store(state.zhash, best_move, 0, node_type, best_score, raw_eval, ply, current_cache_age);
    }
    return best_score;
}





