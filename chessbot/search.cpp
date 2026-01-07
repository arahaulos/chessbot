#include "search.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include "movepicker.hpp"
#include "movegen.hpp"
#include "zobrist.hpp"
#include "search_manager.hpp"

searcher::searcher()
{
    clear_transposition_table();
    clear_evaluation_cache();

    searching_flag = false;
    alphabeta_abort_flag = false;

    test_flag = true;

    current_cache_age = 0;
    num_of_pvs = 1;

    shared_nnue_weights = nnue_weights::get_shared_weights();

    set_threads(DEFAULT_THREADS);
}

searcher::searcher(const searcher &other): transposition_table(other.transposition_table), eval_cache(other.eval_cache)
{
    shared_nnue_weights = other.shared_nnue_weights;

    searching_flag = false;
    alphabeta_abort_flag = false;
    test_flag = other.test_flag;
    current_cache_age = other.current_cache_age;
    num_of_pvs = other.num_of_pvs;

    sp = other.sp;

    set_threads(other.number_of_helper_threads + 1);

    for (int i = 0; i < number_of_helper_threads+1; i++) {
        thread_datas[i]->history = other.thread_datas[i]->history;
    }
}


int32_t searcher::get_transpostion_table_usage_permill()
{
    uint64_t slots_used = 0;
    for (uint64_t i = 0; i < transposition_table.get_size(); i += 100) {
        slots_used += transposition_table[i].used(current_cache_age);
    }
    return (slots_used*100000) / (transposition_table.get_size()*TT_ENTRIES_IN_BUCKET);
}


void searcher::clear_transposition_table()
{
    for (uint64_t i = 0; i < transposition_table.get_size(); i++) {
        transposition_table[i].clear();
    }
}

void searcher::clear_evaluation_cache()
{
    for (uint64_t i = 0; i < eval_cache.get_size(); i++) {
        eval_cache[i].clear();
    }
}

void searcher::clear_history()
{
    for (size_t i = 0; i < thread_datas.size(); i++) {
        thread_datas[i]->history.reset();
    }
}

void searcher::set_threads(int num_of_threads)
{
    number_of_helper_threads = std::clamp(num_of_threads-1, 0, MAX_THREADS);

    thread_datas.clear();
    for (int i = 0; i < number_of_helper_threads+1; i++) {
        thread_datas.push_back(std::make_unique<search_context>(shared_nnue_weights));
    }
}

void searcher::set_transposition_table_size_MB(int size_MB)
{
    transposition_table.resize(size_MB);
    clear_transposition_table();
}


void searcher::new_game()
{
    current_cache_age += 4;

    clear_history();
}



void searcher::search(const board_state &state, std::shared_ptr<search_manager> m)
{
    if (searching_flag) {
        return;
    }
    searching_flag = true;
    manager = m;
    if (manager->tm) {
        manager->tm->test_flag = test_flag;
    }
    for (int i = 0; i < number_of_helper_threads+1; i++) {
        thread_datas[i]->presearch(state);
    }

    current_cache_age += 1;

    std::vector<chess_move> legal_moves = state.get_all_legal_moves(state.get_turn());

    if (legal_moves.size() == 0) {
        std::cout << "No legal moves!!" << std::endl;
        return;
    }

    root_moves.clear();
    for (auto it = legal_moves.begin(); it != legal_moves.end(); it++) {
        root_moves.push_back(std::pair(*it, 0));
    }

    uint64_t total_nodes_searched = 0;
    int depth = 1;
    while (depth <= MAX_DEPTH && searching_flag) {
        all_threads_stats.reset();
        all_threads_stats.nodes = total_nodes_searched;

        aspirated_search(depth);

        total_nodes_searched = all_threads_stats.nodes;

        depth += 1;
    }

    searching_flag = false;
}

void searcher::start_helper_threads(int32_t window_alpha, int32_t window_beta, int depth)
{
    alphabeta_abort_flag = false;
    for (int i = 0; i < number_of_helper_threads; i++) {
        helper_threads.push_back(std::thread(&searcher::search_root, this, window_alpha, window_beta, depth, i + 1));
    }
}

void searcher::stop_helper_threads()
{
    alphabeta_abort_flag = true;
    for (int i = 0; i < number_of_helper_threads; i++) {
        helper_threads[i].join();
    }
    helper_threads.clear();
}

uint64_t searcher::get_total_node_count()
{
    uint64_t nodes = all_threads_stats.nodes;
    for (int i = 0; i < number_of_helper_threads+1; i++) {
        nodes += thread_datas[i]->stats.nodes;
    }
    return nodes;
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



void searcher::aspirated_search(int depth)
{
    nominal_search_depth = depth;

    if (depth > 4) {
        int64_t window_expansion = 25;

        int64_t window_beta_limit = MAX_EVAL-1;
        int64_t window_alpha_limit = MIN_EVAL+1;
        int64_t window_alpha = root_search_pv[num_of_pvs-1].score - window_expansion;
        int64_t window_beta = root_search_pv[0].score + window_expansion;

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
            } else if (root_search_pv[num_of_pvs-1].score <= window_alpha) {
                window_alpha = std::max((int64_t)root_search_pv[num_of_pvs-1].score - window_expansion, window_alpha_limit);
            } else {
                break;
            }

            window_expansion *= 2;
        }
    } else {
        alphabeta_abort_flag = false;
        search_root(MIN_EVAL, MAX_EVAL, depth, 0);
        sort_root_moves(root_moves, root_search_pv[0].moves[0]);
    }
    searching_flag = !manager->on_end_of_iteration(depth, all_threads_stats.max_distance_to_root, all_threads_stats.nodes, root_search_pv, num_of_pvs);
}



void searcher::search_root(int32_t window_alpha, int32_t window_beta, int depth, int thread_id)
{
    search_context &sc = *thread_datas[thread_id];

    int32_t alpha = window_alpha;
    int32_t beta = window_beta;

    sc.history.test_flag = test_flag;
    sc.static_eval[0] = static_evaluation(sc.state, sc.state.get_turn(), sc.stats);
    sc.main_thread = (thread_id == 0);

    pv_table root_pv[MAX_MULTI_PV];
    pv_table line;

    for (auto it = root_moves.begin(); it != root_moves.end(); it++) {
        chess_move root_move = it->first;

        sc.reduction[0] = 0;
        sc.moves[0] = root_move;
        sc.conthist[0] = sc.history.get_continuation_history_table(root_move);
        sc.stats.nodes += 1;

        line.first(root_move);

        unmake_restore restore = sc.state.make_move(root_move);

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
    sc.stats.reset();

    root_search_lock.unlock();
}

int32_t searcher::static_evaluation(const board_state &state, player_type_t player, search_statistics &stats)
{
    int32_t score;
    if (eval_cache[state.zhash].probe(state.zhash, score)) {
        stats.eval_cache_hits++;
    } else {
        stats.eval_cache_misses++;

        score = state.nnue->evaluate(player);

        eval_cache[state.zhash].store(state.zhash, score);
    }
    return score;

}

int32_t searcher::alphabeta(board_state &state, int32_t alpha, int32_t beta, int depth, int ply, search_context &sc, pv_table &pv, chess_move *skip_move, node_type_t expected_node_type)
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

    if (sc.main_thread && (sc.stats.nodes % 2048) == 0) {
        if (manager->on_search_stop_control(get_total_node_count())) {
            alphabeta_abort_flag = true;
            searching_flag = false;
        }
    }


    if (state.check_repetition() >= 2 || state.half_move_clock >= 100 || state.is_insufficient_material()) {
        return 0;
    }

    //Mate distance pruning
    //MIN_EVAL + ply is absolutely worst score we can get with this ply
    //MAX_EVAL - ply is absolutely best score we can get with this ply
    //After alpha and beta gets adjusted with those limits, we can check if it is possible to find better mate at this ply
    alpha = std::max(alpha, MIN_EVAL + ply);
    beta = std::min(beta, MAX_EVAL - ply);

    if (alpha >= beta) {
        return alpha;
    }

    if (depth <= 0 || ply >= MAX_DEPTH-2) {
        return quisearch(state, alpha, beta, 0, ply, sc, is_pv);
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
        if (tt_node_type == CUT_NODE && tt_score >= beta) {
            return tt_score;
        } else if (tt_node_type == ALL_NODE && tt_score <= alpha) {
            return tt_score;
        }
    }

    if (tt_hit) {
        eval_cache[state.zhash].store(state.zhash, tt_static_eval);
    }

    bool in_check = state.in_check(state.get_turn());
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

    //Improving heurestic
    //If static evaluation has been increased since our previous turn we are improving
    bool improving = false;
    if (ply >= 2 && sc.static_eval[ply-2] != MIN_EVAL) {
        improving = (static_eval >= sc.static_eval[ply-2]);
    } else if (ply >= 4 && sc.static_eval[ply-4] != MIN_EVAL) {
        improving = (static_eval >= sc.static_eval[ply-4]);
    }

    sc.static_eval[ply] = (in_check ? MIN_EVAL : static_eval); //NNUE is not trained check positions
    sc.reduction[ply] = 0;
    sc.history.reset_killers(ply+2);

    pv_table line;

    if (sc.reduction[ply-1] > 0) {
        //Adjust LMR based on static evaluation difference
        //Idea of LMR is reduce more bad moves and less good moves
        //Static evaluation difference gives some hint about move quality
        if (static_eval < -sc.static_eval[ply-1]) {
            depth++; //Previous move was good for opponent (bad for current STM), lets decrease reduction
            sc.reduction[ply-1]--;
        } else if (depth > 2 && static_eval - (-sc.static_eval[ply-1]) > 120 && skip_move == nullptr) {
            depth--; //Previous move was bad for opponent, lets increase reduction
            sc.reduction[ply-1]++;
        }
    }

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
    //If position is so good that giving opponent extra move doesn't bring us below beta at reduced search, we can relatively safely prune this subtree
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

        int nmp_reduction = 3 + std::min((eval - beta) / 100, 4) + (depth / 3);
        int nmp_depth = std::max(depth - nmp_reduction, 0);

        sc.moves[ply] = chess_move::null_move();
        sc.conthist[ply] = sc.history.get_continuation_history_table_null(state.get_turn());
        sc.stats.nodes += 1;

        unmake_restore restore = state.make_null_move();

        int32_t null_score = -alphabeta(state, -beta, 1-beta, nmp_depth, ply+1, sc, line, nullptr, CUT_NODE);

        state.unmake_null_move(restore);

        if (null_score >= beta) {
            if (depth < 9) {
                return (is_mate_score(null_score) ? beta : null_score);
            }
            //NMP vertification search
            int restore_min_nmp_ply = sc.min_nmp_ply;
            int restore_prior_reduction = sc.reduction[ply-1];

            sc.min_nmp_ply = ply + ((nmp_depth * 3) / 4) + 1;

            int32_t score = alphabeta(state, beta-1, beta, nmp_depth, ply, sc, line, nullptr, ALL_NODE);

            sc.min_nmp_ply = restore_min_nmp_ply;
            sc.reduction[ply-1] = restore_prior_reduction;

            if (score >= beta) {
                return std::min(null_score, score);
            }
        }
    }

    //Razoring
    //If eval is below alpha with big margin and qsearch doesnt reveal easy capture which saves evaluation, node gets pruned
    int32_t razoring_margin = sp.razoring_margin*depth;
    if (!is_pv && !in_check && depth < 5 && eval + razoring_margin < alpha) {
        int score = quisearch(state, alpha, beta, 0, ply, sc, false);
        if (score < alpha) {
            return score;
        }
    }

    //Internal iterative reduction
    //If this unexplored part of tree proves to be important, there will be tt hits at next iteration
    if ((is_pv || is_cut) && depth >= 3 && !tt_hit) {
        depth -= 1;
    }

    //Probcut
    //If there is good capture, capture is searched at reduced depth with higher beta.
    //If reduced search returns score above raised beta, we can safely prune this subtree
    int probcut_beta = beta + sp.probcut_margin;
    int probcut_depth = depth - 5;
    if (!is_pv &&
        !in_check &&
        skip_move == nullptr &&
        probcut_depth > 0 &&
        (!tt_hit || tt_score >= probcut_beta)) {

        scored_move_array<80> scored_captures;
        chess_move captures[80];
        int num_of_captures = move_generator::generate_capture_moves(state, state.get_turn(), captures);

        for (int i = 0; i < num_of_captures; i++) {
            int see = static_exchange_evaluation(state, captures[i]);
            if (static_eval + see >= probcut_beta && see > 0) {
                scored_captures.add(captures[i], (captures[i] == tt_move ? 32000 : see));
            }
        }

        chess_move cap;
        while (scored_captures.pick(cap)) {
            if (state.causes_check(cap, state.get_turn())) {
                continue;
            }
            uint64_t next_hash = hashgen.update_hash(state.zhash, state, cap);

            transposition_table.prefetch(next_hash);
            eval_cache.prefetch(next_hash);

            sc.moves[ply] = cap;
            sc.conthist[ply] = sc.history.get_continuation_history_table(cap);
            sc.reduction[ply] = 0;
            sc.stats.nodes += 1;

            unmake_restore restore = state.make_move(cap, next_hash);

            int32_t score = -quisearch(state, -probcut_beta, 1-probcut_beta, 0, ply+1, sc, false);
            if (score >= probcut_beta) {
                score = -alphabeta(state, -probcut_beta, 1-probcut_beta, probcut_depth, ply+1, sc, line, nullptr, ALL_NODE);
            }

            state.unmake_move(cap, restore);

            if (score >= probcut_beta) {
                if (!tt_hit) {
                    transposition_table[zhash].store(zhash, cap, probcut_depth+1, CUT_NODE, score, raw_eval, ply, current_cache_age);
                }
                return score;
            }
        }
    }

    //Singular search
    //When tt entry is found which is marked as cut or pv node, reduced search is performed to check if there is alternative good moves
    //When singular search fails high, there might be other moves which can be also good
    //When singular search fails low, there is only one good move and move is so called singular move.
    int tt_move_extensions = 0;
    if (tt_hit &&
        (tt_node_type == CUT_NODE || tt_node_type == PV_NODE) &&
        tt_depth >= (depth-3) &&
        depth > 5 &&
        skip_move == nullptr &&
        !is_mate_score(tt_score) &&
        ply + depth < 2*nominal_search_depth)
    {
        int restore_prior_reduction = sc.reduction[ply-1];

        int32_t singular_beta = tt_score - (2*depth);
        int32_t score = alphabeta(state, singular_beta-1, singular_beta, depth / 2, ply, sc, line, &tt_move, (is_cut ? CUT_NODE : ALL_NODE));

        sc.reduction[ply-1] = restore_prior_reduction;

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
                //Our singular beta wasn't high enough to get cutoff
                tt_move_extensions -= 2;
            } else if (is_cut) {
                tt_move_extensions -= 1;
            }
        }
    }

    move_picker &mpicker = sc.move_pickers[ply];

    mpicker.good_quiet_treshold = sp.good_quiet_treshold;
    mpicker.init(state, ply, tt_move, sc.history, test_flag);

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
        bool is_promotion = mov.is_promotion();


        int extensions = (move_type == TT_MOVE ? tt_move_extensions : 0);
        int reductions = (int)(std::log2(depth) * std::log2(std::max(1, mpicker.legal_moves - is_pv)) * ((float)sp.lmr_modifier) / 100.0f);

        int32_t history_score = (is_capture ? sc.history.get_capture_history(mov) : sc.history.get_quiet_history(ply, mov));

        if (!is_capture) {
            reductions -= std::clamp(history_score / sp.lmr_hist_adjust, -3, 3);
        }

        //Forward pruning at low depth
        if (best_score != MIN_EVAL && !is_mate_score(best_score) && ply > 2) {
            int lmr_depth = std::clamp(depth - reductions, (depth+1)/2, depth);
            int lmp_count = depth*depth + 6 + (is_pv ? 4 : 0);

            if ((mpicker.legal_moves >= lmp_count   && improving) ||
                (mpicker.legal_moves >= lmp_count/2 && !improving)) {
                mpicker.skip_quiets(); //Late move pruning. Currently does not prune killers
            }


            bool prune = false;
            if (!is_capture) {
                //If move is tactical, it should not be pruned
                //Or if we are in PV node or in check
                bool can_be_pruned = (!in_check && !is_promotion && !causes_check && !is_pv);

                //History pruning. If move have bad quiet history, skip it
                int32_t history_treshold = -depth * sp.hmargin_mult;
                if (can_be_pruned &&
                    move_type != KILLER_MOVE &&
                    history_score < history_treshold &&
                    depth < 6) {
                    prune = true;
                }

                //Futility pruning. If static evaluation is way belove alpha, skip move
                //Futility margin gets smaller for later moves
                int32_t fmargin = sp.fmargin_base + sp.fmargin_mult*lmr_depth;
                if (can_be_pruned &&
                    eval + fmargin < alpha &&
                    lmr_depth < 8) {
                    prune = true;
                }

                //SEE pruning for quiet moves.
                //If SEE indicates quiet move loses material, its gets pruned
                if (!prune && !is_pv && !in_check && depth < 6) {
                    int32_t see_treshold = -depth*depth * sp.quiet_see_margin_mult;
                    if (static_exchange_evaluation(state, mov) < see_treshold) {
                        prune = true;
                    }
                }
            } else {
                //SEE pruning for captures.
                //If SEE indicates capture is really bad, its gets pruned
                int32_t see_treshold = -depth * sp.cap_see_margin_mult;
                if (!is_pv && depth < 6 && static_exchange_evaluation(state, mov) < see_treshold) {
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
                reductions += 1 + !improving;
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

            reductions = std::clamp(reductions, 0, new_depth/2);

            reduced_depth = new_depth - reductions;
        }

        sc.moves[ply] = mov;
        sc.conthist[ply] = sc.history.get_continuation_history_table(mov);
        sc.reduction[ply] = (reduced_depth < new_depth ? reductions : 0);
        sc.stats.nodes += 1;

        unmake_restore restore = state.make_move(mov, next_hash);

        int32_t score = MIN_EVAL;
        if (is_pv) {
            if (best_score != MIN_EVAL) {
                //If we have score for this node, we can search with zero window
                score = -alphabeta(state, -alpha-1, -alpha, reduced_depth, new_ply, sc, line, nullptr, CUT_NODE);

                //LMR research with full depth
                if (score > alpha && sc.reduction[ply] > 0) {
                    sc.reduction[ply] = 0;
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

            if (score > alpha && sc.reduction[ply] > 0) {
                sc.reduction[ply] = 0;
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



int32_t searcher::quisearch(board_state &state, int32_t alpha, int32_t beta, int depth, int ply, search_context &sc, bool is_pv)
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

    if (tt_hit) {
        eval_cache[state.zhash].store(state.zhash, tt_static_eval);
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
        //Assumption doesn't hold in many cases, but purpose of quiscence search is to make sure that position where evaluation happens is quiet (opponent can recapture)
        if (eval >= beta) {
            return eval;
        }

        if (alpha < eval) {
            alpha = eval;
        }

        chess_move movelist[80];
        int num_of_moves = move_generator::generate_capture_moves(state, state.get_turn(), movelist);
        num_of_moves += move_generator::generate_promotion_moves(state, state.get_turn(), &movelist[num_of_moves]);
        if (depth == 0) {
            num_of_moves += move_generator::generate_quiet_checks(state, state.get_turn(), &movelist[num_of_moves]);
        }

        if (num_of_moves == 0) {
            //No captures or promotions, position is quiet
            return eval;
        }

        //Non TT moves which have negative SEE are pruned.
        bool tt_move_found = false;
        for (int i = 0; i < num_of_moves; i++) {
            chess_move mov = movelist[i];
            if (mov == tt_move) {
                tt_move_found = true;
                moves.add(mov, 32000);
            } else {
                int32_t see = static_exchange_evaluation(state, mov);
                if (mov.is_capture()) {
                    see += sc.history.get_capture_history(mov)/32;
                }
                if (see >= 0) {
                    moves.add(mov, see);
                }
            }
        }

        //Move from tt will be played even if it is quiet at first ply in qsearch
        if (tt_hit && !tt_move_found && depth == 0) {
            moves.add(tt_move, 32000);
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
    int32_t see;
    chess_move mov;
    while (moves.pick(mov, see)) {
        if (state.causes_check(mov, state.get_turn())) {
            continue;
        }
        legal_moves++;

        uint64_t next_hash = hashgen.update_hash(state.zhash, state, mov);

        eval_cache.prefetch(next_hash);
        transposition_table.prefetch(next_hash);

        sc.stats.nodes += 1;

        unmake_restore restore = state.make_move(mov, next_hash);

        int32_t score = -quisearch(state, -beta, -alpha, depth-1, ply+1, sc, is_pv);

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
        if (node_type == ALL_NODE && tt_hit) {
            best_move = tt_move;
        }
        transposition_table[state.zhash].store(state.zhash, best_move, 0, node_type, best_score, raw_eval, ply, current_cache_age);
    }
    return best_score;
}
