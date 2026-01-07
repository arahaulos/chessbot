#include "uci.hpp"
#include "search_manager.hpp"
#include <sstream>
#include "util/wdl_model.hpp"

int32_t scale_eval(int32_t score)
{
    constexpr int scaling_factor = 719;

    if (!is_mate_score(score)) {
        return (score * 400) / scaling_factor;
    } else {
        return score;
    }
}


std::vector<std::string> split_string(std::string str, char d)
{
    std::vector<std::string> splitted_strings;

    size_t p = str.find(d);
    while (p != std::string::npos) {
        splitted_strings.push_back(str.substr(0, p));
        str.erase(0, p + 1);
        p = str.find(d);
    }
    splitted_strings.push_back(str);

    return splitted_strings;
}


uci_interface::uci_interface(std::shared_ptr<searcher> search_inst, std::shared_ptr<game_state> game_inst)
{
    running = false;
    search_instance = search_inst;
    game_instance = game_inst;
    exit_flag = false;

    show_wdl = false;
    ponder = false;

    time_man = std::make_shared<time_manager>(0,0);
    search_man = std::make_shared<search_manager>();

    search_man->set_uci(this);
    //uci_log.open("uci_log.txt");
}

uci_interface::~uci_interface()
{
    if (running) {
        stop();
    }

    if (uci_log.is_open()) {
        uci_log.close();
    }
}

bool uci_interface::exit()
{
    return exit_flag;
}

void uci_interface::start()
{
    std::cout << "Starting UCI" << std::endl;

    running = true;
    uci_thread = std::thread(&uci_interface::input_loop, this);
}

void uci_interface::stop()
{
    if (!running) {
        return;
    }
    running = false;
    uci_thread.join();

    if (search_thread.joinable()) {
        search_thread.join();
    }
}

std::string uci_interface::parse_command(std::string line, int word)
{
    std::vector<std::string> words = split_string(line, ' ');
    if (word < (int)words.size() && word >= 0) {
        return words[word];
    }
    return std::string("");
}

void uci_interface::search_thread_entry()
{
    search_instance->search(game_instance->get_state(), search_man);
}


void uci_interface::input_loop()
{
    std::string input_str;
    while (running) {
        while (std::getline(std::cin, input_str)) {
            if (uci_log.is_open()) {
                uci_log << "GUI: " << input_str << "\n";
            }

            if (input_str == "quit") {
                exit_flag = true;
                return;
            } else {
                execute_command(input_str);
            }
        }
    }
}


void uci_interface::send_command(std::string cmd)
{
    if (uci_log.is_open()) {
        uci_log << "ENG: " << cmd << "\n";
    }

    std::cout << cmd << std::endl;
}

void uci_interface::execute_command(std::string cmd_line)
{
    std::string cmd = parse_command(cmd_line, 0);

    if (cmd == "uci") {
        send_command("id name chessbot");
        send_command("id author asdf");


        std::stringstream ss;
        ss << "option name Threads type spin default " << DEFAULT_THREADS << " min 1 max " << MAX_THREADS;
        send_command(ss.str());

        ss.str(std::string());
        ss << "option name Hash type spin default " << TT_SIZE << " min 1 max 16384";
        send_command(ss.str());

        ss.str(std::string());
        send_command("option name Clear Hash type button");

        ss.str(std::string());
        ss << "option name MultiPV type spin default 1 min 1 max " << MAX_MULTI_PV;
        send_command(ss.str());

        ss.str(std::string());
        ss << "option name Move Overhead type spin default " << time_manager::get_default_overhead() << " min 0 max 10000";
        send_command(ss.str());

        ss.str(std::string());
        ss << "option name UCI_ShowWDL type check default false";
        send_command(ss.str());

        ss.str(std::string());
        ss << "option name Ponder type check default false";
        send_command(ss.str());

        send_command("uciok");
    } else if (cmd == "isready") {
        send_command("readyok");
    } else if (cmd == "position") {
        std::vector<std::string> splitted_cmd = split_string(cmd_line, ' ');

        std::vector<chess_move> moves;

        std::string fen_str;

        bool parsing_fen = false;
        bool parsing_moves = false;

        int fen_parts = 0;

        for (auto it = splitted_cmd.begin(); it != splitted_cmd.end(); it++) {
            std::string str = *it;

            if (parsing_moves) {
                chess_move mov;
                mov.from_uci(str);
                moves.push_back(mov);
            }
            if (parsing_fen) {
                if (fen_parts > 0) {
                    fen_str += " ";
                }
                fen_str += str;

                fen_parts += 1;
                if (fen_parts >= 6) {
                    parsing_fen = false;
                }
            }
            if (str == "startpos") {
                fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
            }
            if (str == "fen") {
                parsing_fen = true;
            }
            if  (str == "moves") {
                parsing_moves = true;
            }
        }

        game_instance->get_state().load_fen(fen_str);
        for (auto it = moves.begin(); it != moves.end(); it++) {
            game_instance->get_state().make_move(*it);
        }
    } else if (cmd == "go") {
        if (search_thread.joinable()) {
            search_thread.join();
        }

        std::vector<std::string> splitted_cmd = split_string(cmd_line, ' ');

        go_info info;
        info.type = GO_CLOCK;
        info.active = true;
        for (auto it = splitted_cmd.begin(); it != splitted_cmd.end(); it++) {
            std::string str = *it;
            auto next_it = it+1;
            if (next_it != splitted_cmd.end()) {
                if (str == "movetime") {
                    info.type = GO_MOVETIME;
                    info.movetime = std::atoi(next_it->c_str());
                }
                if  (str == "wtime") {
                    info.wtime = std::atoi(next_it->c_str());
                }
                if (str == "btime") {
                    info.btime = std::atoi(next_it->c_str());
                }
                if (str == "winc") {
                    info.winc = std::atoi(next_it->c_str());
                }
                if (str == "binc") {
                    info.binc = std::atoi(next_it->c_str());
                }
                if (str == "depth") {
                    info.type = GO_DEPTH;
                    info.depth = std::atoi(next_it->c_str());

                }
               if (str == "nodes") {
                    info.type = GO_NODES;
                    info.nodes = std::atoi(next_it->c_str());
                }
            }
            if (str == "infinite") {
                info.type = GO_INFINITE;
            }
            if (str == "ponder") {
                info.type = GO_PONDER;
            }
        }
        ginfo = info;

        if (info.type == GO_CLOCK) {
            if (game_instance->get_state().get_turn() == WHITE) {
                time_man->init(info.wtime, info.winc);
            } else {
                time_man->init(info.btime, info.binc);
            }
            search_man->prepare_clock_search(time_man);
        } else if (info.type == GO_NODES) {
            search_man->prepare_fixed_nodes_search(info.nodes);
        } else if (info.type == GO_DEPTH) {
            search_man->prepare_fixed_depth_search(info.depth);
        } else if (info.type == GO_MOVETIME) {
            search_man->prepare_fixed_time_search(info.movetime);
        } else if (info.type == GO_INFINITE || info.type == GO_PONDER) {
            search_man->prepare_infinite_search();
        }

        search_thread = std::thread(&uci_interface::search_thread_entry, this);

        last_info_depth = 0;
    } else if (cmd == "stop") {
        search_man->stop_search();
        if (search_thread.joinable()) {
            search_thread.join();
        }
    } else if (cmd == "ponderhit") {
        //When ponderhit happens, stop search and begin new search with clock
        ginfo.type = GO_CLOCK;
        if (game_instance->get_state().get_turn() == WHITE) {
            time_man->init(ginfo.wtime, ginfo.winc);
        } else {
            time_man->init(ginfo.btime, ginfo.binc);
        }
        search_man->start_clock(time_man);
    } else if (cmd == "ucinewgame") {
        search_instance->new_game();
    } else if (cmd == "quit") {
        exit_flag = true;
    } else if (cmd == "setoption") {
        std::string option_name;
        std::string option_value;
        std::vector<std::string> splitted_cmd = split_string(cmd_line, ' ');
        for (auto it = splitted_cmd.begin(); it != splitted_cmd.end(); it++) {
            std::string str = *it;
            auto next_it = it+1;
            if (next_it != splitted_cmd.end()) {
                if (str == "name") {
                    option_name = *next_it;
                } else if (str == "value") {
                    option_value = *next_it;
                }
            }
        }
        if (option_name == "Clear") {
            search_instance->clear_transposition_table();
        } else if (option_name == "Hash") {
            int tt_size_MB = atoi(option_value.c_str());
            search_instance->set_transposition_table_size_MB(tt_size_MB);
        } else if (option_name == "Threads") {
            int thread_count = atoi(option_value.c_str());
            search_instance->set_threads(thread_count);
        } else if (option_name == "MultiPV") {
            int multi_pv = atoi(option_value.c_str());
            search_instance->set_multi_pv(multi_pv);
        } else if (option_name == "UCI_ShowWDL") {
            if (option_value == "true") {
                show_wdl = true;
            } else if (option_value == "false") {
                show_wdl = false;
            }
        } else if (option_name == "Move Overhead") {
            int overhead = atoi(option_value.c_str());
            time_man->set_overhead(overhead);
        } else if (option_name == "Ponder") {
            if (option_value == "true") {
                ponder = true;
            } else if (option_value == "false") {
                ponder = false;
            }
        }
    } else {
        std::lock_guard<std::mutex> guard(non_uci_cmds_lock);

        non_uci_cmds.push(cmd_line);
    }
}



bool uci_interface::get_non_uci_cmd(std::string &str)
{
    std::lock_guard<std::mutex> guard(non_uci_cmds_lock);

    if (!non_uci_cmds.empty()) {
        str = non_uci_cmds.front();
        non_uci_cmds.pop();

        return true;
    }
    return false;
}

void uci_interface::iteration_end()
{
    int latest_depth = search_man->get_depth();
    if (latest_depth != last_info_depth) {

        int hashfull = search_instance->get_transpostion_table_usage_permill();
        int multi_pv = search_instance->get_multi_pv();
        uint64_t nps = search_man->get_nps();

        for (int i = last_info_depth+1; i <= latest_depth; i++) {
            search_result info = search_man->get_search_result(i);

            for (int j = 0; j < multi_pv; j++) {
                int32_t eval = scale_eval(info.lines[j]->score);

                if (info.lines[j]->score == MIN_EVAL) {
                    continue;
                }

                std::stringstream ss;
                ss << "info depth ";
                ss << info.depth;

                ss << " seldepth ";

                ss << info.seldepth;

                ss << " multipv " << j + 1;

                if (is_mate_score(eval)) {
                    ss << " score mate ";
                    if (eval > 0) {
                        ss << (get_mate_distance(eval)+1)/2;
                    } else {
                        ss << (-(get_mate_distance(eval)+1)/2);
                    }
                } else {
                    ss << " score cp ";
                    ss << eval;
                }

                if (show_wdl) {
                    ss << " wdl ";

                    float w, d, l;
                    wdl_model::get_wdl(game_instance->get_state(), info.lines[j]->score, w, d, l);
                    ss << (int)(w*1000) << " ";
                    ss << (int)(d*1000) << " ";
                    ss << (int)(l*1000);
                }

                ss << " nodes ";
                ss << info.nodes;

                ss << " nps ";
                ss << nps;
                ss << " hashfull ";
                ss << hashfull;

                ss << " time ";
                ss << info.time_ms;
                ss << " pv";

                for (int k = 0; k < info.lines[j]->num_of_moves; k++) {
                    ss << " ";
                    ss << info.lines[j]->moves[k].to_uci();
                }

                send_command(ss.str());
            }

        }
        last_info_depth = latest_depth;
    }
}

void uci_interface::search_end()
{
    if (ponder) {
        pv_table pv;
        search_man->get_pv(pv);
        send_command("bestmove " + pv.moves[0].to_uci() + (pv.num_of_moves > 2 ? " ponder " + pv.moves[1].to_uci() : ""));
    } else {
        send_command("bestmove " + search_man->get_move().to_uci());
    }
}


