#include "uci.hpp"
#include "time_managment.hpp"
#include <sstream>

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


uci_interface::uci_interface(std::shared_ptr<alphabeta_search> search_inst, std::shared_ptr<game_state> game_inst)
{
    running = false;
    search_instance = search_inst;
    game_instance = game_inst;
    exit_flag = false;

    time_man = std::make_shared<time_manager>(0,0);

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
    running = true;
    uci_thread = std::thread(&uci_interface::main_loop, this);
    input_thread = std::thread(&uci_interface::input_loop, this);
}

void uci_interface::stop()
{
    if (!running) {
        return;
    }
    running = false;

    uci_thread.join();
    input_thread.join();
}

std::string uci_interface::parse_command(std::string line, int word)
{
    std::vector<std::string> words = split_string(line, ' ');
    if (word < (int)words.size() && word >= 0) {
        return words[word];
    }
    return std::string("");
}


void uci_interface::input_loop()
{
    std::string input_str;
    while (running) {
        while (std::getline(std::cin, input_str)) {
            input_lock.lock();
            inputs.push(input_str);
            input_lock.unlock();

            if (input_str == "quit") {
                return;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}


bool uci_interface::get_command(std::string &str_out)
{
    bool r = false;

    input_lock.lock();
    if (!inputs.empty()) {
        str_out = inputs.front();
        inputs.pop();
        r = true;

        if (uci_log.is_open()) {
            uci_log << "GUI: " << str_out << "\n";
        }

    }
    input_lock.unlock();

    return r;
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

        send_command("uciok");
    } else if (cmd == "isready") {
        send_command("readyok");
    } else if (cmd == "position") {
        if (search_instance->is_searching()) {
            search_instance->stop_search();
        }

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
        if (search_instance->is_searching()) {
            search_instance->stop_search();
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
                    info.type = GO_INFINITE;
                    search_instance->limit_search(std::atoi(next_it->c_str()), 1);
                }
               if (str == "nodes") {
                    info.type = GO_INFINITE;
                    search_instance->limit_search(0, std::atoi(next_it->c_str()));
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
            search_instance->begin_search(game_instance->get_state(), time_man);
        } else {
            search_instance->begin_search(game_instance->get_state());
        }

        last_info_depth = 0;
    } else if (cmd == "stop") {
        if (search_instance->is_searching()) {
            search_instance->stop_search();

            send_command("bestmove " + search_instance->get_move().to_uci());

            ginfo.active = false;
        }
    } else if (cmd == "ucinewgame") {
        if (search_instance->is_searching()) {
            search_instance->stop_search();
        }
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
        }
    }
}


void uci_interface::send_info_strings(int search_time)
{
    int latest_depth = search_instance->get_depth();
    if (latest_depth != last_info_depth) {

        int hashfull = search_instance->get_transpostion_table_usage_permill();
        int multi_pv = search_instance->get_multi_pv();
        uint64_t nps = search_instance->get_nps();

        for (int i = last_info_depth+1; i <= latest_depth; i++) {
            search_info info = search_instance->get_search_info(i);

            for (int j = 0; j < multi_pv; j++) {
                int32_t eval = info.lines[j]->score;

                if (eval == MIN_EVAL) {
                    continue;
                }
                //std::string bm = info.lines[j]->moves[0].to_uci();

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


                ss << " nodes ";
                ss << info.nodes;

                ss << " nps ";
                ss << nps;
                ss << " hashfull ";
                ss << hashfull;
                //ss << " bm ";
                //ss << bm;

                ss << " time ";
                ss << search_time;
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


void uci_interface::update()
{
    if (!ginfo.active) {
        return;
    }

    int search_time = search_instance->get_search_time_ms();

    send_info_strings(search_time);


    if (search_instance->is_ready_to_stop()) {
        search_instance->stop_search();

        send_info_strings(search_time);

        send_command("bestmove " + search_instance->get_move().to_uci());
        ginfo.active = false;

        return;
    }


    if (ginfo.type == GO_MOVETIME && search_time >= ginfo.movetime) {
        search_instance->stop_search();

        send_command("bestmove " + search_instance->get_move().to_uci());
        ginfo.active = false;

    }
}


void uci_interface::main_loop()
{
    while (running) {
        std::this_thread::sleep_for (std::chrono::milliseconds(1));

        std::string cmd;
        if (get_command(cmd)) {
            execute_command(cmd);
        } else {
            update();
        }
    }
}


#ifdef __MINGW64__

std::string uci_engine::read_line(HANDLE stdout_rd)
{
    std::string line;
    while (true) {
        char c;
        DWORD red = 0;
        ReadFile(stdout_rd, &c, 1, &red, NULL);
        if (red > 0) {
            if (c == '\n') {
                break;
            } else {
                line += c;
            }
        }
    }

    comm_log << "ENGI: " << line << "\n";

    return line;
}

void uci_engine::write_line(std::string line, HANDLE stdin_wr)
{
    if (line[line.length()-1] != '\n') {
        line += '\n';
    }
    comm_log << "HOST: " << line;

    WriteFile(stdin_wr, line.c_str(), line.length(), NULL, NULL);
}

#endif // __MINGW64__


uci_engine::uci_engine()
{
    #ifdef __MINGW64__

    stdin_rd = NULL;
    stdin_wr = NULL;
    stdout_rd = NULL;
    stdout_wr = NULL;

    #endif // __MINGW64__
}

uci_engine::~uci_engine()
{
    #ifdef __MINGW64__

    if (stdin_rd == NULL) {
        return;
    }

    #endif // __MINGW64__

    quit();
}


void uci_engine::set_option(std::string name, int value)
{
    #ifdef __MINGW64__

    std::stringstream ss;
    ss << "setoption name " << name << " value " << value;
    write_line(ss.str(), stdin_wr);

    write_line("isready", stdin_wr);
    std::string line = read_line(stdout_rd);
    while (line.find("readyok") == std::string::npos) {
         line = read_line(stdout_rd);
    }

    #endif // __MINGW64__
}


void uci_engine::launch(std::string engine_path)
{
    #ifdef __MINGW64__

    stdin_rd = NULL;
    stdin_wr = NULL;
    stdout_rd = NULL;
    stdout_wr = NULL;

    SECURITY_ATTRIBUTES sa_attr;

    sa_attr.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa_attr.bInheritHandle = TRUE;
    sa_attr.lpSecurityDescriptor = NULL;

    CreatePipe(&stdout_rd, &stdout_wr, &sa_attr, 0);
    SetHandleInformation(stdout_rd, HANDLE_FLAG_INHERIT, 0);
    CreatePipe(&stdin_rd, &stdin_wr, &sa_attr, 0);
    SetHandleInformation(stdin_wr, HANDLE_FLAG_INHERIT, 0);


    PROCESS_INFORMATION proc_info;
    STARTUPINFO start_info;

    ZeroMemory( &proc_info, sizeof(PROCESS_INFORMATION) );

    ZeroMemory( &start_info, sizeof(STARTUPINFO) );
    start_info.cb = sizeof(STARTUPINFO);
    start_info.hStdError = stdout_wr;
    start_info.hStdOutput = stdout_wr;
    start_info.hStdInput = stdin_rd;
    start_info.dwFlags |= STARTF_USESTDHANDLES;

    CreateProcess(NULL, (char*)engine_path.c_str(), NULL, NULL, TRUE, 0, NULL, NULL, &start_info, &proc_info);


    write_line("uci", stdin_wr);
    std::string line = read_line(stdout_rd);
    while (line.find("uciok") == std::string::npos) {
         line = read_line(stdout_rd);
    }

    write_line("isready", stdin_wr);
    line = read_line(stdout_rd);
    while (line.find("readyok") == std::string::npos) {
         line = read_line(stdout_rd);
    }

    #endif // __MINGW64__
}



chess_move uci_engine::go_clock(const game_state &game, int wtime, int winc, int btime, int binc, int &depth, int &nodes, int &time_searched)
{
    #ifdef __MINGW64__

    std::stringstream ss;
    if (game.moves_played.size() > 0) {
        ss << "position startpos moves";
        for (size_t i = 0; i < game.moves_played.size(); i++) {
            ss << " " << game.moves_played[i].to_uci();
        }
    } else {
        ss << "position startpos";
    }
    write_line(ss.str(), stdin_wr);
    ss.str("");
    ss << "go wtime " << wtime << " btime " << btime << " winc " << winc << " binc " << binc;

    write_line(ss.str(), stdin_wr);

    std::chrono::time_point t0 = std::chrono::high_resolution_clock::now();

    chess_move bm = chess_move::null_move();
    depth = 0;
    nodes = 0;

    std::string line = read_line(stdout_rd);
    while (line.find("bestmove") == std::string::npos) {
        std::vector<std::string> parts = split_string(line, ' ');

        for (auto it = parts.begin(); it != parts.end(); it++) {
            if (*it == "nodes") {
                nodes = atoi((it+1)->c_str());
            }
            if (*it == "depth") {
                depth = atoi((it+1)->c_str());
            }
        }
        line = read_line(stdout_rd);
    }
    std::chrono::time_point t1 = std::chrono::high_resolution_clock::now();


    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 );
    time_searched = ms.count();

    std::vector<std::string> parts = split_string(line, ' ');
    bm.from_uci(*(std::find(parts.begin(), parts.end(), "bestmove")+1));

    return bm;

    #endif
}

void uci_engine::new_game()
{
    #ifdef __MINGW64__

    write_line("ucinewgame", stdin_wr);
    write_line("isready", stdin_wr);
    std::string line = read_line(stdout_rd);
    while (line.find("readyok") == std::string::npos) {
         line = read_line(stdout_rd);
    }

    #endif // __MINGW64__
}


void uci_engine::quit()
{

    #ifdef __MINGW64__

    write_line("quit", stdin_wr);

    CloseHandle(stdin_rd);
    CloseHandle(stdin_wr);
    CloseHandle(stdout_rd);
    CloseHandle(stdout_wr);

    #endif // __MINGW64__
}




























