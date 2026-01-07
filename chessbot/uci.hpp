#pragma once

#include <memory>
#include "search.hpp"
#include "game.hpp"
#include <queue>
#include <sstream>
#include <fstream>
#include "time_manager.hpp"


enum go_type_t {GO_PONDER, GO_CLOCK, GO_MOVETIME, GO_DEPTH, GO_NODES, GO_INFINITE};
struct go_info
{
    go_info() {
        active = false;
        wtime = 0;
        btime = 0;
        winc = 0;
        binc = 0;
        movetime = 0;
    }

    bool active;

    go_type_t type;
    int wtime;
    int btime;
    int winc;
    int binc;

    int depth;
    int nodes;

    int movetime;
};


struct uci_interface
{
    uci_interface(std::shared_ptr<searcher> search_inst, std::shared_ptr<game_state> game_inst);
    ~uci_interface();

    uci_interface(const uci_interface&) = delete;
    uci_interface& operator = (const uci_interface&) = delete;

    bool get_non_uci_cmd(std::string &str);

    void start();
    void stop();

    bool exit();

    void iteration_end();
    void search_end();
private:
    go_info ginfo;

    std::string parse_command(std::string line, int word);
    void execute_command(std::string cmd_line);
    void send_command(std::string str);


    void input_loop();
    void search_thread_entry();

    int last_info_depth;

    std::atomic<bool> exit_flag;
    std::atomic<bool> running;
    std::thread uci_thread;
    std::thread search_thread;

    std::queue<std::string> non_uci_cmds;
    std::mutex non_uci_cmds_lock;

    std::shared_ptr<searcher> search_instance;
    std::shared_ptr<game_state> game_instance;

    std::shared_ptr<time_manager> time_man;
    std::shared_ptr<search_manager> search_man;

    std::ofstream uci_log;

    bool show_wdl;
    bool ponder;
};

std::vector<std::string> split_string(std::string str, char d);

