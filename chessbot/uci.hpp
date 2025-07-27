#pragma once

#include <memory>
#include "search.hpp"
#include "game.hpp"
#include <queue>
#include <sstream>
#include <fstream>

#ifdef __MINGW64__
#include <windows.h>
#endif


enum go_type_t {GO_PONDER, GO_CLOCK, GO_MOVETIME, GO_INFINITE};
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

    int movetime;
};


struct uci_interface
{
    uci_interface(std::shared_ptr<alphabeta_search> search_inst, std::shared_ptr<game_state> game_inst);
    ~uci_interface();

    uci_interface(const uci_interface&) = delete;
    uci_interface& operator = (const uci_interface&) = delete;

    void start();
    void stop();

    bool exit();
private:
    go_info ginfo;

    std::string parse_command(std::string line, int word);

    void execute_command(std::string cmd_line);

    void send_command(std::string str);

    bool get_command(std::string &str_out);


    void update();


    void main_loop();
    void input_loop();

    int last_info_depth;

    std::atomic<bool> exit_flag;
    std::atomic<bool> running;
    std::thread uci_thread;
    std::thread input_thread;

    std::queue<std::string> inputs;
    std::mutex input_lock;

    std::shared_ptr<alphabeta_search> search_instance;
    std::shared_ptr<game_state> game_instance;
    std::shared_ptr<time_manager> time_man;

    std::ofstream uci_log;
};

std::vector<std::string> split_string(std::string str, char d);

struct uci_engine
{
    ~uci_engine();
    uci_engine();

    void launch(std::string engine_path);

    void set_option(std::string name, int value);

    void new_game();
    chess_move go_clock(const game_state &game, int wtime, int winc, int btime, int binc, int &depth_out, int &nodes_out, int &time_out);
    void quit();

    std::stringstream comm_log;

private:
    #ifdef __MINGW64__

    HANDLE stdin_rd;
    HANDLE stdin_wr;
    HANDLE stdout_rd;
    HANDLE stdout_wr;


    std::string read_line(HANDLE stdout_rd);
    void write_line(std::string line, HANDLE stdin_wr);

    #endif
};
