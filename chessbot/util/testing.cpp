
#include "testing.hpp"
#include "../search.hpp"
#include "../game.hpp"
#include <memory>
#include <thread>
#include <fstream>
#include <atomic>
#include "misc.hpp"
#include "../timeman.hpp"


inline double sprt_llr(double elo0, double elo1, int wins, int draws, int losses)
{
    if (wins == 0 || draws == 0 || losses == 0) {
        return 0.0f;
    }

    int n = wins + draws + losses;

    int results[3] = {wins, draws, losses};

    double dr = (double)draws / n;

    double s0 = elo_to_score(elo0);
    double s1 = elo_to_score(elo1);

    double p0[3] = {s0 - 0.5*dr, dr, 1.0f - s0 - 0.5*dr};
    double p1[3] = {s1 - 0.5*dr, dr, 1.0f - s1 - 0.5*dr};

    double llr = 0;
    for (int i = 0; i < 3; i++) {
        llr += std::log(p1[i] / p0[i]) * results[i];
    }
    return llr;
}


struct test_worker
{
    test_worker()
    {
        static int worker_counter = 0;
        worker_id = worker_counter;
        worker_counter += 1;
    }

    void wait()
    {
        t.join();
    }

    void start(std::atomic<int> &games, int time, int time_inc, alphabeta_search &bot00, alphabeta_search &bot10, std::shared_ptr<std::vector<std::string>> opening_suite)
    {
        bot0 = std::make_unique<alphabeta_search>(bot00);
        bot1 = std::make_unique<alphabeta_search>(bot10);

        t = std::thread(&test_worker::play, this, std::ref(games), time, time_inc, opening_suite);
    }

    match_stats stats;
private:
    void play(std::atomic<int> &games, int base_time, int time_inc, std::shared_ptr<std::vector<std::string>> opening_suite)
    {
        std::srand((size_t)time(NULL) + worker_id);

        bool bot0_playing_black = (worker_id & 0x1);

        match_stats mstats;

        while (games > 0) {
            std::string opening_fen = (*opening_suite)[rand() % opening_suite->size()];


            play_game_pair(*bot0, *bot1, opening_fen, base_time, time_inc, bot0_playing_black, mstats);

            bot0_playing_black = !bot0_playing_black;
            stats = mstats;
            games -= 2;
        }
    }

    std::unique_ptr<alphabeta_search> bot0;
    std::unique_ptr<alphabeta_search> bot1;

    std::thread t;
    int worker_id;
};


void get_test_workers_results(std::vector<test_worker> &workers, double &bot0_depth, double &bot1_depth, double &bot0_nps, double &bot1_nps, double &elo, int &wins, int &draws, int &losses)
{
    int bot0_moves = 0;
    int bot1_moves = 0;

    bot0_depth = 0;
    bot1_depth = 0;

    wins = 0;
    draws = 0;
    losses = 0;

    for (size_t i = 0; i < workers.size(); i++) {
        wins += workers[i].stats.wins;
        losses += workers[i].stats.losses;
        draws += workers[i].stats.draws;

        bot0_depth += workers[i].stats.bot0_depth;
        bot1_depth += workers[i].stats.bot1_depth;

        bot0_moves += workers[i].stats.bot0_moves;
        bot1_moves += workers[i].stats.bot1_moves;

        bot0_nps += workers[i].stats.bot0_nps;
        bot1_nps += workers[i].stats.bot1_nps;
    }

    if (bot0_moves > 0 && bot1_moves > 0) {
        bot0_depth /= bot0_moves;
        bot1_depth /= bot1_moves;

        bot0_nps /= bot0_moves;
        bot1_nps /= bot1_moves;
    } else {
        bot0_depth = 0;
        bot1_depth = 0;

        bot0_nps = 0;
        bot1_nps = 0;
    }

    if (wins+losses+draws > 1) {
        elo = score_to_elo(0.5f * (double)(2*wins + draws) / (wins+losses+draws));
    } else {
        elo = 0.0f;
    }
}



int testing_utility::test(int games, int time, int time_inc, alphabeta_search &bot0, alphabeta_search &bot1, int threads, std::string opening_suite, double elo0, double elo1, double alpha, double beta)
{
    std::shared_ptr<std::vector<std::string>> openings = nullptr;
    if (opening_suite != "") {
        openings = std::make_shared<std::vector<std::string>>(load_opening_suite(opening_suite));
    }

    std::atomic<int> agames = games;
    std::vector<test_worker> workers(threads);

    for (size_t i = 0; i < workers.size(); i++) {
        workers[i].start(agames, time, time_inc, bot0, bot1, openings);
    }

    double bot0_depth, bot1_depth, bot0_nps, bot1_nps, elo;
    int wins, draws, losses;

    double lower = std::log(beta / (1 - alpha));
    double upper = std::log((1 - beta) / alpha);

    std::cout << std::endl;
    std::cout << "Elo0: " << elo0 << std::endl;
    std::cout << "Elo1: " << elo1 << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "Beta: " << beta << std::endl;
    std::cout << "LLR upper limit: " << upper << std::endl;
    std::cout << "LLR lower limit: " << lower << std::endl;
    std::cout << "Playing " << games << " games with " << (float)time/1000.0f << "s + " << time_inc << "ms" << " time control.  Threads: " << threads << std::endl;

    while (agames > 0) {
        get_test_workers_results(workers, bot0_depth, bot1_depth, bot0_nps, bot1_nps, elo, wins, draws, losses);

        double llr = sprt_llr(elo0, elo1, wins, draws, losses);

        std::cout << "\rD: "    << std::left << std::setw(8) << bot0_depth << " " << std::setw(8) << bot1_depth
                  << "  Knps: " << std::setw(8) << bot0_nps << " " << std::setw(8) << bot1_nps
                  << "  G: "    << games - agames << "/" << games
                  << "  R: "    << wins << "/" << draws << "/" << losses
                  << "  Elo: "  << elo
                  << "  LLR: "  << llr << "      ";


        if (llr < lower || llr > upper) {
            std::cout << "\nSPRT terminated!" << std::endl;
            agames = 0;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    for (size_t i = 0; i < workers.size(); i++) {
        workers[i].wait();
    }

    get_test_workers_results(workers, bot0_depth, bot1_depth, bot0_nps, bot1_nps, elo, wins, draws, losses);

    int total = wins + draws + losses;

    std::cout << std::endl << std::endl;

    std::cout << "Wins: " << wins << "(" << (wins*100) / total << "%)" << std::endl;
    std::cout << "Draws: " << draws << "(" << (draws*100) / total << "%)" << std::endl;
    std::cout << "Losses: " << losses << "(" << (losses*100) / total << "%)" << std::endl;

    std::cout << "Elo: " << elo << std::endl;

    std::cout << "Depth0: " << bot0_depth << std::endl;
    std::cout << "Depth1: " << bot1_depth << std::endl;
    std::cout << "Knps0: " << bot0_nps << std::endl;
    std::cout << "Knps1: " << bot1_nps << std::endl;

    return elo;
}
