
#include "testing.hpp"
#include "../search.hpp"
#include "../game.hpp"
#include <memory>
#include <thread>
#include <fstream>
#include <atomic>
#include <iomanip>
#include "misc.hpp"
#include "../time_manager.hpp"


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

    void start(std::atomic<int> &games, int time, int time_inc, searcher &s00, searcher &s10, std::shared_ptr<std::vector<std::string>> opening_suite)
    {
        s0 = std::make_unique<searcher>(s00);
        s1 = std::make_unique<searcher>(s10);

        t = std::thread(&test_worker::play, this, std::ref(games), time, time_inc, opening_suite);
    }

    match_stats stats;
private:
    void play(std::atomic<int> &games, int base_time, int time_inc, std::shared_ptr<std::vector<std::string>> opening_suite)
    {
        std::srand((size_t)time(NULL) + worker_id);

        bool s0_playing_black = (worker_id & 0x1);

        match_stats mstats;

        while (games > 0) {
            std::string opening_fen = (*opening_suite)[rand() % opening_suite->size()];


            play_game_pair(*s0, *s1, opening_fen, base_time, time_inc, s0_playing_black, mstats);

            s0_playing_black = !s0_playing_black;
            stats = mstats;
            games -= 2;
        }
    }

    std::unique_ptr<searcher> s0;
    std::unique_ptr<searcher> s1;

    std::thread t;
    int worker_id;
};


void get_test_workers_results(std::vector<test_worker> &workers, double &s0_depth, double &s1_depth, double &s0_nps, double &s1_nps, double &elo, int &wins, int &draws, int &losses)
{
    int s0_moves = 0;
    int s1_moves = 0;

    s0_depth = 0;
    s1_depth = 0;

    wins = 0;
    draws = 0;
    losses = 0;

    for (size_t i = 0; i < workers.size(); i++) {
        wins += workers[i].stats.wins;
        losses += workers[i].stats.losses;
        draws += workers[i].stats.draws;

        s0_depth += workers[i].stats.s0_depth;
        s1_depth += workers[i].stats.s1_depth;

        s0_moves += workers[i].stats.s0_moves;
        s1_moves += workers[i].stats.s1_moves;

        s0_nps += workers[i].stats.s0_nps;
        s1_nps += workers[i].stats.s1_nps;
    }

    if (s0_moves > 0 && s1_moves > 0) {
        s0_depth /= s0_moves;
        s1_depth /= s1_moves;

        s0_nps /= s0_moves;
        s1_nps /= s1_moves;
    } else {
        s0_depth = 0;
        s1_depth = 0;

        s0_nps = 0;
        s1_nps = 0;
    }

    if (wins+losses+draws > 1) {
        elo = score_to_elo(0.5f * (double)(2*wins + draws) / (wins+losses+draws));
    } else {
        elo = 0.0f;
    }
}



int testing_utility::test(int games, int time, int time_inc, searcher &s0, searcher &s1, int threads, std::string opening_suite, double elo0, double elo1, double alpha, double beta)
{
    std::shared_ptr<std::vector<std::string>> openings = nullptr;
    if (opening_suite != "") {
        openings = std::make_shared<std::vector<std::string>>(load_opening_suite(opening_suite));
    }

    std::atomic<int> agames = games;
    std::vector<test_worker> workers(threads);

    for (size_t i = 0; i < workers.size(); i++) {
        workers[i].start(agames, time, time_inc, s0, s1, openings);
    }

    double s0_depth, s1_depth, s0_nps, s1_nps, elo;
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
    std::cout << "Opening suite: " << opening_suite << " (" << openings->size() << " positions)" << std::endl;
    std::cout << "Playing " << games << " games with " << (float)time/1000.0f << "s + " << time_inc << "ms" << " time control.  Threads: " << threads << std::endl;

    while (agames > 0) {
        get_test_workers_results(workers, s0_depth, s1_depth, s0_nps, s1_nps, elo, wins, draws, losses);

        double llr = sprt_llr(elo0, elo1, wins, draws, losses);

        std::cout << "\rD: "    << std::left << std::setw(8) << s0_depth << " " << std::setw(8) << s1_depth
                  << "  Knps: " << std::setw(8) << s0_nps << " " << std::setw(8) << s1_nps
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

    get_test_workers_results(workers, s0_depth, s1_depth, s0_nps, s1_nps, elo, wins, draws, losses);

    int total = wins + draws + losses;

    std::cout << std::endl << std::endl;

    std::cout << "Wins: " << wins << "(" << (wins*100) / total << "%)" << std::endl;
    std::cout << "Draws: " << draws << "(" << (draws*100) / total << "%)" << std::endl;
    std::cout << "Losses: " << losses << "(" << (losses*100) / total << "%)" << std::endl;

    std::cout << "Elo: " << elo << std::endl;

    std::cout << "Depth0: " << s0_depth << std::endl;
    std::cout << "Depth1: " << s1_depth << std::endl;
    std::cout << "Knps0: " << s0_nps << std::endl;
    std::cout << "Knps1: " << s1_nps << std::endl;

    return elo;
}
