#include "training.hpp"
#include "training_nnue.hpp"
#include <memory>
#include <math.h>
#include <condition_variable>
#include <algorithm>
#include <random>
#include <sstream>
#include <filesystem>
#include <random>

#include "../nnue.hpp"

#include "../../search.hpp"
#include "../../tuning.hpp"


void randomize_floats(float *f, int n, float std_mean, float std_deviation, std::mt19937 &gen)
{
    std::normal_distribution nd{std_mean, std_deviation};

    for (int i = 0; i < n; i++) {
        f[i] = nd(gen);
    }

}


void init_weights(training_weights &weights)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};

    weights.zero();


    float output_deviation = sqrt(2.0f / weights.layer0_weights.num_of_biases());
    float layer0_deviation = sqrt(1.0f / weights.perspective_weights.num_of_biases());
    float perspective_deviation = sqrt(2.0f / 768.0f);


    randomize_floats(weights.output_weights.weights, weights.output_weights.num_of_weights(), 0, output_deviation, gen);
    randomize_floats(weights.output_weights.biases, weights.output_weights.num_of_biases(), 0, output_deviation, gen);

    randomize_floats(weights.layer0_weights.weights, weights.layer0_weights.num_of_weights(), 0, layer0_deviation, gen);
    randomize_floats(weights.layer0_weights.biases, weights.layer0_weights.num_of_biases(), 0, layer0_deviation, gen);

    randomize_floats(weights.perspective_weights.biases, weights.perspective_weights.num_of_biases(), 0, perspective_deviation, gen);
    /*std::normal_distribution nd{0.0f, perspective_deviation};
    for (size_t i = 0; i < inputs_per_bucket; i++) {
        for (size_t j = 0; j < num_perspective_neurons; j++) {

            size_t input = num_of_king_buckets*inputs_per_bucket + i;

            weights.perspective_weights.weights[input*num_perspective_neurons + j] = nd(gen);

        }
    }*/

    for (size_t i = 0; i < inputs_per_bucket; i++) {
        for (size_t j = 0; j < num_perspective_neurons; j++) {

            size_t input = num_of_king_buckets*inputs_per_bucket + i;

            weights.perspective_weights.weights[input*num_perspective_neurons + j] = 0.01f;

        }
    }
}




void gradient_descent(training_weights &weights, training_weights &grad, float learning_rate)
{
    weights.output_weights.gradient_descent(&grad.output_weights, learning_rate);
    weights.layer0_weights.gradient_descent(&grad.layer0_weights, learning_rate);
    weights.perspective_weights.gradient_descent(&grad.perspective_weights, learning_rate);
}

void rmsprop(training_weights &weights, training_weights &grad, training_weights &past_gradients, float learning_rate)
{
    weights.output_weights.rmsprop(&grad.output_weights, &past_gradients.output_weights, learning_rate);
    weights.layer0_weights.rmsprop(&grad.layer0_weights, &past_gradients.layer0_weights, learning_rate);
    weights.perspective_weights.rmsprop(&grad.perspective_weights, &past_gradients.perspective_weights, learning_rate);
}


void back_propagate(training_network &net, training_weights &grad, float loss_delta, player_type_t stm)
{
    net.output_layer.grads[0] = loss_delta;

    net.output_layer.back_propagate(&grad.output_weights, net.layer0.grads, net.layer0.neurons);

    if (stm == WHITE) {
        net.layer0.back_propagate(&grad.layer0_weights, net.white_side.grads, net.black_side.grads, net.white_side.neurons, net.black_side.neurons);
    } else {
        net.layer0.back_propagate(&grad.layer0_weights, net.black_side.grads, net.white_side.grads, net.black_side.neurons, net.white_side.neurons);
    }

    net.white_side.back_propagate(&grad.perspective_weights);
    net.black_side.back_propagate(&grad.perspective_weights);
}


class semaphore {
    std::mutex mutex_;
    std::condition_variable condition_;
    unsigned long count_ = 0;

public:
    void signal() {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        ++count_;
        condition_.notify_one();
    }

    void wait() {
        std::unique_lock<decltype(mutex_)> lock(mutex_);
        while(!count_)
            condition_.wait(lock);
        --count_;
    }
};


int get_king_bucket_configuration(training_position &pos)
{
    uint64_t occupation = pos.occupation;
    int index = 0;
    int sq_index;
    piece p;

    int num_of_pieces = pos.count_pieces();
    int white_king_sq = 0;
    int black_king_sq = 0;
    for (int i = 0; i < num_of_pieces; i++) {
        p.d = pos.iterate_pieces(occupation, index, sq_index);

        if (p.get_player() == WHITE && p.get_type() == KING) {
            white_king_sq = sq_index;
        } else if (p.get_player() == BLACK && p.get_type() == KING) {
            black_king_sq = sq_index;
        }
    }
    black_king_sq ^= 56;

    return get_king_bucket(white_king_sq)*64+get_king_bucket(black_king_sq);
}


void loss(float pred, float target, float &loss, float &loss_delta)
{
    constexpr float exponent = 2.5f;

    //Loss = (pred - target)^exponent
    //DLoss = exponent*(pred - target)^(exponent-1)

    float s = (pred - target > 0 ? 1.0f : -1.0f);
    float d = std::abs(pred - target);

    loss = std::pow(d, exponent);
    loss_delta = exponent * s * std::pow(d, exponent - 1.0f);
}


struct worker_thread
{
    worker_thread(std::shared_ptr<training_weights> weights, int tid, int tc) : net(weights)
    {
        thread_id = tid;
        pool_size = tc;

        running = true;

        cost = 0;

        t = std::thread(&worker_thread::thread_entry, this);
    }

    ~worker_thread()
    {
        running = false;
        t.join();
    }

    void thread_entry() {

        srand(time(NULL) + thread_id);

        while (running) {
            thread_wait();

            gradient.zero();

            cost = 0;

            if (worker_batch.size() != (size_t)(batch_size / pool_size)) {
                worker_batch.resize(batch_size / pool_size);
            }

            int c = 0;
            for (int i = thread_id; i < batch_size; i += pool_size) {
                worker_batch[c].first = batch[i];
                worker_batch[c].second = get_king_bucket_configuration(batch[i]);
                c++;
            }

            std::sort(worker_batch.begin(), worker_batch.end(), [] (auto &a, auto &b) {return a.second > b.second;});


            for (size_t i = 0; i < worker_batch.size(); i++) {
                training_position sample = worker_batch[i].first;


                float num_of_pieces = pop_count(sample.occupation);

                float lambda = std::clamp(1.0f - ((num_of_pieces - 4.0f) / 16.0f), min_lambda, max_lambda);

                float result = sample.get_wdl_relative_to_stm();

                float pred_p = net.evaluate(sample);
                float eval_cp = sample.eval;

                float pred = sigmoid(pred_p  / 4.0f);
                float eval = sigmoid(eval_cp / 400.0f);

                float loss_eval, loss_result, loss_eval_delta, loss_result_delta;

                loss(pred, eval, loss_eval, loss_eval_delta);
                loss(pred, result, loss_result, loss_result_delta);

                float loss       = loss_eval       * (1.0f-lambda) + loss_result       * lambda;
                float loss_delta = loss_eval_delta * (1.0f-lambda) + loss_result_delta * lambda;

                back_propagate(net, gradient, loss_delta, sample.get_turn());

                cost += loss;
            }

            gradient.divide(batch_size);

            thread_signal_ready();
        }
    }


    void start(training_position *data, int worker_batch_size, float min_l, float max_l, bool use_factorizer)
    {
        batch = data;
        batch_size = worker_batch_size;
        min_lambda = min_l;
        max_lambda = max_l;

        net.use_factorizer = use_factorizer;

        begin_signal.signal();
    }
    void wait() {
        finish_signal.wait();
    }

    float cost;

    training_weights gradient;
private:
    training_network net;

    std::vector<std::pair<training_position, int>> worker_batch;

    int batch_size;
    float min_lambda;
    float max_lambda;

    int thread_id;
    int pool_size;

    std::thread t;
    training_position *batch;

    std::atomic<bool> running;


    semaphore begin_signal;
    semaphore finish_signal;


    void thread_wait()
    {
        begin_signal.wait();
    }

    void thread_signal_ready()
    {
        finish_signal.signal();
    }
};


struct worker_thread_pool
{
    worker_thread_pool(int num_of_workers, std::shared_ptr<training_weights> weights) {
        for (int i = 0; i < num_of_workers; i++) {
            threads.push_back(new worker_thread(weights, i, num_of_workers));
        }
        prev_batch_size = 0;
    }

    ~worker_thread_pool()
    {
        for (size_t i = 0; i < threads.size(); i++) {
            delete threads[i];
        }
        threads.clear();
    }


    void sync(training_position *batch, training_weights &gradient, int batch_size, float min_lambda, float max_lambda, bool use_factorizer, float &avg_cost)
    {
        gradient.zero();

        avg_cost = 0;

        if (prev_batch_size > 0) {
            for (size_t i = 0; i < threads.size(); i++) {
                threads[i]->wait();

                gradient.add(threads[i]->gradient);

                avg_cost += threads[i]->cost;

                threads[i]->start(batch, batch_size, min_lambda, max_lambda, use_factorizer);
            }
            //gradient.normalize();

            avg_cost = avg_cost / prev_batch_size;
        } else {
            std::for_each(threads.begin(), threads.end(), [batch, batch_size, use_factorizer, min_lambda, max_lambda] (auto p) {p->start(batch, batch_size, min_lambda, max_lambda, use_factorizer);});
        }
        prev_batch_size = batch_size;
    }

    void stop()
    {
        if (prev_batch_size == 0) {
            return;
        }

        std::for_each(threads.begin(), threads.end(), [] (auto p) {p->wait();});

        prev_batch_size = 0;
    }

    int get_pool_size() {
        return threads.size();
    }


private:
    std::vector<worker_thread*> threads;
    int prev_batch_size;
};


void nnue_trainer::test_nets(std::string training_net_file, std::string quantized_net_file, std::vector<selfplay_result> &test_set)
{
    std::cout << "Testing networks." << std::endl;

    std::cout << "Loading training nnue..." << std::endl;

    std::shared_ptr<training_weights> training_nnue_weights = std::make_shared<training_weights>();
    training_nnue_weights->load_file(training_net_file);
    training_network training_nnue(training_nnue_weights);

    training_nnue_weights->save_quantized(quantized_net_file);

    std::cout << "Loading quantized nnue..." << std::endl;

    std::shared_ptr<nnue_weights> quantized_nnue_weights = std::make_shared<nnue_weights>();
    quantized_nnue_weights->load(quantized_net_file);
    nnue_network quantized_nnue(quantized_nnue_weights);


    float avg_error_nnue = 0.0f;
    float avg_error_qnnue = 0.0f;
    float avg_error_static_eval = 0.0f;
    float avg_quantization_error = 0.0f;
    float worst_quantization_error = 0.0f;

    float avg_error_nnue_black = 0.0f;
    float avg_error_nnue_white = 0.0f;

    int black_positions = 0;
    int white_positions = 0;

    board_state state;

    int positions = 0;

    std::cout << "Comparing..." << std::endl;
    for (size_t i = 0; i < test_set.size(); i++) {
        state.load_fen(test_set[i].fen);

        if (state.in_check(state.get_turn()) || state.get_square(test_set[i].bm.to).get_type() != EMPTY) {
            continue;
        }
        positions++;

        float real = test_set[i].eval / 100.0f;
        float qeval = (float)quantized_nnue.evaluate(state) / 100.0f;
        float eval = training_nnue.evaluate(state);

        avg_error_nnue += (real - eval)*(real - eval);
        avg_error_qnnue += (real - qeval)*(real - qeval);

        if (state.get_turn() == WHITE) {
            avg_error_nnue_white += (real - eval)*(real - eval);
            white_positions += 1;
        } else {
            avg_error_nnue_black += (real - eval)*(real - eval);
            black_positions += 1;
        }

        if (worst_quantization_error < std::abs(eval - qeval)) {
            worst_quantization_error = std::abs(eval - qeval);
        }

        float abs_error = std::abs(eval - qeval);
        if (abs_error > 0.1) {
            std::cout << "error: " << abs_error << "  Eval " << eval << "  Qeval " << qeval << std::endl;
        }


        avg_quantization_error += (eval - qeval)*(eval - qeval);
    }
    avg_error_nnue = sqrt(avg_error_nnue / positions);
    avg_error_qnnue = sqrt(avg_error_qnnue / positions);
    avg_error_static_eval = sqrt(avg_error_static_eval / positions);
    avg_quantization_error = sqrt(avg_quantization_error / positions);

    avg_error_nnue_black = sqrt(avg_error_nnue_black / black_positions);
    avg_error_nnue_white = sqrt(avg_error_nnue_white / white_positions);

    std::cout << "NNUE: " << avg_error_nnue << "  Quantized NNUE: " << avg_error_qnnue << "   Static eval: " << avg_error_static_eval << std::endl;
    std::cout << "Avg quantization error: " << avg_quantization_error << "   Worst quantization error: " << worst_quantization_error << std::endl;
    std::cout << "White NNUE: " << avg_error_nnue_white << " Black NNUE: " << avg_error_nnue_black << std::endl;
}



void nnue_trainer::quantize_net(std::string net_file, std::string qnet_file)
{
    std::shared_ptr<training_weights> weights = std::make_shared<training_weights>();
    weights->load_file(net_file);
    training_network training_nnue(weights);

    weights->save_quantized(qnet_file);
}


void training_loop(std::string net_file, std::string qnet_file, std::shared_ptr<training_weights> weights, std::shared_ptr<data_reader> dataset)

{
    std::srand(time(NULL));

    training_weights gradient;
    training_weights gradient_sq;

    training_weights first_moment;
    training_weights second_moment;

    training_weights corrected_first_moment;
    training_weights corrected_second_moment;


    first_moment.zero();
    second_moment.zero();

    worker_thread_pool thread_pool(16, weights);

    training_network net(weights);

    float learning_rate = 0.00001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    int batch_size = 4000*thread_pool.get_pool_size();
    int epoch_size = 100000000;
    float min_lambda = 0.25f;
    float max_lambda = 0.50f;

    training_batch_manager batch_manager(batch_size, epoch_size, dataset);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto last_save_time = t0;

    float batch_cost;
    float training_cost = 0.0f;


    std::cout << std::endl;
    std::cout << "Dataset size: " << dataset->get_size<training_position>() / (1000*1000) << "M" << std::endl;
    std::cout << "Beta1: " << beta1 << std::endl;
    std::cout << "Beta2: " << beta2 << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << "Min lambda: " << min_lambda << std::endl;
    std::cout << "Max lambda: " << max_lambda << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Epoch size: " << epoch_size << std::endl;
    std::cout << "Threads: " << thread_pool.get_pool_size() << std::endl << std::endl;

    while (true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 );
        t0 = t1;


        batch_manager.load_new_batch();

        thread_pool.sync(batch_manager.get_current_batch(), gradient, batch_size, min_lambda, max_lambda, net.use_factorizer, batch_cost);

        if (training_cost == 0.0f) {
            training_cost = batch_cost;
        } else {
            training_cost = training_cost * 0.99f + batch_cost * 0.01f;
        }

        gradient_sq.squared(gradient);


        first_moment.mult(beta1);
        gradient.mult(1.0f - beta1);
        first_moment.add(gradient);

        second_moment.mult(beta2);
        gradient_sq.mult(1.0f - beta2);
        second_moment.add(gradient_sq);

        corrected_first_moment.copy_from(first_moment);
        corrected_second_moment.copy_from(second_moment);

        corrected_first_moment.mult(1.0f / (1.0f - beta1));
        corrected_second_moment.mult(1.0f / (1.0f - beta2));

        //gradient_descent(*weights, gradient_with_momentum, learning_rate);
        rmsprop(*weights, corrected_first_moment, corrected_second_moment, learning_rate);

        if (std::chrono::duration_cast<std::chrono::seconds>( t0 - last_save_time ).count() > 60) {
            last_save_time = t0;

            weights->save_file(net_file);
            weights->save_quantized(qnet_file);

            std::cout << "Net saved!" << std::endl;
        }

        float kspers = std::clamp((float)batch_size / ms.count(), 0.0f, 9999.0f);

        std::cout << "\rTrC: " << std::setprecision(6) << std::left << std::setw(12) << training_cost
                  << "   BC: "  << std::left << std::setw(12) << batch_cost
                  << "   Speed: " << std::right << std::setw(4) << (int)kspers << " KPos/s    "
                  << "   Epoch: " << batch_manager.get_epochs() << " (" << std::setprecision(3) << std::setw(4) << std::right << (float)batch_manager.get_current_batch_number()*100.0f / batch_manager.get_number_of_batches() << "%)   ";
    }

}


void nnue_trainer::train(std::string net_file, std::string qnet_file, std::vector<std::string> selfplay_directories)
{
    std::shared_ptr<data_reader> reader = std::make_shared<data_reader>(selfplay_directories);

    std::shared_ptr<training_weights> weights = std::make_shared<training_weights>();

    init_weights(*weights);
    weights->load_file(net_file);

    training_loop(net_file, qnet_file, weights, reader);
}













