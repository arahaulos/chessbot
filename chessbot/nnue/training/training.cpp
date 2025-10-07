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
#include "save_bmp.hpp"
#include "../nnue.hpp"

#include "../../search.hpp"

void randomize_floats(float *f, int n, float std_mean, float std_deviation, std::mt19937 &gen)
{
    std::normal_distribution nd{std_mean, std_deviation};

    for (int i = 0; i < n; i++) {
        f[i] = nd(gen);
    }

}


void init_weights(training_weights &weights, bool init_perspectives_weights)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};


    if (init_perspectives_weights) {
        weights.zero();
    }


    float layer1_deviation = sqrt(1.0f / num_perspective_neurons);
    float layer2_deviation = sqrt(2.0f / layer1_neurons);
    float output_deviation = sqrt(2.0f / layer2_neurons);

    float perspective_deviation = sqrt(2.0f / 768.0f);


    randomize_floats(weights.output_weights.weights, weights.output_weights.num_of_weights(), 0, output_deviation, gen);
    randomize_floats(weights.output_weights.biases, weights.output_weights.num_of_biases(), 0, output_deviation, gen);

    randomize_floats(weights.layer2_weights.weights, weights.layer2_weights.num_of_weights(), 0, layer2_deviation, gen);
    randomize_floats(weights.layer2_weights.biases, weights.layer2_weights.num_of_biases(), 0, layer2_deviation, gen);

    randomize_floats(weights.layer1_weights.weights, weights.layer1_weights.num_of_weights(), 0, layer1_deviation, gen);
    randomize_floats(weights.layer1_weights.biases, weights.layer1_weights.num_of_biases(), 0, layer1_deviation, gen);

    weights.output_weights.broadcast_bucket_weights(0);
    weights.layer2_weights.broadcast_bucket_weights(0);
    weights.layer1_weights.broadcast_bucket_weights(0);

    if (init_perspectives_weights) {
        randomize_floats(weights.perspective_weights.biases, weights.perspective_weights.num_of_biases(), 0, perspective_deviation, gen);

        for (size_t i = 0; i < inputs_per_bucket; i++) {
            for (size_t j = 0; j < num_perspective_neurons; j++) {

                size_t input = num_of_king_buckets*inputs_per_bucket + i;

                weights.perspective_weights.weights[input*num_perspective_neurons + j] = 0.01f;

            }
        }
    }
}

void back_propagate(training_network &net, training_weights &grad, float loss_delta, player_type_t stm)
{
    net.output_layer.grads[0] = loss_delta;

    net.output_layer.back_propagate(net.output_bucket, &grad.output_weights, net.layer2.grads, net.layer2.neurons);
    net.layer2.back_propagate(net.output_bucket, &grad.layer2_weights, net.layer1.grads, net.layer1.neurons);

    if (stm == WHITE) {
        net.layer1.back_propagate(net.output_bucket, &grad.layer1_weights, net.white_side.grads, net.black_side.grads, net.white_side.neurons, net.black_side.neurons);
    } else {
        net.layer1.back_propagate(net.output_bucket, &grad.layer1_weights, net.black_side.grads, net.white_side.grads, net.black_side.neurons, net.white_side.neurons);
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

void loss(float pred, float target, float &loss, float &loss_delta)
{
    constexpr float exponent = 2.5f;
    //constexpr float exponent = 2.0f;

    //Loss = (pred - target)^exponent
    //DLoss = exponent*(pred - target)^(exponent-1)

    float s = (pred - target > 0 ? 1.0f : -1.0f);
    float d = std::abs(pred - target);

    loss = std::pow(d, exponent);
    loss_delta = exponent * s * std::pow(d, exponent - 1.0f);
}


enum worker_operation {WORK_BACKPROP, WORK_GRAD_CALC, WORK_RMSPROP};
struct worker_thread
{
    worker_thread(std::shared_ptr<training_weights> weights,
                  std::shared_ptr<training_weights> g,
                  std::shared_ptr<training_weights> g_sq,
                  std::shared_ptr<training_weights> m0,
                  std::shared_ptr<training_weights> m1,
                  std::shared_ptr<training_weights> cm0,
                  std::shared_ptr<training_weights> cm1, int tid, int tc) :
                  net(weights),
                  gradient(g),
                  gradient_sq(g_sq),
                  first_moment(m0),
                  second_moment(m1),
                  corrected_first_moment(cm0),
                  corrected_second_moment(cm1)
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
        while (running) {
            thread_wait();

            if (operation == WORK_BACKPROP) {

                backprop_gradient.zero();

                cost = 0;

                int per_thread = batch_size / pool_size;
                int start = per_thread * thread_id;

                for (size_t i = start; i < start + per_thread; i++) {
                    training_position sample = batch[i];

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

                    float sigmoid_delta = pred * (1.0f - pred);

                    back_propagate(net, backprop_gradient, loss_delta*sigmoid_delta, sample.get_turn());

                    cost += loss;
                }

                backprop_gradient.divide(batch_size);
            } else if (operation == WORK_GRAD_CALC) {
                gradient->zero(thread_id, pool_size);
                for (int i = 0; i < grads_to_add.size(); i++) {
                    gradient->add(*grads_to_add[i], thread_id, pool_size);
                }

                gradient_sq->squared(*gradient, thread_id, pool_size);

                first_moment->exponential_smoothing(*gradient, beta1, thread_id, pool_size);
                second_moment->exponential_smoothing(*gradient_sq, beta2, thread_id, pool_size);

                corrected_first_moment->mult(*first_moment, 1.0f / (1.0f - std::pow(beta1, step)), thread_id, pool_size);
                corrected_second_moment->mult(*second_moment, 1.0f / (1.0f - std::pow(beta2, step)), thread_id, pool_size);
            } else if (operation == WORK_RMSPROP) {
                net.weights->output_weights.rmsprop(&corrected_first_moment->output_weights, &corrected_second_moment->output_weights, learning_rate, thread_id, pool_size);

                net.weights->layer2_weights.rmsprop(&corrected_first_moment->layer2_weights, &corrected_second_moment->layer2_weights, learning_rate, thread_id, pool_size);
                net.weights->layer1_weights.rmsprop(&corrected_first_moment->layer1_weights, &corrected_second_moment->layer1_weights, learning_rate, thread_id, pool_size);

                net.weights->perspective_weights.rmsprop(&corrected_first_moment->perspective_weights, &corrected_second_moment->perspective_weights, learning_rate, thread_id, pool_size);
            }

            thread_signal_ready();
        }
    }


    void start_backprop(training_position *data, int worker_batch_size, float min_l, float max_l, bool use_factorizer)
    {
        operation = WORK_BACKPROP;

        batch = data;
        batch_size = worker_batch_size;
        min_lambda = min_l;
        max_lambda = max_l;

        net.use_factorizer = use_factorizer;

        begin_signal.signal();
    }

    void start_grad_calc(std::vector<training_weights*> grads, float b1, float b2, int t)
    {
        operation = WORK_GRAD_CALC;
        grads_to_add = grads;
        beta1 = b1;
        beta2 = b2;
        step = t;
        begin_signal.signal();
    }

    void start_rmsprop(float lr)
    {
        operation = WORK_RMSPROP;
        learning_rate = lr;
        begin_signal.signal();
    }


    void wait() {
        finish_signal.wait();
    }

    float cost;

    training_weights backprop_gradient;
private:
    training_network net;

    std::vector<training_weights*> grads_to_add;

    float learning_rate;
    float beta1;
    float beta2;

    int batch_size;
    float min_lambda;
    float max_lambda;

    int step;

    int thread_id;
    int pool_size;

    std::thread t;
    training_position *batch;

    std::atomic<bool> running;

    semaphore begin_signal;
    semaphore finish_signal;

    std::shared_ptr<training_weights> gradient;
    std::shared_ptr<training_weights> gradient_sq;

    std::shared_ptr<training_weights> first_moment;
    std::shared_ptr<training_weights> second_moment;

    std::shared_ptr<training_weights> corrected_first_moment;
    std::shared_ptr<training_weights> corrected_second_moment;

    void thread_wait()
    {
        begin_signal.wait();
    }

    void thread_signal_ready()
    {
        finish_signal.signal();
    }

    worker_operation operation;
};


struct worker_thread_pool
{
    worker_thread_pool(int num_of_workers,  std::shared_ptr<training_weights> weights,
                                            std::shared_ptr<training_weights> gradient,
                                            std::shared_ptr<training_weights> gradient_sq,
                                            std::shared_ptr<training_weights> first_moment,
                                            std::shared_ptr<training_weights> second_moment,
                                            std::shared_ptr<training_weights> corrected_first_moment,
                                            std::shared_ptr<training_weights> corrected_second_moment) {
        for (int i = 0; i < num_of_workers; i++) {
            threads.push_back(new worker_thread(weights, gradient, gradient_sq, first_moment, second_moment, corrected_first_moment, corrected_second_moment, i, num_of_workers));
        }
        steps = 0;
    }

    ~worker_thread_pool()
    {
        for (size_t i = 0; i < threads.size(); i++) {
            delete threads[i];
        }
        threads.clear();
    }


    void step(training_position *batch, int batch_size, float min_lambda, float max_lambda, bool use_factorizer, float beta1, float beta2, float learning_rate, float &avg_cost)
    {
        steps++;

        std::for_each(threads.begin(), threads.end(), [batch, batch_size, use_factorizer, min_lambda, max_lambda] (auto p) {p->start_backprop(batch, batch_size, min_lambda, max_lambda, use_factorizer);});
        std::vector<training_weights*> grads_to_add;
        avg_cost = 0;
        for (size_t i = 0; i < threads.size(); i++) {
            threads[i]->wait();

            grads_to_add.push_back(&threads[i]->backprop_gradient);

            avg_cost += threads[i]->cost;
        }
        avg_cost /= batch_size;

        std::for_each(threads.begin(), threads.end(), [grads_to_add, beta1, beta2, this] (auto p) {p->start_grad_calc(grads_to_add, beta1, beta2, steps);});
        std::for_each(threads.begin(), threads.end(), [] (auto p) {p->wait();});

        std::for_each(threads.begin(), threads.end(), [learning_rate] (auto p) {p->start_rmsprop(learning_rate);});
        std::for_each(threads.begin(), threads.end(), [] (auto p) {p->wait();});
    }

    int get_pool_size() {
        return threads.size();
    }
private:
    std::vector<worker_thread*> threads;
    int steps;
};

void visualize_net(std::string output_folder, training_weights &weights)
{
    int neurons_per_line = 32;

    int neuron_width = 2*8;
    int neuron_height = 6*8;

    int image_width = neurons_per_line * neuron_width;
    int image_height = (num_perspective_neurons / neurons_per_line) * neuron_height;

    for (int bucket = 0; bucket < num_of_king_buckets+1; bucket++) {

        uint8_t *image = new uint8_t[image_width*image_height*3];

        for (int neuron = 0; neuron < num_perspective_neurons; neuron++) {


            float std_mean = 0;
            float std_dev = 0;
            for (int piece = 0; piece < 12; piece++) {
                for (int square = 0; square < 64; square++) {
                    int input = bucket*12*64 + square*12 + piece;

                    float f_w = 0.0f;
                    if (bucket < num_of_king_buckets) {
                        int f_input = num_of_king_buckets*12*64 + square*12 + piece;
                        f_w = weights.perspective_weights.weights[f_input*num_perspective_neurons + neuron];
                    }

                    float w = weights.perspective_weights.weights[input*num_perspective_neurons + neuron] + f_w;

                    std_mean += w;
                }
            }
            std_mean /= 12*64;
            for (int piece = 0; piece < 12; piece++) {
                for (int square = 0; square < 64; square++) {
                    int input = bucket*12*64 + square*12 + piece;

                    float f_w = 0.0f;
                    if (bucket < num_of_king_buckets) {
                        int f_input = num_of_king_buckets*12*64 + square*12 + piece;
                        f_w = weights.perspective_weights.weights[f_input*num_perspective_neurons + neuron];
                    }

                    float w = weights.perspective_weights.weights[input*num_perspective_neurons + neuron] + f_w;

                    std_dev += w*w;
                }
            }
            std_dev = sqrt(std_dev / (12*64));

            for (int piece = 0; piece < 12; piece++) {
                for (int square = 0; square < 64; square++) {
                    int input = bucket*12*64 + square*12 + piece;

                    float f_w = 0.0f;
                    if (bucket < num_of_king_buckets) {
                        int f_input = num_of_king_buckets*12*64 + square*12 + piece;
                        f_w = weights.perspective_weights.weights[f_input*num_perspective_neurons + neuron];
                    }

                    float w = weights.perspective_weights.weights[input*num_perspective_neurons + neuron] + f_w;

                    float nw = (w - std_mean) / std_dev;

                    int sq_x = square % 8;
                    int sq_y = square / 8;

                    int black = piece % 2;
                    int piece_type = piece / 2;

                    int piece_y = piece_type*8;
                    int piece_x = black*8;

                    int neuron_x = (neuron % neurons_per_line)*neuron_width;
                    int neuron_y = (neuron / neurons_per_line)*neuron_height;

                    int x = neuron_x + piece_x + sq_x;
                    int y = neuron_y + piece_y + sq_y;

                    uint8_t lum = static_cast<uint8_t>(std::clamp((nw*0.5f+0.5f)*255.0f, 0.0f, 255.0f));

                    image[(y*image_width + x)*3+0] = lum;
                    image[(y*image_width + x)*3+1] = lum;
                    image[(y*image_width + x)*3+2] = lum;
                }
            }
        }



        std::stringstream ss;
        ss << output_folder;
        ss << "/net_bucket_";
        if (bucket == num_of_king_buckets) {
            ss << "factorizer";
        } else {
            ss << bucket;
        }
        ss << ".bmp";
        bmp_image_utility::save_pixels(ss.str(), image, image_width, image_height, 24);

        delete [] image;
    }
}



void nnue_trainer::test_nets(std::string training_net_file, std::string quantized_net_file, std::vector<selfplay_result> &test_set)
{
    std::cout << "Testing networks." << std::endl;

    std::cout << "Loading training nnue..." << std::endl;

    std::shared_ptr<training_weights> training_nnue_weights = std::make_shared<training_weights>();
    training_nnue_weights->load_file(training_net_file);
    training_network training_nnue(training_nnue_weights);

    visualize_net("vis", *training_nnue_weights);

    training_nnue_weights->save_quantized(quantized_net_file);

    std::cout << "Loading quantized nnue..." << std::endl;

    std::shared_ptr<nnue_weights> quantized_nnue_weights = std::make_shared<nnue_weights>();
    quantized_nnue_weights->load(quantized_net_file);
    nnue_network quantized_nnue(quantized_nnue_weights);


    float avg_error = 0.0f;

    float nnue_mse = 0.0f;
    float qnnue_mse = 0.0f;
    float quantization_mse = 0.0f;
    float worst_quantization_error = 0.0f;

    float black_nnue_mse = 0.0f;
    float white_nnue_mse = 0.0f;

    int black_positions = 0;
    int white_positions = 0;

    board_state state;

    int positions = 0;

    std::cout << "Evaluating..." << std::endl;
    for (size_t i = 0; i < test_set.size(); i++) {
        state.load_fen(test_set[i].fen);

        if (state.in_check(state.get_turn()) || state.get_square(test_set[i].bm.to).get_type() != EMPTY) {
            continue;
        }
        positions++;

        float real = test_set[i].eval / 100.0f;
        float qeval = (float)quantized_nnue.evaluate(state) / 100.0f;
        float eval = training_nnue.evaluate(state);

        nnue_mse += (real - eval)*(real - eval);
        qnnue_mse += (real - qeval)*(real - qeval);
        avg_error += std::abs(real - eval);

        if (state.get_turn() == WHITE) {
            white_nnue_mse += (real - eval)*(real - eval);
            white_positions += 1;
        } else {
            black_nnue_mse += (real - eval)*(real - eval);
            black_positions += 1;
        }

        worst_quantization_error = std::max(worst_quantization_error, std::abs(eval - qeval));
        quantization_mse += (eval - qeval)*(eval - qeval);
    }



    nnue_mse = nnue_mse / positions;
    qnnue_mse = qnnue_mse / positions;
    quantization_mse = quantization_mse / positions;
    avg_error = avg_error / positions;

    black_nnue_mse = black_nnue_mse / black_positions;
    white_nnue_mse = white_nnue_mse / white_positions;

    float nnue_rmse = sqrt(nnue_mse);
    float qnnue_rmse = sqrt(qnnue_mse);

    float quantization_rmse = sqrt(quantization_mse);
    float black_rmse = sqrt(black_nnue_mse);
    float white_rmse = sqrt(white_nnue_mse);



    std::cout << "NNUE avg error: " << avg_error << std::endl;
    std::cout << "NNUE rmse: " << nnue_rmse << "  Quantized NNUE rmse: " << qnnue_rmse << std::endl;
    std::cout << "Quantization rmse: " << quantization_rmse << "   Worst quantization error: " << worst_quantization_error << std::endl;
    std::cout << "White NNUE rmse: " << white_rmse << " Black NNUE rmse: " << black_rmse << std::endl;
}



void nnue_trainer::quantize_net(std::string net_file, std::string qnet_file)
{
    std::shared_ptr<training_weights> weights = std::make_shared<training_weights>();
    weights->load_file(net_file);
    weights->save_quantized(qnet_file);
}




void training_loop(std::string net_file, std::string qnet_file, std::shared_ptr<training_weights> weights, std::shared_ptr<data_reader> dataset)

{
    std::srand(time(NULL));


    std::shared_ptr<training_weights> gradient = std::make_shared<training_weights>();
    std::shared_ptr<training_weights> gradient_sq = std::make_shared<training_weights>();

    std::shared_ptr<training_weights> first_moment = std::make_shared<training_weights>();
    std::shared_ptr<training_weights> second_moment = std::make_shared<training_weights>();

    std::shared_ptr<training_weights> corrected_first_moment = std::make_shared<training_weights>();
    std::shared_ptr<training_weights> corrected_second_moment  = std::make_shared<training_weights>();


    first_moment->zero();
    second_moment->zero();

    worker_thread_pool thread_pool(16, weights, gradient, gradient_sq, first_moment, second_moment, corrected_first_moment, corrected_second_moment);

    bool use_factorized = true;
    float learning_rate = 0.0001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    int batch_size = 4000*thread_pool.get_pool_size();
    int epoch_size = 100000000;
    float min_lambda = 0.2f;
    float max_lambda = 0.5f;

    float learning_rate_decay = 0.98f;

    training_batch_manager batch_manager(batch_size, epoch_size, dataset);

    auto t0 = std::chrono::high_resolution_clock::now();

    float batch_cost;
    float training_cost = 0.0f;

    int epoch = 0;

    std::cout << std::endl;
    std::cout << "Dataset size: " << dataset->get_size<training_position>() / (1000*1000) << "M" << std::endl;
    std::cout << "Beta1: " << beta1 << std::endl;
    std::cout << "Beta2: " << beta2 << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << "Learning rate decay: " << learning_rate_decay << std::endl;
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


        if (batch_manager.get_epochs() != epoch) {
            epoch = batch_manager.get_epochs();

            weights->save_file(net_file);
            weights->save_quantized(qnet_file);

            visualize_net("vis", *weights);

            learning_rate *= learning_rate_decay;

            std::cout << "Net saved!" << std::endl;
        }


        thread_pool.step(batch_manager.get_current_batch(), batch_size, min_lambda, max_lambda, use_factorized, beta1, beta2, learning_rate, batch_cost);

        if (training_cost == 0.0f) {
            training_cost = batch_cost;
        } else {
            training_cost = training_cost * 0.99f + batch_cost * 0.01f;
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

    init_weights(*weights, true);
    weights->load_file(net_file);

    visualize_net("vis", *weights);

    training_loop(net_file, qnet_file, weights, reader);
}


float nnue_trainer::find_scaling_factor_for_net(std::string qnet_file, std::vector<selfplay_result> &positions)
{
    std::shared_ptr<nnue_weights> weights = std::make_shared<nnue_weights>();
    weights->load(qnet_file);
    nnue_network net(weights);

    std::vector<training_position> data;
    data.reserve(positions.size());

    std::cout << "Evaluating positions... ";

    board_state state;

    for (size_t i = 0; i < positions.size(); i++) {
        state.load_fen(positions[i].fen);
        data.emplace_back(positions[i]);
        data.back().eval = net.evaluate(state);
    }
    std::cout << "done." << std::endl;

    float scaling_factor = training_data_utility::find_scaling_factor_for_data(data);

    std::cout << "Scaling factor: " << scaling_factor << std::endl;

    return scaling_factor;
}



void nnue_trainer::optimize_quantized_net(std::string input_file, std::string output_file, std::vector<selfplay_result> &test_set)
{
    //Sorts neurons based on probability of activation

    std::shared_ptr<nnue_weights> weights = std::make_shared<nnue_weights>();
    std::shared_ptr<nnue_weights> optimized_weights = std::make_shared<nnue_weights>();
    weights->load(input_file);
    optimized_weights->load(input_file);

    nnue_network nnue(weights);

    constexpr int N = num_perspective_neurons;

    std::vector<std::pair<int, int>> act;
    for (int j = 0; j < N; j++) {
        act.emplace_back(j, 0);
    }

    board_state state;

    std::cout << "Evaluating..." << std::endl;
    for (size_t i = 0; i < test_set.size(); i++) {
        state.load_fen(test_set[i].fen);
        nnue.evaluate(state);

        int16_t *white_activations = nnue.get_perspective_activations(WHITE);
        int16_t *black_activations = nnue.get_perspective_activations(BLACK);
        for (int j = 0; j < N; j++) {
            act[j].second += (white_activations[j] > 0);
            act[j].second += (black_activations[j] > 0);
        }
    }


    std::cout << "Activations" << std::endl;

    for (int j = 0; j < N; j++) {
        std::cout << j << ": " << std::setprecision(3) << ((double)act[j].second*50 / test_set.size()) << "%"  << std::endl;
    }


    std::cout << "Sorting neurons..." << std::endl;

    std::sort(act.begin(), act.end(), [] (auto &a, auto &b) {return a.second > b.second;});

    int16_t *src_persp_weights = weights->perspective_weights.weights;
    int16_t *src_persp_biases = weights->perspective_weights.biases;

    int16_t *dst_persp_weights = optimized_weights->perspective_weights.weights;
    int16_t *dst_persp_biases = optimized_weights->perspective_weights.biases;

    for (int i = 0; i < N; i++) {
        int src_index = act[i].first;
        int dst_index = i;

        for (int j = 0; j < num_perspective_inputs; j++) {
            dst_persp_weights[j * N + dst_index] = src_persp_weights[j * N + src_index];
        }
        dst_persp_biases[dst_index] = src_persp_biases[src_index];
    }


    for (int b = 0; b < layer_stack_size; b++) {
        int16_t *src_layer_weights = &weights->layer1_weights.weights[b*2*N*layer1_neurons];
        int16_t *dst_layer_weights = &optimized_weights->layer1_weights.weights[b*2*N*layer1_neurons];

        for (int i = 0; i < layer1_neurons; i++) {
            for (int j = 0; j < N; j++) {
                int src_index0 = act[j].first;
                int dst_index0 = j;

                int src_index1 = act[j].first + N;
                int dst_index1 = j + N;

                dst_layer_weights[i*2*N + dst_index0] = src_layer_weights[i*2*N + src_index0];
                dst_layer_weights[i*2*N + dst_index1] = src_layer_weights[i*2*N + src_index1];
            }
        }

        int16_t *dst_layer_transposed_weights = &optimized_weights->layer1_weights.transposed_weights[b*2*N*layer1_neurons];
        for (int i = 0; i < layer1_neurons; i++) {
            for (int j = 0; j < 2*N; j++) {
                dst_layer_transposed_weights[j*layer1_neurons + i] = dst_layer_weights[i*2*N + j];
            }
        }

    }


    nnue_network optimized_nnue(optimized_weights);

    std::cout << "Validating..." << std::endl;

    std::for_each(act.begin(), act.end(), [] (auto &x) {x.second = 0;});

    for (size_t i = 0; i < test_set.size(); i++) {
        state.load_fen(test_set[i].fen);
        if (nnue.evaluate(state) != optimized_nnue.evaluate(state)) {
            std::cout << "Validation failed!!!" << std::endl;
            return;
        }

        int16_t *white_activations = optimized_nnue.get_perspective_activations(WHITE);
        int16_t *black_activations = optimized_nnue.get_perspective_activations(BLACK);
        for (int j = 0; j < N; j++) {
            act[j].second += (white_activations[j] > 0);
            act[j].second += (black_activations[j] > 0);
        }
    }

    std::cout << "Activations after optimization" << std::endl;
    for (int j = 0; j < N; j++) {
        std::cout << j << ": " << std::setprecision(3) << ((double)act[j].second*50 / test_set.size()) << "%"  << std::endl;
    }

    std::cout << "Saving optimized NNUE" << std::endl;
    optimized_weights->save(output_file);

}













