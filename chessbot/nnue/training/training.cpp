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


struct trainer_params
{
    bool use_factorized;
    float learning_rate;
    float weight_decay;
    float beta1;
    float beta2;
    int batch_size;
    int epoch_size;
    float min_lambda;
    float max_lambda;

    bool freeze_perspective;
    bool freeze_l1_weights;
    bool enable_position_skipping;

    float learning_rate_decay;
};


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

                weights.perspective_weights.weights[input*(num_perspective_neurons + num_perspective_psqt) + j] = 0.01f;
            }


            for (size_t j = num_perspective_neurons; j < num_perspective_neurons+num_perspective_psqt; j++) {
                size_t input = num_of_king_buckets*inputs_per_bucket + i;

                int piece_type = (i / 2) % 6;
                int piece_color = i % 2;

                static float piece_values[6] = {1.0, 3.2, 3.3, 5.5, 9.5, 0};

                float val = 0;
                if (piece_color == 0) {
                    val = piece_values[piece_type]*1.8f;
                } else {
                    val = -piece_values[piece_type]*1.8f;
                }

                weights.perspective_weights.weights[input*(num_perspective_neurons + num_perspective_psqt) + j] = val;
            }
        }
    }
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

bool skip_position(training_position &pos)
{
    constexpr int max_eval_error = 200;

    int32_t eval = pos.eval;
    float wdl = pos.get_wdl_relative_to_stm();

    return ((eval > max_eval_error && wdl < 0.25f) ||
            (eval < -max_eval_error && wdl > 0.75f));
}


enum worker_operation {WORK_BACKPROP, WORK_GRAD_CALC, WORK_RMSPROP};
struct optimizer_worker
{
    optimizer_worker(std::shared_ptr<training_weights> weights,
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
        non_skipped_positions = 0;

        t = std::thread(&optimizer_worker::thread_entry, this);
    }

    ~optimizer_worker()
    {
        running = false;
        t.join();
    }

    void thread_entry() {
        while (running) {
            thread_wait();

            if (operation == WORK_BACKPROP) {
                do_backprop();
            } else if (operation == WORK_GRAD_CALC) {
                do_grad_calc();
            } else if (operation == WORK_RMSPROP) {
                do_rmsprop();
            }

            thread_signal_ready();
        }
    }

    void set_params(trainer_params &p)
    {
        params = p;
    }

    void start_backprop(training_position *data)
    {
        operation = WORK_BACKPROP;

        batch = data;
        net.use_factorizer = params.use_factorized;

        begin_signal.signal();
    }

    void start_grad_calc(std::vector<training_weights*> grads, int t)
    {
        operation = WORK_GRAD_CALC;
        grads_to_add = grads;
        step = t;
        begin_signal.signal();
    }

    void start_rmsprop()
    {
        operation = WORK_RMSPROP;
        begin_signal.signal();
    }


    void wait() {
        finish_signal.wait();
    }

    float psqt_portion;
    float ff_sparsity;
    float cost;
    size_t non_skipped_positions;

    training_weights backprop_gradient;
private:
    training_network net;

    std::vector<training_weights*> grads_to_add;

    trainer_params params;

    std::thread t;
    training_position *batch;
    int step;
    int thread_id;
    int pool_size;

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


    void do_backprop()
    {

        backprop_gradient.zero();

        cost = 0;
        non_skipped_positions = 0;

        psqt_portion = 0.0f;
        ff_sparsity = 0.0f;

        int per_thread = params.batch_size / pool_size;
        int start = per_thread * thread_id;

        int sparsity_sample_count = 0;

        for (size_t i = start; i < start + per_thread; i++) {
            training_position sample = batch[i];

            if (params.enable_position_skipping && skip_position(sample)) {
                continue;
            }

            non_skipped_positions += 1;

            float num_of_pieces = pop_count(sample.occupation);

            float lambda_weight = std::clamp((num_of_pieces - 4.0f)/28.0f, 0.0f, 1.0f);
            float lambda = lambda_weight*params.min_lambda + (1.0f-lambda_weight)*params.max_lambda;

            float result = sample.get_wdl_relative_to_stm();

            float pred_p = net.evaluate(sample);
            float eval_p = static_cast<float>(sample.eval) / 100.0f;

            float pred = sigmoid(pred_p / 4.0f);
            float eval = sigmoid(eval_p / 4.0f);

            if (std::isnan(pred)    || std::isinf(pred) ||
                std::isnan(eval)    || std::isinf(eval) ||
                std::isnan(result)  || std::isinf(result))
            {
                std::cout << "Warning: bad prediction or label: ";
                std::cout << "Label (result, eval): (" << result << ", " << eval << ")  Prediction: " << pred << std::endl;
                continue;
            }

            psqt_portion += std::abs(net.last_psqt_eval) / (std::abs(net.last_psqt_eval) + std::abs(net.last_pos_eval));
            if (i % 16 == 0) {
                ff_sparsity += net.white_side.output_sparsity() + net.black_side.output_sparsity();
                sparsity_sample_count += 2;
            }

            float loss_eval, loss_result, loss_eval_delta, loss_result_delta;

            loss(pred, eval, loss_eval, loss_eval_delta);
            loss(pred, result, loss_result, loss_result_delta);

            float loss       = loss_eval       * (1.0f-lambda) + loss_result       * lambda;
            float loss_delta = loss_eval_delta * (1.0f-lambda) + loss_result_delta * lambda;
            float sigmoid_delta = pred * (1.0f - pred);

            if (std::isnan(loss)          || std::isinf(loss) ||
                std::isnan(loss_delta)    || std::isinf(loss_delta) ||
                std::isnan(sigmoid_delta) || std::isinf(sigmoid_delta))
            {
                std::cout << "Warning: bad loss or derivative: ";
                std::cout << loss << " " << loss_delta << " " << sigmoid_delta << std::endl;
                continue;
            }

            net.back_propagate(backprop_gradient, loss_delta*sigmoid_delta, sample.get_turn(), params.freeze_perspective);

            cost += loss;
        }

        psqt_portion /= non_skipped_positions;
        ff_sparsity /= sparsity_sample_count;

        backprop_gradient.divide(non_skipped_positions*pool_size);
    }

    void do_grad_calc()
    {
        gradient->zero(thread_id, pool_size);
        for (int i = 0; i < grads_to_add.size(); i++) {
            gradient->add(*grads_to_add[i], thread_id, pool_size);
        }

        gradient_sq->squared(*gradient, thread_id, pool_size);

        first_moment->exponential_smoothing(*gradient, params.beta1, thread_id, pool_size);
        second_moment->exponential_smoothing(*gradient_sq, params.beta2, thread_id, pool_size);

        corrected_first_moment->mult(*first_moment, 1.0f / (1.0f - std::pow(params.beta1, step)), thread_id, pool_size);
        corrected_second_moment->mult(*second_moment, 1.0f / (1.0f - std::pow(params.beta2, step)), thread_id, pool_size);
    }

    void do_rmsprop()
    {
        net.weights->output_weights.rmsprop(&corrected_first_moment->output_weights, &corrected_second_moment->output_weights, params.learning_rate, params.weight_decay, thread_id, pool_size);

        net.weights->layer2_weights.rmsprop(&corrected_first_moment->layer2_weights, &corrected_second_moment->layer2_weights, params.learning_rate, params.weight_decay, thread_id, pool_size);

        if (!params.freeze_l1_weights) {
            net.weights->layer1_weights.rmsprop(&corrected_first_moment->layer1_weights, &corrected_second_moment->layer1_weights, params.learning_rate, params.weight_decay, thread_id, pool_size);
        }

        if (!params.freeze_perspective) {
            net.weights->perspective_weights.rmsprop(&corrected_first_moment->perspective_weights, &corrected_second_moment->perspective_weights, params.learning_rate, params.weight_decay, thread_id, pool_size);
        }
    }



    worker_operation operation;
};


struct optimizer
{
    optimizer(int num_of_workers,   std::shared_ptr<training_weights> weights,
                                    std::shared_ptr<training_weights> gradient,
                                    std::shared_ptr<training_weights> gradient_sq,
                                    std::shared_ptr<training_weights> first_moment,
                                    std::shared_ptr<training_weights> second_moment,
                                    std::shared_ptr<training_weights> corrected_first_moment,
                                    std::shared_ptr<training_weights> corrected_second_moment) {
        for (int i = 0; i < num_of_workers; i++) {
            workers.push_back(new optimizer_worker(weights, gradient, gradient_sq, first_moment, second_moment, corrected_first_moment, corrected_second_moment, i, num_of_workers));
        }
        steps = 0;
    }

    ~optimizer()
    {
        for (size_t i = 0; i < workers.size(); i++) {
            delete workers[i];
        }
        workers.clear();
    }


    size_t step(training_position *batch, float &avg_cost, float &psqt_portion, float &ff_sparsity, size_t &non_skipped_positions, trainer_params &params)
    {
        non_skipped_positions = 0;
        avg_cost = 0;
        psqt_portion = 0;
        ff_sparsity = 0;
        steps++;

        std::for_each(workers.begin(), workers.end(), [&] (auto p) {p->set_params(params); });
        std::for_each(workers.begin(), workers.end(), [&] (auto p) {p->start_backprop(batch); });
        std::vector<training_weights*> grads_to_add;
        for (size_t i = 0; i < workers.size(); i++) {
            workers[i]->wait();

            grads_to_add.push_back(&workers[i]->backprop_gradient);

            avg_cost += workers[i]->cost;
            non_skipped_positions += workers[i]->non_skipped_positions;
            psqt_portion += workers[i]->psqt_portion / workers.size();
            ff_sparsity += workers[i]->ff_sparsity / workers.size();
        }
        avg_cost /= non_skipped_positions;

        std::for_each(workers.begin(), workers.end(), [&] (auto p) {p->start_grad_calc(grads_to_add, steps);});
        std::for_each(workers.begin(), workers.end(), [] (auto p) {p->wait();});
        std::for_each(workers.begin(), workers.end(), [] (auto p) {p->start_rmsprop();});
        std::for_each(workers.begin(), workers.end(), [] (auto p) {p->wait();});

        return non_skipped_positions;
    }

    int get_pool_size() {
        return workers.size();
    }
private:
    std::vector<optimizer_worker*> workers;
    int steps;
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
    uint64_t white_act = 0;
    uint64_t black_act = 0;


    float pos_quantization_mse = 0.0f;
    float psqt_quantization_mse = 0.0f;
    float worst_pos_quantization_error = 0.0f;
    float worst_psqt_quantization_error = 0.0f;

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


        float pos_eval = training_nnue.last_pos_eval;
        float psqt_eval = training_nnue.last_psqt_eval;
        float qpos_eval = (float)quantized_nnue.last_pos_eval / 100.0f;
        float qpsqt_eval = (float)quantized_nnue.last_psqt_eval / 100.0f;

        pos_quantization_mse += (pos_eval - qpos_eval)*(pos_eval - qpos_eval);
        psqt_quantization_mse += (psqt_eval - qpsqt_eval)*(psqt_eval - qpsqt_eval);

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

        white_act += quantized_nnue.get_perspective(WHITE).num_of_outputs;
        black_act += quantized_nnue.get_perspective(BLACK).num_of_outputs;

        worst_quantization_error = std::max(worst_quantization_error, std::abs(eval - qeval));
        quantization_mse += (eval - qeval)*(eval - qeval);

        worst_pos_quantization_error = std::max(worst_pos_quantization_error, std::abs(pos_eval - qpos_eval));
        worst_psqt_quantization_error = std::max(worst_psqt_quantization_error, std::abs(psqt_eval - qpsqt_eval));
    }



    nnue_mse = nnue_mse / positions;
    qnnue_mse = qnnue_mse / positions;
    quantization_mse = quantization_mse / positions;
    avg_error = avg_error / positions;
    pos_quantization_mse = pos_quantization_mse / positions;
    psqt_quantization_mse = psqt_quantization_mse / positions;

    black_nnue_mse = black_nnue_mse / black_positions;
    white_nnue_mse = white_nnue_mse / white_positions;

    float nnue_rmse = sqrt(nnue_mse);
    float qnnue_rmse = sqrt(qnnue_mse);

    float quantization_rmse = sqrt(quantization_mse);
    float black_rmse = sqrt(black_nnue_mse);
    float white_rmse = sqrt(white_nnue_mse);


    float pos_quant_rmse = sqrt(pos_quantization_mse);
    float psqt_quant_rmse = sqrt(psqt_quantization_mse);


    std::cout << "NNUE avg error: " << avg_error << std::endl;
    std::cout << "White NNUE rmse: " << white_rmse << " Black NNUE rmse: " << black_rmse << std::endl;
    std::cout << "NNUE rmse: " << nnue_rmse << "  Quantized NNUE rmse: " << qnnue_rmse << std::endl;
    std::cout << "Quantization rmse: " << quantization_rmse
              << "\nWorst quantization error: " << worst_quantization_error << std::endl;


    float avg_activations = (float)(black_act + white_act) / positions;
    std::cout << "Average feature transformer sparsity: " << avg_activations*100 / num_perspective_neurons
              << "% (" << avg_activations << "/" << num_perspective_neurons << ")" << std::endl;


    std::cout << std::endl << "Pos quant rmse: " << pos_quant_rmse << "  Psqt quant rmse: " << psqt_quant_rmse << std::endl;
    std::cout << "Worst pos quant error: " << worst_pos_quantization_error << "  Worst psqt quant error: " << worst_psqt_quantization_error << std::endl;
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

    optimizer opt(8, weights, gradient, gradient_sq, first_moment, second_moment, corrected_first_moment, corrected_second_moment);


    trainer_params params;

    params.use_factorized = true;
    params.learning_rate = 0.0001f;
    params.weight_decay = 0.01f;//0.0f;
    params.beta1 = 0.9f;
    params.beta2 = 0.999f;
    params.batch_size = 32000*opt.get_pool_size();
    params.epoch_size = 100000000;
    params.min_lambda = 0.2f;
    params.max_lambda = 0.4f;

    params.freeze_perspective = false;
    params.freeze_l1_weights = false;

    params.enable_position_skipping = false;

    params.learning_rate_decay = 0.98f;

    std::cout << std::endl;
    std::cout << "Dataset size: " << dataset->get_size<training_position>() / (1000*1000) << "M" << std::endl;
    std::cout << "Beta1: " << params.beta1 << std::endl;
    std::cout << "Beta2: " << params.beta2 << std::endl;
    std::cout << "Learning rate: " << params.learning_rate << std::endl;
    std::cout << "Learning rate decay: " << params.learning_rate_decay << std::endl;
    std::cout << "Weight decay: " << params.weight_decay << std::endl;
    std::cout << "Min lambda: " << params.min_lambda << std::endl;
    std::cout << "Max lambda: " << params.max_lambda << std::endl;
    std::cout << "Batch size: " << params.batch_size << std::endl;
    std::cout << "Epoch size: " << params.epoch_size << std::endl;
    std::cout << "Threads: " << opt.get_pool_size() << std::endl;
    std::cout << "Enable position skipping: " << params.enable_position_skipping << std::endl;
    std::cout << "Freeze perspective weights: " << params.freeze_perspective << std::endl;
    std::cout << "Freeze L1 weights: " << params.freeze_l1_weights << std::endl << std::endl;

    training_batch_manager batch_manager(params.batch_size, params.epoch_size, dataset);

    auto t0 = std::chrono::high_resolution_clock::now();

    float batch_cost;
    float training_cost = 0.0f;
    float psqt_portion, ff_sparsity;

    int epoch = 0;
    size_t non_skipped_positions;

    while (true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 );
        t0 = t1;

        batch_manager.load_new_batch();

        if (batch_manager.get_epochs() != epoch) {
            epoch = batch_manager.get_epochs();

            weights->save_file(net_file);
            weights->save_quantized(qnet_file);

            params.learning_rate *= params.learning_rate_decay;

            std::cout << "Net saved!" << std::endl;
        }

        opt.step(batch_manager.get_current_batch(), batch_cost, psqt_portion, ff_sparsity, non_skipped_positions, params);

        if (training_cost == 0.0f) {
            training_cost = batch_cost;
        } else {
            training_cost = training_cost * 0.99f + batch_cost * 0.01f;
        }

        float kspers = std::clamp((float)non_skipped_positions / ms.count(), 0.0f, 9999.0f);

        std::cout << "\rTrC: " << std::setprecision(6) << std::left << std::setw(12) << training_cost
                  << "   BC: "  << std::left << std::setw(12) << batch_cost
                  << "   Speed: " << std::right << std::setw(4) << (int)kspers << " KPos/s    "
                  << "   Epoch: " << batch_manager.get_epochs()
                  << " (" << std::setprecision(3) << std::setw(4) << std::right << (float)batch_manager.get_current_batch_number()*100.0f / batch_manager.get_number_of_batches() << "%)   "
                  << "PSQT: " << psqt_portion << "  FF sparsity: " << ff_sparsity << "   " << std::flush;
    }

}


void nnue_trainer::train(std::string net_file, std::string qnet_file, std::vector<std::string> selfplay_directories)
{
    std::shared_ptr<data_reader> reader = std::make_shared<data_reader>(selfplay_directories);

    std::shared_ptr<training_weights> weights = std::make_shared<training_weights>();

    init_weights(*weights, true);
    weights->load_file(net_file);

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










