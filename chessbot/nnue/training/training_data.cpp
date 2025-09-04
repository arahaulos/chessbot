#include "training_data.hpp"
#include <math.h>
#include <algorithm>
#include <random>
#include <sstream>
#include <filesystem>


float evaluation_wdl_mse(const std::vector<training_position> &data, float scaling_factor)
{
    float mse = 0.0f;

    for (size_t i = 0; i < data.size(); i++) {
        float result = data[i].get_wdl_relative_to_stm();
        float eval = 1.0f/(1.0f+(std::pow(10, -((float)data[i].eval / scaling_factor))));
        //float eval = sigmoid((float)data[i].eval / scaling_factor);

        mse += (eval - result) * (eval - result);

    }
    mse /= data.size();
    return mse;
}



float find_scaling_factor_for_data(const std::vector<training_position> &data)
{
    constexpr float min_scaling = 200.0f;
    constexpr float max_scaling = 800.0f;
    constexpr int steps = 200;

    float best_scaling_factor = 0.0f;
    float min_mse = 10.0f;

    for (int i = 0; i < steps; i++) {
        float scaling_factor = min_scaling + (max_scaling - min_scaling)*i / steps;
        float mse = evaluation_wdl_mse(data, scaling_factor);

        if (mse < min_mse) {
            best_scaling_factor = scaling_factor;
            min_mse = mse;
        }
        std::cout << "\rFinding scaling factor " << (i*100)/steps << "%      ";
    }

    std::cout << std::endl << "Scaling factor: " << best_scaling_factor << "   MSE: " << min_mse << std::endl;

    return best_scaling_factor;
}




void filter_and_convert_data(std::vector<selfplay_result> &data_in, std::vector<training_position> &data_out)
{
    //This function filters selfplay results

    std::vector<training_position> data;

    data.reserve(data_in.size());

    bool is_tactical0 = false;
    /*bool is_tactical1 = false;
    bool is_tactical2 = false;*/

    //training_position pos_to_add = data_in[0];

    board_state state;
    for (size_t i = 0; i < data_in.size(); i++) {
        state.load_fen(data_in[i].fen);

        //float result = (state.get_turn() == WHITE ? data_in[i].wdl : 1.0f-data_in[i].wdl);

        is_tactical0 = (state.get_square(data_in[i].bm.to).get_type() != EMPTY) || state.in_check(WHITE) || state.in_check(BLACK);

        //bool wdl_eval_conflict = (result > 0.5f && data_in[i].eval < -300) || (result < 0.5f && data_in[i].eval > 300);

        if (!is_tactical0/* && !is_tactical1 && !is_tactical2*/) {
            //data.push_back(pos_to_add);
            data.emplace_back(state, data_in[i]);
        }

        /*pos_to_add = training_position(state, data_in[i]);
        is_tactical2 = is_tactical1;
        is_tactical1 = is_tactical0;*/

        if ((i % 100000) == 0) {
            std::cout << "\rFiltering and converting data: " << (i*100)/data_in.size() << "%   ";
        }
    }
    std::cout << std::endl;

    std::vector<training_position> batch = get_random_batch(data, 20000);
    float scaling_factor = find_scaling_factor_for_data(batch);


    data_out.reserve(data_out.size() + data.size());

    for (size_t i = 0; i < data.size(); i++) {
        data[i].eval = (data[i].eval * 400) / scaling_factor;
        data_out.push_back(data[i]);
    }
}


void save_binary_data(std::vector<training_position> &dataset, std::string output_folder, size_t file_number, size_t save_positions)
{
    std::stringstream ss;
    ss << output_folder;
    ss << "\\";
    ss << "data" << file_number << ".bin";

    std::cout << "Saving data: " << ss.str() << "... ";
    std::ofstream file(ss.str(), std::ios::binary);
    if (file.is_open()) {
        for (size_t i = 0; i < save_positions; i++) {
            training_position pos = dataset[dataset.size()-1];
            dataset.pop_back();
            file.write((char*)&pos, sizeof(training_position));
        }
        std::cout << " done." << std::endl;
    } else {
        std::cout << " error." << std::endl;
    }
    file.close();

}



void training_data_utility::convert_training_data(std::vector<std::string> selfplay_directories, std::string output_folder, size_t output_file_sizes_MB)
{
    std::vector <training_position> positions;

    size_t positions_per_file = (output_file_sizes_MB*1024*1024) / sizeof(training_position);
    size_t file_count = 0;
    size_t raw_data_size = 0;
    size_t filtered_data_size = 0;
    int input_files = 0;
    int output_files = 0;

    for (size_t i = 0; i < selfplay_directories.size(); i++) {
        std::string folder = selfplay_directories[i];
        for (const auto& entry : std::filesystem::directory_iterator(folder)) {
            std::string filename = entry.path().string();
            std::string extension = entry.path().extension().string();

            if (extension == ".txt") {
                std::vector<selfplay_result> sprdata;
                load_selfplay_results(sprdata, filename);

                filter_and_convert_data(sprdata, positions);

                while (positions.size() >= positions_per_file) {
                    save_binary_data(positions, output_folder, file_count++, positions_per_file);
                    filtered_data_size += positions_per_file;
                    output_files++;
                }
                raw_data_size += sprdata.size();
                input_files++;
            }
        }
    }
    filtered_data_size += positions.size();

    while (positions.size() > 0) {
        save_binary_data(positions, output_folder, file_count++, std::min(positions.size(), positions_per_file));
        output_files++;
    }

    std::cout << std::endl << std::endl;
    std::cout << "Input files: " << input_files << std::endl;
    std::cout << "Output files: " << output_files << std::endl;
    std::cout << "Raw dataset size: " << (raw_data_size / 1000000) << "M" << std::endl;
    std::cout << "Filtered dataset size: " << (filtered_data_size / 1000000) << "M" << std::endl;
    std::cout << std::endl;
}



data_reader::data_reader(std::vector<std::string> folders)
{
    dataset_size = 0;

    auto enumerate_files = [&folders] (auto action) {
        for (size_t i = 0; i < folders.size(); i++) {
            std::string folder = folders[i];
            for (const auto& entry : std::filesystem::directory_iterator(folder)) {
                std::string filename = entry.path().string();
                std::string extension = entry.path().extension().string();

                if (extension == ".bin") {
                    action(filename);
                }
            }
        }
    };

    enumerate_files([this] (std::string filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        size_t size_of_file = file.tellg();
        files.emplace_back(filename, size_of_file);
        dataset_size += size_of_file;

        std::cout << "File: " << filename << "  Size: " << size_of_file / (1024*1024) << "MB" << std::endl;
    }
    file.close();
    });
    std::cout << "Total dataset size: " << dataset_size / (1024*1024) << "MB" << std::endl;
}

void data_reader::read_raw(size_t position, size_t bytes_to_read, char *dst)
{
    size_t index = 0;
    for (size_t i = 0; i < files.size() && bytes_to_read > 0; i++) {
        std::string filename = files[i].first;
        size_t file_size = files[i].second;

        if (index + file_size > position) {
            size_t offset = position - index;
            size_t read_size = std::min(bytes_to_read, file_size - offset);

            std::ifstream file(filename, std::ios::binary);
            if (file.is_open()) {
                file.seekg(offset, std::ios::beg);

                size_t chunk_size = 1024*1024*64;
                size_t dst_index = 0;
                size_t read_left = read_size;
                while (read_left > 0) {
                    size_t s = std::min(read_left, chunk_size);
                    file.read(&dst[dst_index], s);
                    dst_index += s;
                    read_left -= s;
                }
            } else {
                std::cout << "Cannot read file " << filename << std::endl;
            }

            file.close();
            position += read_size;
            bytes_to_read -= read_size;
            dst += read_size;
        }
        index += file_size;
    }
}




training_batch_manager::training_batch_manager(size_t bs, size_t es, std::shared_ptr<data_reader> dr)
{
    epoch_size = es;
    batch_size = bs;

    reader = dr;
    buffer_index = 0;

    buffers[0] = new training_position[epoch_size];
    buffers[1] = new training_position[epoch_size];

    epochs = 0;
    batch_number = 0;

    batch_index = 0;

    load_epoch(buffers[0]);
    loader_thread = std::thread(&training_batch_manager::load_epoch, this, buffers[1]);
}


training_batch_manager::~training_batch_manager()
{
    delete [] buffers[0];
    delete [] buffers[1];
}

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


    int white_bucket = get_king_bucket(white_king_sq);
    int black_bucket = get_king_bucket(black_king_sq);

    return std::min(white_bucket,black_bucket)*16 + std::max(white_bucket, black_bucket);
}


void training_batch_manager::load_epoch(training_position *buffer)
{
    std::random_device rd;
    std::mt19937_64 gen(rd());

    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    std::vector<size_t> indices(epoch_size);
    for (int i = 0; i < epoch_size; i++) {
        indices[i] = dist(gen) % reader->get_size<training_position>();
    }

    std::sort(indices.begin(), indices.end());

    size_t fetch_index = 0;

    size_t chunk_size = 1024*1024*64;
    std::vector<training_position> chunk(chunk_size);

    size_t read_index = 0;
    while (read_index < reader->get_size<training_position>()) {
        if (read_index + chunk_size > reader->get_size<training_position>()) {
            chunk_size = reader->get_size<training_position>() - read_index;
        }

        reader->read<training_position>(read_index, chunk_size, chunk.data());

        while (fetch_index < epoch_size && indices[fetch_index] < read_index + chunk_size) {
            buffer[fetch_index] = chunk[indices[fetch_index] - read_index];
            fetch_index++;
        }

        read_index += chunk_size;
    }


    /*size_t chunk_size = 1000;
    size_t chunks = epoch_size / chunk_size;

    for (int i = 0; i < chunks; i++) {
        uint64_t pos = dist(gen) % (reader->get_size<training_position>() - chunk_size);

        reader->read<training_position>(pos, chunk_size, &buffer[i*chunk_size]);
    }*/

    for (size_t i = epoch_size - 1; i > 0; --i) {
        size_t j = dist(gen) % (i+1);

        std::swap(buffer[i], buffer[j]);
    }


    std::vector<std::pair<training_position, int>> batch(batch_size);

    for (size_t i = 0; i < epoch_size - batch_size; i += batch_size) {
        for (size_t j = 0; j < batch_size; j++) {
            batch[j].first = buffer[i + j];
            batch[j].second = get_king_bucket_configuration(buffer[i + j]);
        }
        std::sort(batch.begin(), batch.end(), [] (auto &a, auto &b) {return a.second > b.second;});
        for (size_t j = 0; j < batch_size; j++) {
            buffer[i + j] = batch[j].first;
        }
    }
}


void training_batch_manager::load_new_batch()
{
    batch_number += 1;
    batch_index += batch_size;

    if (batch_index + batch_size >= epoch_size) {
        loader_thread.join();
        loader_thread = std::thread(&training_batch_manager::load_epoch, this, buffers[buffer_index]);

        buffer_index = (buffer_index + 1) % 2;
        batch_index = 0;
        batch_number = 0;
        epochs += 1;
    }
}








