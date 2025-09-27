#pragma once

#include <vector>
#include "training_position.hpp"
#include "../../tuning.hpp"


template <typename T>
std::vector<T> get_random_batch(std::vector<T> &samples, int batch_size)
{
    std::vector<T> batch;

    batch.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        int r = ((rand() & 0x7FFF) << 15) | (rand() & 0x7FFF);
        batch.push_back(samples[r % samples.size()]);
    }
    return batch;
}

struct data_reader
{
    data_reader(std::vector<std::string> directories);

    template <typename T>
    void read(size_t index, size_t num, T *buffer) {
        read_raw(index * sizeof(T), num * sizeof(T), (char*)buffer);
    }

    template <typename T>
    size_t get_size() const {
        return dataset_size / sizeof(T);
    }
private:
    std::vector<std::pair<std::string, size_t>> files;

    size_t dataset_size;

    void read_raw(size_t position, size_t bytes_to_read, char *dst);
};



struct training_batch_manager
{
    training_batch_manager(size_t bs, size_t es, std::shared_ptr<data_reader> dr);
    ~training_batch_manager();


    size_t get_number_of_batches() {
        return epoch_size / batch_size;
    }

    size_t get_current_batch_number() {
        return batch_number;
    }

    size_t get_epochs() {
        return epochs;
    }

    training_position *get_current_batch() {
        return &buffers[buffer_index][batch_index];
    }

    void load_new_batch();
private:

    void load_epoch(training_position *buffer);

    size_t batch_index;

    size_t epochs;
    size_t epoch_size;

    size_t batch_size;
    size_t batch_number;

    training_position *buffers[2];
    size_t buffer_index;

    std::shared_ptr<data_reader> reader;

    std::thread loader_thread;
};



struct training_data_utility
{
    static void convert_training_data(std::vector<std::string> selfplay_directories, std::string output_folder, size_t output_file_sizes_MB);

    static float find_scaling_factor_for_data(const std::vector<training_position> &data);
};
