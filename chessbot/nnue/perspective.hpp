#pragma once

#include "layer.hpp"

constexpr int acculumator_update_table_size = 64;


template <int INPUTS, int NEURONS>
struct nnue_perspective_weights
{
    nnue_perspective_weights() {
        weights_buffer = new int16_t[INPUTS*NEURONS+64];
        biases_buffer = new int16_t[INPUTS+64];

        biases = align_ptr(biases_buffer);
        weights = align_ptr(weights_buffer);
    }

    ~nnue_perspective_weights() {
        delete [] weights_buffer;
        delete [] biases_buffer;
    }

    void load(int16_t *data, size_t &index)
    {
        for (size_t i = 0; i < NEURONS*INPUTS; i++) {
            weights[i] = data[index++];
        }
        for (size_t i = 0; i < NEURONS; i++) {
            biases[i] = data[index++];
        }
    }

    int num_of_biases() const {
        return NEURONS;
    }

    int num_of_weights() const {
        return NEURONS*INPUTS;
    }

    int16_t *weights;
    int16_t *biases;

    int16_t *weights_buffer;
    int16_t *biases_buffer;
};

enum fused_update_type_t {FU_NONE, FU_ADDSUB, FU_ADDSUBSUB};

struct acculumator_update_table
{
    int16_t updates[acculumator_update_table_size];
    int16_t num_of_updates;
    bool is_clean;
    bool refresh;
    int16_t *acculumator;

    fused_update_type_t fused;

    void add(int index)
    {
        fused = FU_NONE;
        updates[num_of_updates++] = index+1;
    }

    void sub(int index)
    {
        fused = FU_NONE;
        updates[num_of_updates++] = -index-1;
    }

    void addsub(int add_index, int sub_index)
    {
        fused = (num_of_updates == 0 ? FU_ADDSUB : FU_NONE);
        updates[num_of_updates++] = add_index+1;
        updates[num_of_updates++] = -sub_index-1;
    }

    void addsubsub(int add_index, int sub_index0, int sub_index1)
    {
        fused = (num_of_updates == 0 ? FU_ADDSUBSUB : FU_NONE);
        updates[num_of_updates++] = add_index+1;
        updates[num_of_updates++] = -sub_index0-1;
        updates[num_of_updates++] = -sub_index1-1;
    }

    void clear(int16_t *acc, bool clean)
    {
        fused = FU_NONE;
        num_of_updates = 0;
        refresh = false;
        acculumator = acc;
        is_clean = clean;
    }
};


template <int INPUTS, int NEURONS>
struct nnue_perspective
{
    nnue_perspective(nnue_perspective_weights<INPUTS, NEURONS> *w) {
        weights = w;
        neurons_buffer = new int16_t[NEURONS+64];
        acculumator_buffer = new int16_t[NEURONS*130+64];

        neurons = align_ptr(neurons_buffer);
        acculumator = align_ptr(acculumator_buffer);

        outputs_idx_buffer = new int16_t[NEURONS+64];
        outputs_idx = align_ptr(outputs_idx_buffer);

        if (shuffle_lut_buffer == nullptr) {
            shuffle_lut_buffer = new uint8_t[16*256+64];
            shuffle_lut = align_ptr(shuffle_lut_buffer);

            for (int m = 0; m < 256; m++) {
                uint8_t *ctl = &shuffle_lut[m*16];
                int w = 0;
                for (int i = 0; i < 8; i++) {
                    if ((m & (1 << i)) != 0) {
                        ctl[w+0] = 2*i + 0;
                        ctl[w+1] = 2*i + 1;
                        w += 2;
                    }
                }
                for (; w < 16; w++) {
                    ctl[w] = 0x80;
                }
            }
        }

        update_table_index = 0;
        update_table = &update_tables[update_table_index];
    }

    ~nnue_perspective() {
        delete [] neurons_buffer;
        delete [] acculumator_buffer;

        delete [] outputs_idx_buffer;
    }

    void reset_stack()
    {
        acculumator = align_ptr(acculumator_buffer);
        update_table_index = 0;
        update_table = &update_tables[update_table_index];

        reset();
    }


    void reset() {
        int16_t *acc = (int16_t*)__builtin_assume_aligned(acculumator, 64);
        int16_t *bias = (int16_t*)__builtin_assume_aligned(weights->biases, 64);

        for (int i = 0; i < NEURONS; i += 32) {
            _mm256_store_si256((__m256i*)&acc[i], _mm256_load_si256((__m256i*)&bias[i]));
            _mm256_store_si256((__m256i*)&acc[i+16], _mm256_load_si256((__m256i*)&bias[i+16]));
        }

        update_tables[update_table_index].clear(acculumator, true);
    }


    void acculumator_copy(int16_t *dst, int16_t *src)
    {
        for (int i = 0; i < NEURONS; i += 64) {
            _mm256_store_si256((__m256i*)&dst[i], _mm256_load_si256((__m256i*)&src[i]));
            _mm256_store_si256((__m256i*)&dst[i+16], _mm256_load_si256((__m256i*)&src[i+16]));
            _mm256_store_si256((__m256i*)&dst[i+32], _mm256_load_si256((__m256i*)&src[i+32]));
            _mm256_store_si256((__m256i*)&dst[i+48], _mm256_load_si256((__m256i*)&src[i+48]));
        }
    }

    void acculumator_add(int16_t *acc, int index)
    {
        int16_t *weight = &weights->weights[index*NEURONS];

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&acc[i]);
            __m256i b0 = _mm256_load_si256((__m256i*)&weight[i]);

            __m256i a1 = _mm256_load_si256((__m256i*)&acc[i+16]);
            __m256i b1 = _mm256_load_si256((__m256i*)&weight[i+16]);


            a0 = _mm256_add_epi16(a0, b0);
            a1 = _mm256_add_epi16(a1, b1);

            _mm256_store_si256((__m256i*)&acc[i], a0);
            _mm256_store_si256((__m256i*)&acc[i+16], a1);
        }
    }

    void acculumator_sub(int16_t *acc, int index)
    {
        int16_t *weight = &weights->weights[index*NEURONS];

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&acc[i]);
            __m256i b0 = _mm256_load_si256((__m256i*)&weight[i]);

            __m256i a1 = _mm256_load_si256((__m256i*)&acc[i+16]);
            __m256i b1 = _mm256_load_si256((__m256i*)&weight[i+16]);


            a0 = _mm256_sub_epi16(a0, b0);
            a1 = _mm256_sub_epi16(a1, b1);

            _mm256_store_si256((__m256i*)&acc[i], a0);
            _mm256_store_si256((__m256i*)&acc[i+16], a1);
        }
    }

    void acculumator_addsub(int16_t *acc, int add_index, int sub_index)
    {
        int16_t *add_weight = &weights->weights[add_index*NEURONS];
        int16_t *sub_weight = &weights->weights[sub_index*NEURONS];

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&acc[i]);
            __m256i a1 = _mm256_load_si256((__m256i*)&acc[i+16]);

            __m256i sub0 = _mm256_load_si256((__m256i*)&sub_weight[i]);
            __m256i sub1 = _mm256_load_si256((__m256i*)&sub_weight[i+16]);

            __m256i add0 = _mm256_load_si256((__m256i*)&add_weight[i]);
            __m256i add1 = _mm256_load_si256((__m256i*)&add_weight[i+16]);

            a0 = _mm256_add_epi16(_mm256_sub_epi16(a0, sub0), add0);
            a1 = _mm256_add_epi16(_mm256_sub_epi16(a1, sub1), add1);

            _mm256_store_si256((__m256i*)&acc[i], a0);
            _mm256_store_si256((__m256i*)&acc[i+16], a1);
        }
    }

    void acculumator_addsubsub(int16_t *acc, int add_index, int sub_index0, int sub_index1)
    {
        int16_t *add_weight = &weights->weights[add_index*NEURONS];
        int16_t *sub_weight0 = &weights->weights[sub_index0*NEURONS];
        int16_t *sub_weight1 = &weights->weights[sub_index1*NEURONS];

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&acc[i]);
            __m256i a1 = _mm256_load_si256((__m256i*)&acc[i+16]);

            __m256i sub00 = _mm256_load_si256((__m256i*)&sub_weight0[i]);
            __m256i sub01 = _mm256_load_si256((__m256i*)&sub_weight0[i+16]);

            __m256i sub10 = _mm256_load_si256((__m256i*)&sub_weight1[i]);
            __m256i sub11 = _mm256_load_si256((__m256i*)&sub_weight1[i+16]);

            __m256i add0 = _mm256_load_si256((__m256i*)&add_weight[i]);
            __m256i add1 = _mm256_load_si256((__m256i*)&add_weight[i+16]);

            a0 = _mm256_add_epi16(_mm256_sub_epi16(_mm256_sub_epi16(a0, sub00), sub10), add0);
            a1 = _mm256_add_epi16(_mm256_sub_epi16(_mm256_sub_epi16(a1, sub01), sub11), add1);

            _mm256_store_si256((__m256i*)&acc[i], a0);
            _mm256_store_si256((__m256i*)&acc[i+16], a1);
        }
    }


    void apply_updates(acculumator_update_table *table)
    {
        if (table->fused == FU_ADDSUB) {
            acculumator_addsub(table->acculumator, table->updates[0]-1, -table->updates[1]-1);
        } else if (table->fused == FU_ADDSUBSUB) {
            acculumator_addsubsub(table->acculumator, table->updates[0]-1, -table->updates[1]-1, -table->updates[2]-1);
        } else {
            for (int i = 0; i < table->num_of_updates; i++) {
                int op = table->updates[i];
                int index = std::abs(op)-1;

                if (op > 0) {
                    acculumator_add(table->acculumator, index);
                } else {
                    acculumator_sub(table->acculumator, index);
                }
            }
        }
        table->clear(table->acculumator, true);
    }

    int apply_all_updates()
    {
        bool is_refresh = false;

        int start = update_table_index;
        while (update_tables[start].is_clean == false && start > 0) {
            start--;
        }

        int16_t *prev_acculumator = nullptr;
        for (int i = start; i <= update_table_index; i++) {
            acculumator_update_table *table = &update_tables[i];

            if (prev_acculumator != nullptr) {
                acculumator_copy(table->acculumator, prev_acculumator);
            }
            prev_acculumator = table->acculumator;

            if (table->refresh) {
                is_refresh = true;
            }

            apply_updates(table);
        }

        return is_refresh;
    }

    void update() {
        if (!update_table->is_clean || update_table->num_of_updates > 0) {
            apply_all_updates();
        }

        int16_t *accul = (int16_t*)__builtin_assume_aligned(acculumator, 64);
        int16_t *neuron = (int16_t*)__builtin_assume_aligned(neurons, 64);

        num_of_outputs = 0;

        __m128i idx_lo = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        __m128i idx_hi = _mm_set_epi16(15, 14, 13, 12, 11, 10, 9, 8);

        const __m128i idx_add = _mm_set1_epi16(16);

        const __m256i minv = _mm256_set1_epi16(0);
        const __m256i maxv = _mm256_set1_epi16(halfkp_quantization_fractions);

        for (int i = 0; i < NEURONS; i += 16) {
            __m256i acc = _mm256_load_si256((__m256i*)&accul[i]);

            const __m256i z = _mm256_setzero_si256();

            uint32_t gt_mm = _mm256_movemask_epi8(_mm256_packs_epi16(_mm256_cmpgt_epi16(acc, z), z));

            uint32_t gt_lo = gt_mm & 0xFF;
            uint32_t gt_hi = gt_mm >> 16;

            __m128i ctl_lo = _mm_load_si128((__m128i*)&shuffle_lut[gt_lo*16]);
            __m128i ctl_hi = _mm_load_si128((__m128i*)&shuffle_lut[gt_hi*16]);

            int k_lo = _mm_popcnt_u32(gt_lo);
            int k_hi = _mm_popcnt_u32(gt_hi);

            __m256i act = _mm256_min_epi16(_mm256_max_epi16(acc, minv), maxv);

            _mm256_store_si256((__m256i*)&neuron[i], act);

            __m128i packed_idx_lo = _mm_shuffle_epi8(idx_lo, ctl_lo);
            __m128i packed_idx_hi = _mm_shuffle_epi8(idx_hi, ctl_hi);

            _mm_storeu_si128((__m128i*)(outputs_idx + num_of_outputs), packed_idx_lo);
            _mm_storeu_si128((__m128i*)(outputs_idx + num_of_outputs + k_lo), packed_idx_hi);

            num_of_outputs += k_lo + k_hi;

            idx_lo = _mm_add_epi16(idx_lo, idx_add);
            idx_hi = _mm_add_epi16(idx_hi, idx_add);
        }

        neurons[NEURONS] = 0;
        _mm_storeu_si128((__m128i*)(outputs_idx + num_of_outputs), _mm_set1_epi16(NEURONS));
    }

    void push_acculumator() {
        acculumator += NEURONS;

        update_table_index += 1;
        update_tables[update_table_index].clear(acculumator, false);
        update_table = &update_tables[update_table_index];
    }

    void pop_acculumator() {
        acculumator -= NEURONS;

        update_table_index -= 1;
        update_table = &update_tables[update_table_index];
    }

    acculumator_update_table *update_table;
    int16_t *neurons;
    int16_t *outputs_idx;
    int num_of_outputs;

    int16_t *acculumator;
private:

    nnue_perspective_weights<INPUTS, NEURONS> *weights;

    int16_t *neurons_buffer;
    int16_t *acculumator_buffer;
    int16_t *outputs_idx_buffer;

    int update_table_index;
    acculumator_update_table update_tables[130];

    static uint8_t *shuffle_lut;
    static uint8_t *shuffle_lut_buffer;
};

template <int INPUTS, int NEURONS>
uint8_t* nnue_perspective<INPUTS, NEURONS>::shuffle_lut = nullptr;

template <int INPUTS, int NEURONS>
uint8_t* nnue_perspective<INPUTS, NEURONS>::shuffle_lut_buffer = nullptr;




