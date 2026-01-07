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

    void save(int16_t *data, size_t &index)
    {
        for (size_t i = 0; i < NEURONS*INPUTS; i++) {
            data[index++] = weights[i];
        }
        for (size_t i = 0; i < NEURONS; i++) {
            data[index++] = biases[i];
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

    void op_add(int index)
    {
        fused = FU_NONE;
        updates[num_of_updates++] = index+1;
    }

    void op_sub(int index)
    {
        fused = FU_NONE;
        updates[num_of_updates++] = -index-1;
    }

    void op_addsub(int add_index, int sub_index)
    {
        fused = (num_of_updates == 0 ? FU_ADDSUB : FU_NONE);
        updates[num_of_updates+0] = add_index+1;
        updates[num_of_updates+1] = -sub_index-1;

        num_of_updates += 2;
    }

    void op_addsubsub(int add_index, int sub_index0, int sub_index1)
    {
        fused = (num_of_updates == 0 ? FU_ADDSUBSUB : FU_NONE);
        updates[num_of_updates+0] = add_index+1;
        updates[num_of_updates+1] = -sub_index0-1;
        updates[num_of_updates+2] = -sub_index1-1;

        num_of_updates += 3;
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
        outputs_idx_buffer = new int16_t[NEURONS+64];

        neurons = align_ptr(neurons_buffer);
        acculumator = align_ptr(acculumator_buffer);
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

        acculumator_copy(acc, bias);

        update_tables[update_table_index].clear(acculumator, true);
    }


    void acculumator_copy(int16_t *dst, int16_t *src)
    {
        #if USE_AVX2
        for (int i = 0; i < NEURONS; i += 64) {
            _mm256_store_si256((__m256i*)&dst[i], _mm256_load_si256((__m256i*)&src[i]));
            _mm256_store_si256((__m256i*)&dst[i+16], _mm256_load_si256((__m256i*)&src[i+16]));
            _mm256_store_si256((__m256i*)&dst[i+32], _mm256_load_si256((__m256i*)&src[i+32]));
            _mm256_store_si256((__m256i*)&dst[i+48], _mm256_load_si256((__m256i*)&src[i+48]));
        }
        #else
        for (int i = 0; i < NEURONS; i += 32) {
            _mm_store_si128((__m128i*)&dst[i], _mm_load_si128((__m128i*)&src[i]));
            _mm_store_si128((__m128i*)&dst[i+8], _mm_load_si128((__m128i*)&src[i+8]));
            _mm_store_si128((__m128i*)&dst[i+16], _mm_load_si128((__m128i*)&src[i+16]));
            _mm_store_si128((__m128i*)&dst[i+24], _mm_load_si128((__m128i*)&src[i+24]));
        }
        #endif // USE_AVX2
    }

    void acculumator_add(int16_t *src, int16_t *dst, int index)
    {
        int16_t *weight = &weights->weights[index*NEURONS];

        #if USE_AVX2

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&src[i]);
            __m256i b0 = _mm256_load_si256((__m256i*)&weight[i]);

            __m256i a1 = _mm256_load_si256((__m256i*)&src[i+16]);
            __m256i b1 = _mm256_load_si256((__m256i*)&weight[i+16]);


            a0 = _mm256_add_epi16(a0, b0);
            a1 = _mm256_add_epi16(a1, b1);

            _mm256_store_si256((__m256i*)&dst[i], a0);
            _mm256_store_si256((__m256i*)&dst[i+16], a1);
        }

        #else

        for (int i = 0; i < NEURONS; i += 16) {
            __m128i a0 = _mm_load_si128((__m128i*)&src[i]);
            __m128i b0 = _mm_load_si128((__m128i*)&weight[i]);

            __m128i a1 = _mm_load_si128((__m128i*)&src[i+8]);
            __m128i b1 = _mm_load_si128((__m128i*)&weight[i+8]);


            a0 = _mm_add_epi16(a0, b0);
            a1 = _mm_add_epi16(a1, b1);

            _mm_store_si128((__m128i*)&dst[i], a0);
            _mm_store_si128((__m128i*)&dst[i+8], a1);
        }

        #endif // USE_AVX2
    }

    void acculumator_sub(int16_t *src, int16_t *dst, int index)
    {
        int16_t *weight = &weights->weights[index*NEURONS];

        #if USE_AVX2

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&src[i]);
            __m256i b0 = _mm256_load_si256((__m256i*)&weight[i]);

            __m256i a1 = _mm256_load_si256((__m256i*)&src[i+16]);
            __m256i b1 = _mm256_load_si256((__m256i*)&weight[i+16]);


            a0 = _mm256_sub_epi16(a0, b0);
            a1 = _mm256_sub_epi16(a1, b1);

            _mm256_store_si256((__m256i*)&dst[i], a0);
            _mm256_store_si256((__m256i*)&dst[i+16], a1);
        }

        #else

        for (int i = 0; i < NEURONS; i += 16) {
            __m128i a0 = _mm_load_si128((__m128i*)&src[i]);
            __m128i b0 = _mm_load_si128((__m128i*)&weight[i]);

            __m128i a1 = _mm_load_si128((__m128i*)&src[i+8]);
            __m128i b1 = _mm_load_si128((__m128i*)&weight[i+8]);


            a0 = _mm_sub_epi16(a0, b0);
            a1 = _mm_sub_epi16(a1, b1);

            _mm_store_si128((__m128i*)&dst[i], a0);
            _mm_store_si128((__m128i*)&dst[i+8], a1);
        }

        #endif
    }

    void acculumator_addsub(int16_t *src, int16_t *dst, int add_index, int sub_index)
    {
        int16_t *add_weight = &weights->weights[add_index*NEURONS];
        int16_t *sub_weight = &weights->weights[sub_index*NEURONS];

        #if USE_AVX2

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&src[i]);
            __m256i a1 = _mm256_load_si256((__m256i*)&src[i+16]);

            __m256i sub0 = _mm256_load_si256((__m256i*)&sub_weight[i]);
            __m256i sub1 = _mm256_load_si256((__m256i*)&sub_weight[i+16]);

            __m256i add0 = _mm256_load_si256((__m256i*)&add_weight[i]);
            __m256i add1 = _mm256_load_si256((__m256i*)&add_weight[i+16]);

            a0 = _mm256_add_epi16(_mm256_sub_epi16(a0, sub0), add0);
            a1 = _mm256_add_epi16(_mm256_sub_epi16(a1, sub1), add1);

            _mm256_store_si256((__m256i*)&dst[i], a0);
            _mm256_store_si256((__m256i*)&dst[i+16], a1);
        }

        #else

        for (int i = 0; i < NEURONS; i += 16) {
            __m128i a0 = _mm_load_si128((__m128i*)&src[i]);
            __m128i a1 = _mm_load_si128((__m128i*)&src[i+8]);

            __m128i sub0 = _mm_load_si128((__m128i*)&sub_weight[i]);
            __m128i sub1 = _mm_load_si128((__m128i*)&sub_weight[i+8]);

            __m128i add0 = _mm_load_si128((__m128i*)&add_weight[i]);
            __m128i add1 = _mm_load_si128((__m128i*)&add_weight[i+8]);

            a0 = _mm_add_epi16(_mm_sub_epi16(a0, sub0), add0);
            a1 = _mm_add_epi16(_mm_sub_epi16(a1, sub1), add1);

            _mm_store_si128((__m128i*)&dst[i], a0);
            _mm_store_si128((__m128i*)&dst[i+8], a1);
        }

        #endif // USE_AVX2
    }

    void acculumator_addsubsub(int16_t *src, int16_t *dst, int add_index, int sub_index0, int sub_index1)
    {
        int16_t *add_weight = &weights->weights[add_index*NEURONS];
        int16_t *sub_weight0 = &weights->weights[sub_index0*NEURONS];
        int16_t *sub_weight1 = &weights->weights[sub_index1*NEURONS];


        #if USE_AVX2

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i a0 = _mm256_load_si256((__m256i*)&src[i]);
            __m256i a1 = _mm256_load_si256((__m256i*)&src[i+16]);

            __m256i sub00 = _mm256_load_si256((__m256i*)&sub_weight0[i]);
            __m256i sub01 = _mm256_load_si256((__m256i*)&sub_weight0[i+16]);

            __m256i sub10 = _mm256_load_si256((__m256i*)&sub_weight1[i]);
            __m256i sub11 = _mm256_load_si256((__m256i*)&sub_weight1[i+16]);

            __m256i add0 = _mm256_load_si256((__m256i*)&add_weight[i]);
            __m256i add1 = _mm256_load_si256((__m256i*)&add_weight[i+16]);

            a0 = _mm256_add_epi16(_mm256_sub_epi16(_mm256_sub_epi16(a0, sub00), sub10), add0);
            a1 = _mm256_add_epi16(_mm256_sub_epi16(_mm256_sub_epi16(a1, sub01), sub11), add1);

            _mm256_store_si256((__m256i*)&dst[i], a0);
            _mm256_store_si256((__m256i*)&dst[i+16], a1);
        }

        #else

        for (int i = 0; i < NEURONS; i += 16) {
            __m128i a0 = _mm_load_si128((__m128i*)&src[i]);
            __m128i a1 = _mm_load_si128((__m128i*)&src[i+8]);

            __m128i sub00 = _mm_load_si128((__m128i*)&sub_weight0[i]);
            __m128i sub01 = _mm_load_si128((__m128i*)&sub_weight0[i+8]);

            __m128i sub10 = _mm_load_si128((__m128i*)&sub_weight1[i]);
            __m128i sub11 = _mm_load_si128((__m128i*)&sub_weight1[i+8]);

            __m128i add0 = _mm_load_si128((__m128i*)&add_weight[i]);
            __m128i add1 = _mm_load_si128((__m128i*)&add_weight[i+8]);

            a0 = _mm_add_epi16(_mm_sub_epi16(_mm_sub_epi16(a0, sub00), sub10), add0);
            a1 = _mm_add_epi16(_mm_sub_epi16(_mm_sub_epi16(a1, sub01), sub11), add1);

            _mm_store_si128((__m128i*)&dst[i], a0);
            _mm_store_si128((__m128i*)&dst[i+8], a1);
        }

        #endif
    }


    void apply_updates(acculumator_update_table *table, int16_t *src_acc)
    {
        if (table->fused == FU_ADDSUB) {
            acculumator_addsub(src_acc, table->acculumator, table->updates[0]-1, -table->updates[1]-1);
        } else if (table->fused == FU_ADDSUBSUB) {
            acculumator_addsubsub(src_acc, table->acculumator, table->updates[0]-1, -table->updates[1]-1, -table->updates[2]-1);
        } else {
            for (int i = 0; i < table->num_of_updates; i++) {
                int op = table->updates[i];
                int index = std::abs(op)-1;

                if (op > 0) {
                    acculumator_add(src_acc, table->acculumator, index);
                } else {
                    acculumator_sub(src_acc, table->acculumator, index);
                }
                src_acc = table->acculumator;
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
        int16_t *prev_acculumator = update_tables[start].acculumator;

        start += (update_tables[start].num_of_updates == 0);

        for (int i = start; i <= update_table_index; i++) {
            is_refresh |= update_tables[i].refresh;

            apply_updates(&update_tables[i], prev_acculumator);

            prev_acculumator = update_tables[i].acculumator;
        }

        return is_refresh;
    }

    void update_activations() {
        int16_t *accul = (int16_t*)__builtin_assume_aligned(acculumator, 64);
        int16_t *neuron = (int16_t*)__builtin_assume_aligned(neurons, 64);

        num_of_outputs = 0;

        #if USE_AVX2

        const __m256i z = _mm256_setzero_si256();
        const __m256i minv = _mm256_set1_epi16(0);
        const __m256i maxv = _mm256_set1_epi16(halfkp_quantization_fractions);

        __m256i idx0 = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        __m256i idx1 = _mm256_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16);
        const __m256i idx_add = _mm256_set1_epi16(32);

        for (int i = 0; i < NEURONS; i += 32) {
            __m256i acc0 = _mm256_load_si256((__m256i*)&accul[i]);
            __m256i acc1 = _mm256_load_si256((__m256i*)&accul[i+16]);

            uint32_t gt_mm = _mm256_movemask_epi8(_mm256_packs_epi16(_mm256_cmpgt_epi16(acc0, z), _mm256_cmpgt_epi16(acc1, z)));

            uint32_t gt_lo0 = (gt_mm <<  4) & 0xFF0;
            uint32_t gt_hi0 = (gt_mm >> 12) & 0xFF0;

            uint32_t gt_lo1 = (gt_mm >>  4) & 0xFF0;
            uint32_t gt_hi1 = (gt_mm >> 20) & 0xFF0;

            __m128i ctl_lo0 = _mm_load_si128((__m128i*)&shuffle_lut[gt_lo0]);
            __m128i ctl_hi0 = _mm_load_si128((__m128i*)&shuffle_lut[gt_hi0]);

            __m128i ctl_lo1 = _mm_load_si128((__m128i*)&shuffle_lut[gt_lo1]);
            __m128i ctl_hi1 = _mm_load_si128((__m128i*)&shuffle_lut[gt_hi1]);

            __m256i ctl0  = _mm256_set_m128i(ctl_hi0, ctl_lo0);
            __m256i ctl1  = _mm256_set_m128i(ctl_hi1, ctl_lo1);

            __m256i act0 = _mm256_min_epi16(_mm256_max_epi16(acc0, minv), maxv);
            __m256i act1 = _mm256_min_epi16(_mm256_max_epi16(acc1, minv), maxv);

            _mm256_store_si256((__m256i*)&neuron[i], act0);
            _mm256_store_si256((__m256i*)&neuron[i+16], act1);

            int k_lo0 = _mm_popcnt_u32(gt_lo0);
            int k_hi0 = _mm_popcnt_u32(gt_hi0);

            int k_lo1 = _mm_popcnt_u32(gt_lo1);
            int k_hi1 = _mm_popcnt_u32(gt_hi1);

            __m256i packed_idx0 = _mm256_shuffle_epi8(idx0, ctl0);
            __m256i packed_idx1 = _mm256_shuffle_epi8(idx1, ctl1);

            _mm256_storeu2_m128i((__m128i*)&outputs_idx[num_of_outputs + k_lo0], (__m128i*)&outputs_idx[num_of_outputs], packed_idx0);
            num_of_outputs += k_lo0 + k_hi0;

            _mm256_storeu2_m128i((__m128i*)&outputs_idx[num_of_outputs + k_lo1], (__m128i*)&outputs_idx[num_of_outputs], packed_idx1);
            num_of_outputs += k_lo1 + k_hi1;


            idx0 = _mm256_add_epi16(idx0, idx_add);
            idx1 = _mm256_add_epi16(idx1, idx_add);
        }

        #else

        __m128i idx_lo = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
        __m128i idx_hi = _mm_set_epi16(15, 14, 13, 12, 11, 10, 9, 8);

        const __m128i idx_add = _mm_set1_epi16(16);

        const __m128i minv = _mm_set1_epi16(0);
        const __m128i maxv = _mm_set1_epi16(halfkp_quantization_fractions);

        for (int i = 0; i < NEURONS; i += 16) {
            __m128i acc0 = _mm_load_si128((__m128i*)&accul[i]);
            __m128i acc1 = _mm_load_si128((__m128i*)&accul[i+8]);

            const __m128i z = _mm_setzero_si128();

            uint32_t gt_lo = _mm_movemask_epi8(_mm_packs_epi16(_mm_cmpgt_epi16(acc0, z), z));
            uint32_t gt_hi = _mm_movemask_epi8(_mm_packs_epi16(_mm_cmpgt_epi16(acc1, z), z));

            __m128i ctl_lo = _mm_load_si128((__m128i*)&shuffle_lut[gt_lo*16]);
            __m128i ctl_hi = _mm_load_si128((__m128i*)&shuffle_lut[gt_hi*16]);

            __m128i act0 = _mm_min_epi16(_mm_max_epi16(acc0, minv), maxv);
            __m128i act1 = _mm_min_epi16(_mm_max_epi16(acc1, minv), maxv);

            _mm_store_si128((__m128i*)&neuron[i], act0);
            _mm_store_si128((__m128i*)&neuron[i+8], act1);

            int k_lo = _mm_popcnt_u32(gt_lo);
            int k_hi = _mm_popcnt_u32(gt_hi);

            __m128i packed_idx_lo = _mm_shuffle_epi8(idx_lo, ctl_lo);
            __m128i packed_idx_hi = _mm_shuffle_epi8(idx_hi, ctl_hi);

            _mm_storeu_si128((__m128i*)(outputs_idx + num_of_outputs), packed_idx_lo);
            _mm_storeu_si128((__m128i*)(outputs_idx + num_of_outputs + k_lo), packed_idx_hi);

            num_of_outputs += k_lo + k_hi;

            idx_lo = _mm_add_epi16(idx_lo, idx_add);
            idx_hi = _mm_add_epi16(idx_hi, idx_add);
        }

        #endif


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




