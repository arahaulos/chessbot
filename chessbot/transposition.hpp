#pragma once

#include "state.hpp"
#include "movegen.hpp"


constexpr int32_t TT_MAX_EVAL = std::numeric_limits<int16_t>::max() - 512;
constexpr int32_t TT_MIN_EVAL = -TT_MAX_EVAL;

constexpr int EVAL_CACHE_ENTRIES_PER_BUCKET = 4;
constexpr int TT_ENTRIES_IN_BUCKET = 4;

inline int32_t score_to_tt(int32_t score, int ply)
{
    if (is_mate_score(score)) {
        if (score <= MIN_EVAL + MAX_DEPTH) {
            int dist = score - MIN_EVAL - ply;

            return TT_MIN_EVAL + dist;

        } else if (score >= MAX_EVAL - MAX_DEPTH) {
            int dist = MAX_EVAL - score - ply;

            return TT_MAX_EVAL - dist;
        }
    }

    return score;
}

inline int32_t score_from_tt(int32_t tt_score, int ply)
{
    if (tt_score <= TT_MIN_EVAL + MAX_DEPTH*2) {
        int dist = std::min(tt_score - TT_MIN_EVAL + ply, MAX_DEPTH-1);

        return MIN_EVAL + dist;

    } else if (tt_score >= TT_MAX_EVAL - MAX_DEPTH*2) {
        int dist = std::min(TT_MAX_EVAL - tt_score + ply, MAX_DEPTH-1);

        return MAX_EVAL - dist;
    }
    return tt_score;
}


struct eval_cache_bucket
{
    eval_cache_bucket() { clear(); };

    void clear()
    {
        for (int i = 0; i < EVAL_CACHE_ENTRIES_PER_BUCKET; i++) {
            entries[i] = 0;
        }
    }

    void bring_front(int index, const uint64_t &new_entry)
    {
        for (int i = index; i >= 1; i--) {
            entries[i] = entries[i-1];
        }
        entries[0] = new_entry;
    }

    uint64_t entries[EVAL_CACHE_ENTRIES_PER_BUCKET];

    bool probe(const uint64_t &zhash, int32_t &score_out) {
        for (int i = 0; i < EVAL_CACHE_ENTRIES_PER_BUCKET; i++) {
            uint64_t entry = entries[i];
            int16_t score16 = entry & 0xFFFF;
            if ((entry & 0xFFFFFFFFFFFF0000ull) == (zhash & 0xFFFFFFFFFFFF0000ull)) {
                score_out = score16;

                bring_front(i, entry);

                return true;
            }
        }
        return false;
    }

    void store(const uint64_t &zhash, int32_t score_to_write) {
        uint64_t new_entry = (zhash & 0xFFFFFFFFFFFF0000ull) | (score_to_write & 0xFFFF);

        for (int i = 0; i < EVAL_CACHE_ENTRIES_PER_BUCKET; i++) {
            if ((entries[i] & 0xFFFFFFFFFFFF0000ull) == (zhash & 0xFFFFFFFFFFFF0000ull)) {
                bring_front(i, new_entry);
                return;
            }
        }
        bring_front(EVAL_CACHE_ENTRIES_PER_BUCKET - 1, new_entry);
    }
};



struct tt_entry
{
    tt_entry() {};

    void clear()
    {
        data = 0;
        key = 0;
        static_eval_type = 0x7FFF;
    }

    int get_depth() const {
        return (int)(age_depth & 0x7f);
    }

    int get_effective_depth(int current_cache_age) const {

        int age = (int)(age_depth >> 7);
        int depth = (int)(age_depth & 0x7f);
        int type = static_eval_type & 0x3;

        if (current_cache_age >= age + 2) {
            return -1;
        }
        if (type == PV_NODE) {
            return depth + 2;
        } else if (type == CUT_NODE) {
            return depth + 1;
        } else {
            return depth;
        }
    }

    bool is_valid(const uint64_t zhash) const {
        return ((zhash ^ data) == key);
    }

    union {
        struct {
            int16_t best_move;
            int16_t static_eval_type;
            uint16_t age_depth;
            int16_t score;
        };
        uint64_t data;
    };

    uint64_t key;

    inline void encode(const uint64_t &zhash, chess_move bm, int d, int t, int32_t e, int32_t se, int ply, int current_age)
    {
        e = score_to_tt(e, ply);

        best_move = (bm.from.index << 10) | (bm.to.index << 4) | bm.promotion;
        static_eval_type = (se << 2) | (t & 0x3);
        age_depth = ((current_age & 0x1ff) << 7) | (d & 0x7f);
        score = e;

        key = zhash ^ data;
    }

    inline void decode(const board_state &state, chess_move &bm, int &d, int &t, int32_t &e, int32_t &se, int ply) const
    {
        bm.from = (best_move >> 10) & 0x3F;
        bm.to = (best_move >> 4) & 0x3F;
        bm.promotion = (piece_type_t)(best_move & 0xF);

        d = (age_depth & 0x7f);
        t = static_eval_type & 0x3;
        e = score;
        se = (static_eval_type >> 2);

        e = score_from_tt(e, ply);

        bm.encoded_pieces = move_generator::encode_move_pieces(state, bm);
    }
};


struct tt_bucket
{
    tt_bucket() {};

    void clear()
    {
        for (int i = 0; i < TT_ENTRIES_IN_BUCKET; i++) {
            entries[i].clear();
        }
    }

    int used(int current_cache_age) {
        int used = 0;
        for (int i = 0; i < TT_ENTRIES_IN_BUCKET; i++) {
            if (((entries[i].age_depth >> 7) + 2) > current_cache_age && entries[i].static_eval_type != 0x7FFF) {
                used++;
            }
        }
        return used;
    }

    bool probe(const board_state &state, const uint64_t &zhash, chess_move &bm, int &d, int &t, int32_t &e, int32_t &se, int ply) const {

        tt_bucket buck;
        memcpy(&buck, this, sizeof(tt_bucket));

        for (int i = 0; i < TT_ENTRIES_IN_BUCKET; i++) {
            if (buck.entries[i].is_valid(zhash)) {
                buck.entries[i].decode(state, bm, d, t, e, se, ply);

                return true;
            }
        }

        return false;
    }

    void store(const uint64_t &zhash, chess_move bm, int d, int t, int32_t e, int32_t se, int ply, int current_age) {

        tt_entry new_entry;
        new_entry.encode(zhash, bm, d, t, e, se, ply, current_age);

        for (int i = 0; i < TT_ENTRIES_IN_BUCKET; i++) {
            if (entries[i].is_valid(zhash)) {
                entries[i] = new_entry;
                return;
            }
        }
        int lowest_edepth = entries[0].get_effective_depth(current_age);
        int lowest_entry = 0;
        for (int i = 1; i < TT_ENTRIES_IN_BUCKET; i++) {
            int edepth = entries[i].get_effective_depth(current_age);
            if (edepth < lowest_edepth) {
                lowest_entry = i;
                lowest_edepth = edepth;
            }
        }

        entries[lowest_entry] = new_entry;
    }

    tt_entry entries[TT_ENTRIES_IN_BUCKET];
};
















