#pragma once

#define TT_SLOT_DEPTH_PREFERRED 0
#define TT_SLOT_ALWAYS_REPLACE 1


constexpr int32_t TT_MAX_EVAL = std::numeric_limits<int16_t>::max() - 512;
constexpr int32_t TT_MIN_EVAL = -TT_MAX_EVAL;


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


struct eval_table_entry
{
    eval_table_entry() {};

    void clear()
    {
        key = 0;
        score = 0;
    }

    uint64_t key;
    int32_t score;
    bool read(const uint64_t &zhash, int32_t &score_out) const  {
        eval_table_entry entry;
        memcpy(&entry, this, sizeof(eval_table_entry));

        if ((zhash ^ ((int64_t)entry.score)) == entry.key) {
            score_out = entry.score;

            return true;
        } else {
            return false;
        }
    }

    void write(const uint64_t &zhash, int32_t score_to_write) {

        eval_table_entry entry;
        entry.score = score_to_write;
        entry.key = (zhash ^ ((int64_t)entry.score));

        memcpy(this, &entry, sizeof(eval_table_entry));
    }
};


struct tt_entry_slot
{
    tt_entry_slot() {};

    void clear()
    {
        data = 0;
        key = 0;
        static_eval_type = 0x7FFF;
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

    inline void decode(chess_move &bm, int &d, int &t, int32_t &e, int32_t &se, int ply) const
    {
        bm.from = (best_move >> 10) & 0x3F;
        bm.to = (best_move >> 4) & 0x3F;
        bm.promotion = (piece_type_t)(best_move & 0xF);

        d = (age_depth & 0x7f);
        t = static_eval_type & 0x3;
        e = score;
        se = (static_eval_type >> 2);

        e = score_from_tt(e, ply);
    }
};


struct tt_entry
{
    tt_entry() {};

    void clear()
    {
        slots[0].clear();
        slots[1].clear();
    }

    int used(int current_cache_age) {
        bool slot0_free = ((slots[TT_SLOT_DEPTH_PREFERRED].age_depth >> 7) + 2) <= current_cache_age || slots[TT_SLOT_DEPTH_PREFERRED].static_eval_type == 0x7FFF;
        bool slot1_free = ((slots[TT_SLOT_ALWAYS_REPLACE].age_depth >> 7)  + 2) <= current_cache_age || slots[TT_SLOT_ALWAYS_REPLACE].static_eval_type  == 0x7FFF;
        return 2 - slot0_free - slot1_free;
    }

    bool read(const uint64_t &zhash, chess_move &bm, int &d, int &t, int32_t &e, int32_t &se, int ply) const {
        tt_entry_slot entry0;
        tt_entry_slot entry1;

        memcpy(&entry0, &slots[TT_SLOT_DEPTH_PREFERRED], sizeof(tt_entry_slot));
        memcpy(&entry1, &slots[TT_SLOT_ALWAYS_REPLACE], sizeof(tt_entry_slot));

        if (entry0.is_valid(zhash)) {

            entry0.decode(bm, d, t, e, se, ply);

            return true;
        } else if (entry1.is_valid(zhash)) {

            entry1.decode(bm, d, t, e, se, ply);

            return true;
        }

        return false;
    }

    void write(const uint64_t &zhash, chess_move bm, int d, int t, int32_t e, int32_t se, int ply, int current_age) {
        tt_entry_slot entry;

        entry.encode(zhash, bm, d, t, e, se, ply, current_age);

        if (entry.get_effective_depth(current_age) >= slots[TT_SLOT_DEPTH_PREFERRED].get_effective_depth(current_age)) {
            memcpy(&slots[TT_SLOT_ALWAYS_REPLACE], &slots[TT_SLOT_DEPTH_PREFERRED], sizeof(tt_entry_slot));
            memcpy(&slots[TT_SLOT_DEPTH_PREFERRED], &entry, sizeof(tt_entry_slot));
        } else {
            memcpy(&slots[TT_SLOT_ALWAYS_REPLACE], &entry, sizeof(tt_entry_slot));
        }
    }

    tt_entry_slot slots[2];
};
















