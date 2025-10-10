#pragma once

#include <stdint.h>
#include <x86gprintrin.h>
#include <x86intrin.h>

#if defined(__BMI2__)
    #define USE_PEXT 1
#else
    #define USE_PEXT 0
#endif

typedef uint64_t bitboard;

inline uint32_t bit_scan_forward_32(uint32_t u32) {
    uint32_t r = 0;
    __asm__ ("rep bsf\t%1, %0" : "+r"(r) : "r"(u32));
    return r;
}


inline uint64_t bit_scan_forward(uint64_t u64) {
    uint64_t r = 0;
    __asm__ ("rep bsfq\t%1, %0" : "+r"(r) : "r"(u64));
    return r;
}


inline uint_fast8_t bit_scan_forward_clear(uint64_t &u64)
{
    uint_fast8_t index = bit_scan_forward (u64);
    u64 &= u64-1;

    return index;
}


inline int32_t pop_count(uint64_t u64) {
    return (int32_t)_mm_popcnt_u64(u64);//__builtin_popcountll(u64);
}

struct sliding_piece_pattern
{
    uint64_t pattern;
    uint32_t offset;

    #if !USE_PEXT

    uint64_t magic;
    uint8_t shift;

    #endif // USE_PEXT
};


struct bitboard_utility {
    int distance_to_edge_table[64][8];

    bitboard king_pattern[64];
    bitboard knight_pattern[64];

    bitboard pawn_attack_pattern[64][2];
    bitboard pawn_advance_pattern[64][2];
    uint8_t pawn_block_square[64][2];

    sliding_piece_pattern bishop_pattern[64];
    sliding_piece_pattern rook_pattern[64];

    bitboard bishop_lookup[4096*64];
    bitboard rook_lookup[4096*64];

    bitboard bishop_xray_lookup[4096*64];
    bitboard rook_xray_lookup[4096*64];


    bitboard_utility();


    inline bool get_occupation(uint_fast8_t sq_index, const bitboard &occ)
    {
        bitboard sq = ((uint64_t)0x1 << sq_index);

        return ((occ & sq) != 0);
    }

    inline void set_occupation(uint_fast8_t sq_index, bitboard &occ, bool set_occ)
    {
        bitboard sq = ((uint64_t)0x1 << sq_index);

        if (set_occ) {
            occ |= sq;
        } else {
            occ &= (~sq);
        }
    }


    inline bitboard rook_attack(uint_fast8_t square_index, const bitboard &occupation, const bitboard &mask) {
        #if USE_PEXT
        uint64_t pattern = rook_pattern[square_index].pattern;
        uint32_t offset = rook_pattern[square_index].offset;
        return (rook_lookup[offset + _pext_u64(occupation, pattern)] & mask);
        #else
        uint64_t pattern = rook_pattern[square_index].pattern;
        uint32_t offset = rook_pattern[square_index].offset;

        uint64_t magic = rook_pattern[square_index].magic;
        uint8_t shift = rook_pattern[square_index].shift;

        return (rook_lookup[offset + ((occupation & pattern)*magic >> shift)] & mask);
        #endif
    }

    inline bitboard bishop_attack(uint_fast8_t square_index, const bitboard &occupation, const bitboard &mask) {
        #if USE_PEXT

        uint64_t pattern = bishop_pattern[square_index].pattern;
        uint32_t offset = bishop_pattern[square_index].offset;

        return (bishop_lookup[offset + _pext_u64(occupation, pattern)] & mask);

        #else

        uint64_t pattern = bishop_pattern[square_index].pattern;
        uint32_t offset = bishop_pattern[square_index].offset;

        uint64_t magic = bishop_pattern[square_index].magic;
        uint8_t shift = bishop_pattern[square_index].shift;

        return (bishop_lookup[offset + ((occupation & pattern)*magic >> shift)] & mask);

        #endif
    }

    inline bitboard queen_attack(uint_fast8_t square_index, const bitboard &occupation, const bitboard &mask) {
        bitboard rook_style = rook_attack(square_index, occupation, mask);
        bitboard bishop_style = bishop_attack(square_index, occupation, mask);

        return (rook_style | bishop_style);
    }

    inline bitboard king_attack(uint_fast8_t square_index, const bitboard &mask) {
        return (king_pattern[square_index] & mask);
    }

    inline bitboard knight_attack(uint_fast8_t square_index, const bitboard &mask) {
        return (knight_pattern[square_index] & mask);
    }

    inline bitboard pawn_advance(uint_fast8_t square_index, const bitboard &occupation, const uint_fast8_t &black) {
        if (get_occupation(pawn_block_square[square_index][black], occupation)) {
            return 0;
        }
        return (pawn_advance_pattern[square_index][black] & (~occupation));
    }

    inline bitboard pawn_attack(uint_fast8_t square_index, const bitboard &occupation, const uint_fast8_t &black, const bitboard &mask, const bitboard &en_passant) {
        return (pawn_attack_pattern[square_index][black] & (occupation | en_passant) & mask);
    }


    inline bitboard rook_xray_attack(uint_fast8_t square_index, const bitboard &occupation, const bitboard &mask) {
        #if USE_PEXT
        uint64_t pattern = rook_pattern[square_index].pattern;
        uint32_t offset = rook_pattern[square_index].offset;

        return (rook_xray_lookup[offset + _pext_u64(occupation, pattern)] & mask);
        #else
        uint64_t pattern = rook_pattern[square_index].pattern;
        uint32_t offset = rook_pattern[square_index].offset;

        uint64_t magic = rook_pattern[square_index].magic;
        uint8_t shift = rook_pattern[square_index].shift;

        return (rook_xray_lookup[offset + ((occupation & pattern)*magic >> shift)] & mask);
        #endif
    }

    inline bitboard bishop_xray_attack(uint_fast8_t square_index, const bitboard &occupation, const bitboard &mask) {
        #if USE_PEXT

        uint64_t pattern = bishop_pattern[square_index].pattern;
        uint32_t offset = bishop_pattern[square_index].offset;

        return (bishop_xray_lookup[offset + _pext_u64(occupation, pattern)] & mask);

        #else

        uint64_t pattern = bishop_pattern[square_index].pattern;
        uint32_t offset = bishop_pattern[square_index].offset;

        uint64_t magic = bishop_pattern[square_index].magic;
        uint8_t shift = bishop_pattern[square_index].shift;

        return (bishop_xray_lookup[offset + ((occupation & pattern)*magic >> shift)] & mask);

        #endif
    }

    inline bitboard queen_xray_attack(uint_fast8_t square_index, const bitboard &occupation, const bitboard &mask) {
        bitboard rook_style = rook_xray_attack(square_index, occupation, mask);
        bitboard bishop_style = bishop_xray_attack(square_index, occupation, mask);

        return (rook_style | bishop_style);
    }
private:
    void init_knight_tables();
    void init_pawn_tables(bool black);
    void init_rook_lookup();
    void init_bishop_lookup();

    bitboard get_sliding_xray_pattern(int sq_index, bitboard occ, bool rook_style, bool bishop_style, bool include_edges);
    bitboard get_sliding_pattern(int sq_index, bitboard occ, bool rook_style, bool bishop_style, bool include_edges);

    void generate_sliding_tables();
    void init_board_area_masks();

    void init_passed_pawn_lookups();
    void init_file_rank_lookups();
    void init_king_tables();
};

extern bitboard_utility bitboard_utils;

void print_bitboard(bitboard bb);



