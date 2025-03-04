#pragma once

#include <stdint.h>
#include <x86gprintrin.h>
#include <x86intrin.h>


typedef uint64_t bitboard;


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


struct bitboard_utility {

    int sliding_vectors_table[8*2];
    int distance_to_edge_table[64][8];

    uint32_t bishop_lookup_offset[64];
    uint32_t rook_lookup_offset[64];

    bitboard bishop_pattern0[64];
    bitboard rook_pattern0[64];

    bitboard king_pattern[64];
    bitboard knight_pattern[64];

    bitboard pawn_attack_pattern[64][2];
    bitboard pawn_advance_pattern[64][2];
    uint8_t pawn_block_square[64][2];

    bitboard bishop_lookup[4096*64];
    bitboard rook_lookup[4096*64];

    bitboard bishop_xray_lookup[4096*64];
    bitboard rook_xray_lookup[4096*64];

    bitboard bitboard_files[8];
    bitboard bitboard_ranks[8];
    bitboard bitboard_neighbor_files[8];

    bitboard passed_pawn_mask[64][2];

    bitboard center_board_mask;
    bitboard extended_center_board_mask;
    int distance_to_center[64];

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
        uint64_t pattern = rook_pattern0[square_index];
        return (rook_lookup[rook_lookup_offset[square_index] + _pext_u64(occupation, pattern)] & mask);
    }

    inline bitboard bishop_attack(uint_fast8_t square_index, const bitboard &occupation, const bitboard &mask) {
        uint64_t pattern = bishop_pattern0[square_index];
        return (bishop_lookup[bishop_lookup_offset[square_index] + _pext_u64(occupation, pattern)] & mask);
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


    inline bitboard get_file(uint_fast8_t square_index) {
        return bitboard_files[square_index % 8];
    }

    inline bitboard get_neighbor_file(uint_fast8_t square_index) {
        return bitboard_neighbor_files[square_index % 8];
    }

    inline bitboard get_rank(uint_fast8_t square_index) {
        return bitboard_ranks[square_index / 8];
    }



    inline bitboard rook_xray_attack(uint_fast8_t square_index, const bitboard &occupation, const bitboard &mask) {
        uint64_t pattern = rook_pattern0[square_index];
        return (rook_xray_lookup[rook_lookup_offset[square_index] + _pext_u64 (occupation, pattern)] & mask);
    }

    inline bitboard bishop_xray_attack(uint_fast8_t square_index, const bitboard &occupation, const bitboard &mask) {
        uint64_t pattern = bishop_pattern0[square_index];
        return (bishop_xray_lookup[bishop_lookup_offset[square_index] + _pext_u64 (occupation, pattern)] & mask);
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



