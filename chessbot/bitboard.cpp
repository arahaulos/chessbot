#include "bitboard.hpp"
#include "state.hpp"
#include <iostream>
#include <math.h>


uint64_t deposit_bits(uint64_t bits, uint64_t pattern)
{
    uint64_t result = 0;

    while (pattern) {
        uint8_t index = bit_scan_forward_clear(pattern);
        result |= ((bits & 0x1) << index);
        bits >>= 1;
    }

    return result;
}

uint64_t find_magic(const uint64_t &pattern, int bits)
{
    static uint64_t rand = 123456789123456789;

    int lookups = std::pow(2, bits);

    uint64_t occ[4096];
    bool used[4096];

    for (int i = 0; i < lookups; i++) {
        occ[i] = deposit_bits(i, pattern);
    }

    uint64_t magic = 0;
    uint8_t shift = 64 - bits;
    bool found = false;
    while (!found) {
        magic = ~0ull;
        for (int i = 0; i < 3; i++) {
            rand ^= rand >> 12;
            rand ^= rand << 25;
            rand ^= rand >> 27;
            magic &= (rand * 0x2545F4914F6CDD1DULL);
        }

        if (pop_count((pattern * magic) & 0xFF00000000000000ull) < 6) {
            continue;
        }
        for (int i = 0; i < lookups; i++) {
            used[i] = false;
        }
        found = true;
        for (int i = 0; i < lookups; i++) {
            uint64_t index = ((occ[i] * magic) >> shift);
            if (used[index]) {
                found = false;
                break;
            }
            used[index] = true;
        }
    }

    return magic;
}



static const int sliding_vectors_table[8*2] =
{
    1, 1, 1, -1, -1, 1, -1, -1, 1, 0, -1, 0, 0, 1, 0, -1
};


bitboard_utility bitboard_utils;

void bitboard_utility::generate_sliding_tables()
{
    for (int v = 0; v < 8; v++) {
        int vx = sliding_vectors_table[v*2+0];
        int vy = sliding_vectors_table[v*2+1];

        for (int i = 0; i < BOARD_WIDTH*BOARD_HEIGHT; i++) {
            int x = i % BOARD_WIDTH;
            int y = i / BOARD_WIDTH;

            int c = 0;

            for (int j = 0; j < 8; j++) {
                x += vx;
                y += vy;
                if (x < 0 || y < 0 || x >= BOARD_WIDTH || y >= BOARD_HEIGHT) {
                    break;
                }
                c++;
            }
            distance_to_edge_table[i][v] = c;
        }
    }
}


bitboard bitboard_utility::get_sliding_pattern(int sq_index, bitboard occ, bool rook_style, bool bishop_style, bool include_edges)
{
    bitboard moves = 0;
    int posx = sq_index % 8;
    int posy = sq_index / 8;

    int start_index = (rook_style && !bishop_style ? 4 : 0);
    int end_index = (bishop_style && !rook_style ? 4 : 8);


    for (int i = start_index; i < end_index; i++) {
        int vx = sliding_vectors_table[i*2 + 0];
        int vy = sliding_vectors_table[i*2 + 1];
        int x = posx;
        int y = posy;

        for (int j = 0; j < distance_to_edge_table[sq_index][i] - (include_edges ? 0 : 1); j++) {
            x += vx;
            y += vy;

            int new_sq_index = y * 8 + x;

            set_occupation(new_sq_index, moves, true);

            if (get_occupation(new_sq_index, occ)) {
                break;
            }
        }
    }
    return moves;
}


bitboard bitboard_utility::get_sliding_xray_pattern(int sq_index, bitboard occ, bool rook_style, bool bishop_style, bool include_edges)
{
    bitboard moves = 0;
    int posx = sq_index % 8;
    int posy = sq_index / 8;

    int start_index = (rook_style && !bishop_style ? 4 : 0);
    int end_index = (bishop_style && !rook_style ? 4 : 8);


    for (int i = start_index; i < end_index; i++) {
        int vx = sliding_vectors_table[i*2 + 0];
        int vy = sliding_vectors_table[i*2 + 1];
        int x = posx;
        int y = posy;

        bool xray = false;

        for (int j = 0; j < distance_to_edge_table[sq_index][i] - (include_edges ? 0 : 1); j++) {
            x += vx;
            y += vy;

            int new_sq_index = y * 8 + x;

            if (xray) {
                set_occupation(new_sq_index, moves, true);
            }

            if (get_occupation(new_sq_index, occ)) {
                if (xray) {
                    break;
                } else {
                    xray = true;
                }
            }
        }
    }
    return moves;
}

void bitboard_utility::init_bishop_lookup()
{
    uint32_t lookup_offset = 0;

    for (int i = 0; i < 64; i++) {

        bitboard pattern = get_sliding_pattern(i, 0, false, true, false);
        int bits = pop_count(pattern);
        int lookups = std::pow(2, bits);

        bishop_pattern[i].pattern = pattern;
        bishop_pattern[i].offset = lookup_offset;

        #if !USE_PEXT

        bishop_pattern[i].magic = find_magic(pattern, bits);
        bishop_pattern[i].shift = 64 - bits;

        #endif

        for (int j = 0; j < lookups; j++) {
            bitboard occ = deposit_bits(j, pattern);

            #if USE_PEXT
            uint32_t index = j;
            #else
            uint32_t index = ((occ * bishop_pattern[i].magic) >> bishop_pattern[i].shift);
            #endif


            bishop_lookup[lookup_offset + index] = get_sliding_pattern(i, occ, false, true, true);
            bishop_xray_lookup[lookup_offset + index] = get_sliding_xray_pattern(i, occ, false, true, true);
        }

        lookup_offset += lookups;
    }
}

void bitboard_utility::init_rook_lookup()
{
    uint32_t lookup_offset = 0;

    for (int i = 0; i < 64; i++) {
        bitboard pattern = get_sliding_pattern(i, 0, true, false, false);
        int bits = pop_count(pattern);
        int lookups = std::pow(2, bits);

        rook_pattern[i].pattern = pattern;
        rook_pattern[i].offset = lookup_offset;

        #if !USE_PEXT

        rook_pattern[i].magic = find_magic(pattern, bits);
        rook_pattern[i].shift = 64 - bits;

        #endif

        for (int j = 0; j < lookups; j++) {
            bitboard occ = deposit_bits(j, pattern);

            #if USE_PEXT
            uint32_t index = j;
            #else
            uint32_t index = ((occ * rook_pattern[i].magic) >> rook_pattern[i].shift);
            #endif

            rook_lookup[lookup_offset + index] = get_sliding_pattern(i, occ, true, false, true);
            rook_xray_lookup[lookup_offset + index] = get_sliding_xray_pattern(i, occ, true, false, true);
        }

        lookup_offset += lookups;
    }
}

void bitboard_utility::init_pawn_tables(bool black)
{
    for (int i = 0; i < 64; i++) {
        int posx = i % 8;
        int posy = i / 8;


        int attack[4];
        int advance[4];
        bool initial_pos;
        if (black) {
            advance[0] = posx;
            advance[1] = posy+1;

            advance[2] = posx;
            advance[3] = posy+2;

            attack[0] = posx+1;
            attack[1] = posy+1;

            attack[2] = posx-1;
            attack[3] = posy+1;

            initial_pos = (posy == 1);
        } else {
            advance[0] = posx;
            advance[1] = posy-1;

            advance[2] = posx;
            advance[3] = posy-2;

            attack[0] = posx+1;
            attack[1] = posy-1;

            attack[2] = posx-1;
            attack[3] = posy-1;

            initial_pos = (posy == 6);
        }

        bitboard attack_pattern = 0;
        bitboard advance_pattern = 0;

        for (int j = 0; j < 2; j++) {
            int px = attack[j*2+0];
            int py = attack[j*2+1];

            if (px < 0 || py < 0 || px >= BOARD_WIDTH || py >= BOARD_HEIGHT) {
                continue;
            }

            set_occupation(py*8+px, attack_pattern, true);
        }
        int advance_possibilies = (initial_pos ? 2 : 1);

        pawn_block_square[i][black] = i;

        for (int j = 0; j < advance_possibilies; j++) {
            int px = advance[j*2+0];
            int py = advance[j*2+1];


            if (px < 0 || py < 0 || px >= BOARD_WIDTH || py >= BOARD_HEIGHT) {
                continue;
            }

            if (j == 0) {
                pawn_block_square[i][black] = py*8+px;
            }

            set_occupation(py*8+px, advance_pattern, true);
        }

        pawn_advance_pattern[i][black] = advance_pattern;
        pawn_attack_pattern[i][black] = attack_pattern;
    }
}


void bitboard_utility::init_knight_tables()
{
    static const int8_t pattern[16] = {
        2,  1,
        2, -1,
        -2,  1,
        -2, -1,
        1,  2,
        -1,  2,
        1, -2,
        -1, -2
    };

    for (int i = 0; i < 64; i++) {
        int posx = i % 8;
        int posy = i / 8;

        bitboard bb = 0;
        for (int j = 0; j < 8; j++) {
            int nx = posx + pattern[j*2+0];
            int ny = posy + pattern[j*2+1];

            if (nx < 0 || ny < 0 || nx >= BOARD_WIDTH || ny >= BOARD_HEIGHT) {
                continue;
            }

            set_occupation(ny*8+nx, bb, true);
        }
        knight_pattern[i] = bb;
    }
}

void bitboard_utility::init_king_tables()
{
    static const int pattern[16] = {
        1,   0,
        -1,   0,
        0,   1,
        0,  -1,
        1,   1,
        1,  -1,
        -1,  1,
        -1, -1
    };

    for (int i = 0; i < 64; i++) {
        int posx = i % 8;
        int posy = i / 8;

        bitboard bb = 0;
        for (int j = 0; j < 8; j++) {
            int nx = posx + pattern[j*2+0];
            int ny = posy + pattern[j*2+1];

            if (nx < 0 || ny < 0 || nx >= BOARD_WIDTH || ny >= BOARD_HEIGHT) {
                continue;
            }

            set_occupation(ny*8+nx, bb, true);
        }
        king_pattern[i] = bb;
    }
}



bitboard_utility::bitboard_utility() {
    std::cout << "Initializing bitboard lookup tables... ";

    generate_sliding_tables();

    init_pawn_tables(false);
    init_pawn_tables(true);
    init_king_tables();
    init_knight_tables();
    init_bishop_lookup();
    init_rook_lookup();

    std::cout << "done." << std::endl;

    #if USE_PEXT

    std::cout << "Using PEXT bitboards" << std::endl;

    #else

    std::cout << "Using magic bitboards" << std::endl;

    #endif // PEXT
}

void print_bitboard(bitboard bb)
{
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            if (bitboard_utils.get_occupation(y*8+x, bb)) {
                std::cout << "1";
            } else {
                std::cout << "0";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}





