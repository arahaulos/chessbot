#include "bitboard.hpp"
#include "state.hpp"
#include <iostream>
#include <math.h>


int sliding_vectors_table[8*2] =
{
    1, 1, 1, -1, -1, 1, -1, -1, 1, 0, -1, 0, 0, 1, 0, -1
};

int distance_to_center[64] =
{
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 3, 3, 3, 3, 3, 3, 4,
    4, 3, 2, 2, 2, 2, 3, 4,
    4, 3, 2, 1, 1, 2, 3, 4,
    4, 3, 2, 1, 1, 2, 3, 4,
    4, 3, 2, 2, 2, 2, 3, 4,
    4, 3, 3, 3, 3, 3, 3, 4,
    4, 4, 4, 4, 4, 4, 4, 4
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

        bishop_pattern0[i] = pattern;
        bishop_lookup_offset[i] = lookup_offset;


        int bits = __builtin_popcountll(pattern);
        int lookups = std::pow(2, bits);

        for (int j = 0; j < lookups; j++) {
            bitboard occ = _pdep_u64((uint64_t)j, pattern);
            bishop_lookup[lookup_offset + j] = get_sliding_pattern(i, occ, false, true, true);
            bishop_xray_lookup[lookup_offset + j] = get_sliding_xray_pattern(i, occ, false, true, true);
        }

        lookup_offset += lookups;
    }
}

void bitboard_utility::init_rook_lookup() {

    uint32_t lookup_offset = 0;

    for (int i = 0; i < 64; i++) {
        bitboard pattern = get_sliding_pattern(i, 0, true, false, false);

        rook_pattern0[i] = pattern;
        rook_lookup_offset[i] = lookup_offset;


        int bits = __builtin_popcountll(pattern);
        int lookups = std::pow(2, bits);

        for (int j = 0; j < lookups; j++) {
            bitboard occ = _pdep_u64((uint64_t)j, pattern);
            rook_lookup[lookup_offset + j] = get_sliding_pattern(i, occ, true, false, true);
            rook_xray_lookup[lookup_offset + j] = get_sliding_xray_pattern(i, occ, true, false, true);
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


void bitboard_utility::init_file_rank_lookups()
{
    for (int i = 0; i < 8; i++) {
        bitboard f = 0;
        bitboard r = 0;
        for (int j = 0; j < 8; j++) {
            set_occupation(i*8+j, r, true);
            set_occupation(j*8+i, f, true);
        }

        bitboard_files[i] = f;
        bitboard_ranks[i] = r;
    }


    for (int i = 0; i < 8; i++) {
        bitboard neightbor_files = 0;
        if (i > 0) {
            neightbor_files |= bitboard_files[i-1];
        }
        if (i < 7) {
            neightbor_files |= bitboard_files[i+1];
        }
        bitboard_neighbor_files[i] = neightbor_files;
    }

}

void bitboard_utility::init_passed_pawn_lookups()
{
    for (int i = 0; i < 64; i++) {
        int sx = i % 8;
        int sy = i / 8;


        bitboard bb = 0;
        bitboard wb = 0;
        for (int y = sy+1; y < 8; y++) {
            for (int x = sx-1; x <= sx+1; x++) {
                if (x >= 0 && y >= 0 && x < 8 && y < 8) {
                    set_occupation(y*8+x, bb, true);
                }
            }
        }
        for (int y = sy-1; y >= 0; y--) {
            for (int x = sx-1; x <= sx+1; x++) {
                if (x >= 0 && y >= 0 && x < 8 && y < 8) {
                    set_occupation(y*8+x, wb, true);
                }
            }
        }

        passed_pawn_mask[i][0] = wb;
        passed_pawn_mask[i][1] = bb;
    }
}


void bitboard_utility::init_board_area_masks()
{
    center_board_mask = 0;
    extended_center_board_mask = 0;

    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int sq = y*8+x;

            if (y >= 2 && y <= 5) {
                set_occupation(sq, extended_center_board_mask, true);
            }
            if (y >= 3 && y <= 4) {
                set_occupation(sq, center_board_mask, true);
            }
        }
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
    init_file_rank_lookups();
    init_passed_pawn_lookups();
    init_board_area_masks();

    std::cout << "done." << std::endl;
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





