#include "zobrist.hpp"
#include <stdlib.h>
#include <bitset>

constexpr uint64_t seed = 123456789123456789;


uint64_t zobrist::rand_bitstring()
{
    rand ^= rand >> 12;
    rand ^= rand << 25;
    rand ^= rand >> 27;
    return rand * 0x2545F4914F6CDD1DULL;
}


zobrist::zobrist()
{
    rand = seed;
    for (int i = 0; i < 20; i++) {
        rand_bitstring();
    }

    turn_xor = rand_bitstring();

    for (int i = 0; i < BOARD_HEIGHT*BOARD_WIDTH; i++) {
        for (int j = 0; j < 16; j++) {
            table_xor[i][j] = rand_bitstring();
        }
        piece p0(WHITE, EMPTY);
        piece p1(BLACK, EMPTY);
        table_xor[i][p0.d] = 0;
        table_xor[i][p1.d] = 0;
    }

    for (int i = 0; i < BOARD_WIDTH; i++) {
        en_passant_xor[i] = rand_bitstring();
    }

    for (int i = 0; i < 4; i++) {
        castling_xor[i] = rand_bitstring();
    }
}

zobrist hashgen;
