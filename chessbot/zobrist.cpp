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
    std::cout << "Initializing zobrist hashing tables... ";

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
        singular_search_promotion[i] = rand_bitstring();
    }

    for (int i = 0; i < 4; i++) {
        castling_xor[i] = rand_bitstring();
    }

    for (int i = 0; i < 64; i++) {
        singular_search_hashing_from[i] = rand_bitstring();
        singular_search_hashing_to[i] = rand_bitstring();
    }

    piece minors[] = {piece(WHITE, KNIGHT), piece(BLACK, KNIGHT), piece(WHITE, BISHOP), piece(BLACK, BISHOP)};
    piece majors[] = {piece(WHITE, ROOK),   piece(BLACK, ROOK),   piece(WHITE, QUEEN),  piece(BLACK, QUEEN)};
    piece pawns[] =  {piece(WHITE, PAWN),   piece(BLACK, PAWN)};

    piece kings[] = {piece(WHITE, KING),   piece(BLACK, KING)};

    for (int i = 0; i < BOARD_HEIGHT*BOARD_WIDTH; i++) {
        for (int j = 0; j < 16; j++) {
            structure_table_xor[i][j] = 0;
        }
    }

    for (int i = 0; i < BOARD_HEIGHT*BOARD_WIDTH; i++) {
        for (int j = 0; j < 4; j++) {
            uint64_t r = rand_bitstring();

            structure_table_xor[i][minors[j].d] |= (r & MINOR_STRUCTURE_HASH_MASK);
            structure_table_xor[i][majors[j].d] |= (r & MAJOR_STRUCTURE_HASH_MASK);
        }
        for (int j = 0; j < 2; j++) {
            uint64_t r = rand_bitstring();

            structure_table_xor[i][pawns[j].d] |= (r & PAWN_STRUCTURE_HASH_MASK);
        }
        for (int j = 0; j < 2; j++) {
            uint64_t r = rand_bitstring();

            structure_table_xor[i][kings[j].d] |= (r & (MINOR_STRUCTURE_HASH_MASK | MAJOR_STRUCTURE_HASH_MASK));
        }
    }

    std::cout << "done." << std::endl;
}

zobrist hashgen;
