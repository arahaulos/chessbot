#include "huffman.hpp"
#include <array>
#include <vector>

struct tree_node
{
    uint32_t frequency;
    int left;
    int right;
    int parent;
};

std::vector<tree_node> construct_tree(const std::array<uint32_t, 0x100> &frequencies)
{
    std::vector<tree_node> nodes;
    for (int i = 0; i < 0x100; i++) {
        nodes.push_back( {frequencies[i], -1, -1, -1} );
    }
    while (true) {
        int least1 = -1;
        int least0 = -1;
        for (size_t i = 0; i < nodes.size(); i++) {
            if (nodes[i].parent == -1 && (least0 < 0 || nodes[i].frequency < nodes[least0].frequency)) {
                least1 = least0;
                least0 = i;
            } else if (nodes[i].parent == -1 && (least1 < 0 || nodes[i].frequency < nodes[least1].frequency)) {
                least1 = i;
            }
        }
        if (least0 >= 0 && least1 >= 0) {
            nodes[least0].parent = nodes.size();
            nodes[least1].parent = nodes.size();
            nodes.push_back( {nodes[least0].frequency + nodes[least1].frequency, least0, least1, -1} );
        } else {
            break;
        }
    }
    return nodes;
}


void huffman_coder::encode(uint8_t *input, int input_size, uint8_t *&output, int &output_size)
{
    std::array<uint32_t, 0x100> frequencies = {};
    std::fill(frequencies.begin(), frequencies.end(), 0);
    for (int i = 0; i < input_size; i++) {
        frequencies[input[i]] += 1;
    }

    std::vector<tree_node> nodes = construct_tree(frequencies);
    std::array<std::pair<int, int>, 0x100> codes;

    output_size = 0;
    for (int i = 0; i < 0x100; i++) {
        int bits = 0;
        int code = 0;

        int node = i;
        int parent = nodes[i].parent;
        while (parent != -1) {
            code <<= 1;
            if (node == nodes[parent].right) {
                code |= 0x1;
            }
            bits++;
            node = parent;
            parent = nodes[node].parent;
        }

        codes[i].first = bits;
        codes[i].second = code;

        output_size += nodes[i].frequency*bits;
    }

    output_size = 0x100*sizeof(uint32_t) + (output_size / 8) + 1;
    output = new uint8_t[output_size];

    uint32_t *freq_table = reinterpret_cast<uint32_t*>(output);
    for (int i = 0; i < 0x100; i++) {
        freq_table[i] = frequencies[i];
    }

    int write_index = 0x100*sizeof(uint32_t);
    uint8_t bit_index = 0;
    uint8_t byte = 0;
    for (int i = 0; i < input_size; i++) {
        int bits = codes[input[i]].first;
        int code = codes[input[i]].second;

        while (bits > 0) {
            uint8_t bit = (code & 0x1);
            code >>= 1;

            byte |= (bit << bit_index);

            bit_index++;
            if (bit_index > 7) {
                output[write_index++] = byte;
                byte = 0;
                bit_index = 0;
            }
            bits--;
        }
    }
    output[write_index++] = byte;
}

void huffman_coder::decode(uint8_t *input, int input_size, uint8_t *&output, int &output_size)
{
    constexpr uint32_t symbol_lookup_bits = 12;
    constexpr uint32_t symbol_lookup_size = 0x1 << symbol_lookup_bits;
    constexpr uint32_t symbol_lookup_mask = symbol_lookup_size-1;

    output_size = 0;

    uint32_t *freq_table = reinterpret_cast<uint32_t*>(input);
    std::array<uint32_t, 0x100> frequencies = {};
    for (int i = 0; i < 0x100; i++) {
        frequencies[i] = freq_table[i];
        output_size += freq_table[i];
    }

    output = new uint8_t[output_size];

    std::vector<tree_node> nodes = construct_tree(frequencies);

    std::array<std::pair<uint8_t, uint8_t>, symbol_lookup_size> symbols;
    for (uint32_t i = 0; i < symbol_lookup_size; i++) {
        bool found = false;
        int node = nodes.size()-1;
        for (uint8_t j = 0; j < symbol_lookup_bits; j++) {
            if ((i >> j) & 0x1) {
                node = nodes[node].right;
            } else {
                node = nodes[node].left;
            }
            if (node < 0x100) {
                symbols[i].first = node;
                symbols[i].second = j+1;
                found = true;
                break;
            }
        }

        if (!found) {
            symbols[i].second = 0;
        }
    }

    uint32_t read_index = 0x800*sizeof(uint32_t);
    uint32_t write_index = 0;

    auto decode_next_symbol = [&]() { //Slow decoding for edge cases
        int node = nodes.size()-1;
        while (read_index < input_size*8) {
            uint8_t byte = input[read_index >> 3];
            if ((byte >> (read_index & 0x7)) & 0x1) {
                node = nodes[node].right;
            } else {
                node = nodes[node].left;
            }
            read_index += 1;

            if (node < 0x100) {
                if (write_index < output_size) {
                    output[write_index++] = node;
                }
                break;
            }
        }
    };
    while (write_index < output_size) {
        uint32_t byte_index = read_index >> 3;
        uint32_t bit_index = read_index & 0x7;
        uint32_t data;

        if (symbol_lookup_bits > 9) {
            data = ((input[byte_index + 0]) | (input[byte_index + 1] << 8) | (input[byte_index + 2] << 16)) >> bit_index;
        } else {
            data = ((input[byte_index + 0]) | (input[byte_index + 1] << 8)) >> bit_index;
        }

        uint8_t symbol = symbols[data & symbol_lookup_mask].first;
        uint8_t bits = symbols[data & symbol_lookup_mask].second;

        if (bits != 0) {
            output[write_index++] = symbol;
            read_index += bits;
        } else {
            decode_next_symbol(); //Code is longer than lookup bits. lets decode manually
        }
        if ((read_index >> 3) + 2 >= input_size) {
            break;
        }
    }
    while (read_index < input_size*8) {
        decode_next_symbol();
    }
}



