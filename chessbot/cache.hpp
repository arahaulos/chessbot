#pragma once

#include <stdint.h>
#include <iostream>


template <typename T, size_t N>
class cache
{
public:
    cache() {
        size = MB_to_size(N);
        data = new T[size];
    }
    ~cache() {
        delete [] data;
    }

    cache(const cache<T, N> &other) {
        size = other.size;
        data = new T[size];
        for (uint64_t i = 0; i < size; i++) {
            data[i] = other.data[i];
        }
    }

    cache<T,N>& operator = (cache<T, N> other) {
        std::swap(other.data, data);
        std::swap(other.size, size);

        return *this;
    }

    void resize(int size_MB) {
        delete [] data;
        size = MB_to_size(size_MB);
        data = new T[size];
    }


    inline void prefetch(const uint64_t &key) {
        __builtin_prefetch(&data[key % size]);
    }

    T& operator [] (const uint64_t &key) {
        return data[key % size];
    }

    uint64_t get_size() {
        return size;
    }

    uint64_t get_size_MB() {
        return size_to_MB(size);
    }
private:
    uint64_t size_to_MB(uint64_t s) const {
        return ((s * sizeof(T)) / (1024*1024));
    }
    uint64_t MB_to_size(uint64_t s) const {
        return  ((s * 1024 * 1024) / sizeof(T));
    }


    T *data;
    uint64_t size;
};
