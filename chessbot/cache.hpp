#pragma once

#include <stdint.h>
#include <iostream>

#define USE_HUGEPAGES 1

#if USE_HUGEPAGES==1

#include <sys/mman.h>

#endif // USE_HUGEPAGES



template <typename T, size_t N>
class cache
{
public:
    cache() {
        size = MB_to_size(N);

        allocate_data(size, true);
    }
    ~cache() {
        free_data();
    }

    cache(const cache<T, N> &other) {
        size = other.size;

        allocate_data(size, false);

        for (uint64_t i = 0; i < size; i++) {
            new (&data[i])T(other.data[i]);
        }
    }

    cache<T,N>& operator = (cache<T, N> other) {
        std::swap(other.data, data);
        std::swap(other.size, size);
        std::swap(other.memory, memory);

        return *this;
    }

    void resize(int size_MB) {
        free_data();

        size = MB_to_size(size_MB);
        allocate_data(size, true);
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

    void allocate_data(int s, bool construct)
    {
        allocation_size = s*sizeof(T) + 64;


        #if USE_HUGEPAGES==1

        memory = mmap(NULL, allocation_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (allocation_size > 4*1024*1024) {
            madvise(memory, allocation_size, MADV_HUGEPAGE);
        }

        #else

        memory = static_cast<void*>(new uint8_t[allocation_size]);

        #endif

        data = (T*)((size_t)memory + 64 - ((size_t)memory % 64));

        if (construct) {
            for (int i = 0; i < s; i++) {
                new (&data[i])T();
            }
        }
    }

    void free_data()
    {
        for (int i = 0; i < size; i++) {
            data[i].~T();
        }

        #if USE_HUGEPAGES==1

        munmap(memory, allocation_size);

        #else

        delete [] memory;

        #endif // USE_HUGEPAGES
    }

    uint64_t size_to_MB(uint64_t s) const {
        return ((s * sizeof(T)) / (1024*1024));
    }
    uint64_t MB_to_size(uint64_t s) const {
        return  ((s * 1024 * 1024) / sizeof(T));
    }

    T *data;
    uint64_t size;

private:
    void *memory;

    uint64_t allocation_size;
};
