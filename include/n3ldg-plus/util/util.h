#ifndef N3LDG_PLUS_UTIL_H
#define N3LDG_PLUS_UTIL_H

#include <fstream>
#include <string>

#include "n3ldg-plus/util/nrmat.h"
#include "n3ldg-plus/base/def.h"

namespace n3ldg_plus {

bool my_getline(std::ifstream &inf, std::string &line);

void split_bychar(const std::string& str, std::vector<std::string>& vec, char separator = ' ');

template <typename T, typename S>
std::vector<S *> toPointers(std::vector<T> &v, int size) {
    std::vector<S *> pointers;
    for (int i = 0; i < size; ++i) {
        pointers.push_back(&v.at(i));
    }
    return pointers;
}

template <typename T, typename S>
std::vector<S *> toPointers(std::vector<T> &v) {
    return toPointers<T, S>(v, v.size());
}

bool isEqual(dtype a, dtype b);

inline size_t typeSignature(void *p) {
    auto addr = reinterpret_cast<uintptr_t>(p);
#if SIZE_MAX < UINTPTR_MAX
    addr %= SIZE_MAX;
#endif
    return addr;
}

template <typename T, typename S>
std::vector<T> transferVector(const std::vector<S> &src_vector,
        const std::function<T(const S&)> &transfer) {
    std::vector<T> result;
    for (const S& src : src_vector) {
        result.push_back(transfer(src));
    }
    return result;
}

}

#endif
