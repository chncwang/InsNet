#ifndef N3LDG_PLUS_MEMORY_POOL_H
#define N3LDG_PLUS_MEMORY_POOL_H

#include <vector>
#include <sstream>
#include <list>
#include <unordered_map>
#include <map>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include <iostream>
#include "fmt/core.h"

namespace n3ldg_plus {
namespace cuda {

struct MemoryBlock {
    void *p;
    int64_t size;
    void *buddy = nullptr;
    int id;

    MemoryBlock() {
        abort();
    }

    MemoryBlock(void *p, int size, void *buddy = nullptr) {
        static int global_id;
        if (size <= 0 || (size & (size - 1)) != 0) {
            std::cerr << "illegal size:" << size << std::endl;
            abort();
        }
        this->p = p;
        this->size = size;
        this->buddy = buddy;
        this->id = global_id++;
    }

    std::string toString() const {
        std::stringstream p_stream;
        p_stream << p;
        std::stringstream buddy_stream;
        buddy_stream << buddy;
        return fmt::format("p:{} size:{} buddy:{} id:{}", p_stream.str(), size, buddy_stream.str(),
                id);
    }
};

class MemoryPool {
public:
    MemoryPool(const MemoryPool &) = delete;
    static MemoryPool& Ins();

    void Malloc(void **p, int size);
    void Free(void *p);

    void Init(float size_in_gb) {
        std::cout << fmt::format("MemoryPool Init size:{}\n", size_in_gb);
        std::vector<void*> pointers;
        if (size_in_gb > 0.0f) {
            for (int i = 0; i < static_cast<int>(size_in_gb); ++i) {
                void *m = nullptr;
                Malloc(&m, (1 << 30));
                pointers.push_back(m);
            }
            for (void* m : pointers) {
                this->Free(m);
            }
        }
    }

    std::string toString() const {
        std::string free_block_str = "[";
        int i = 0;
        for (auto & v : free_blocks_) {
            std::string arr = "[";
            for (auto &it : v) {
                arr += it.second.toString() + ",";
            }
            arr += "]";
            if (!v.empty()) {
                std::string str = fmt::format("{i:{}, v:{}}", i, arr);
                free_block_str += str + ",";
            }
            i++;
        }
        free_block_str += "]";
        return free_block_str;
    }

private:
    MemoryPool() = default;
    std::vector<std::map<void*, MemoryBlock>> free_blocks_;
    std::unordered_map<void *, MemoryBlock> busy_blocks_;
};

}
}
#endif
