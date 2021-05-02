#include "n3ldg-plus/base/memory.h"
#include "n3ldg-plus/cuda/memory_pool.h"
#include <cstdlib>
#include <iostream>
#include "fmt/core.h"

using std::cout;
using std::cerr;
using std::endl;
using std::shared_ptr;
using std::make_shared;

namespace n3ldg_plus {

void MemoryContainer::init(int size_in_bytes) {
    if (addr_ != nullptr) {
        cerr << "MemoryContainer::init - addr_ is not nullptr" << endl;
        abort();
    }
    if (offset_ != 0) {
        cerr << fmt::format("MemoryContainer::init - offset_:{}", offset_) << endl;
        abort();
    }
    size_in_bytes_ = size_in_bytes;
    initMemory();
}

void *MemoryContainer::allocate(int size_in_bytes) {
    if (offset_ + size_in_bytes > size_in_bytes_) {
        cerr << fmt::format("MemoryContainer::allocate - offset_:{} size_in_bytes:{} size_in_bytes_:{}",
                offset_, size_in_bytes, size_in_bytes_) << endl;
        abort();
    }

    void *ret = static_cast<char *>(addr_) + offset_;
    offset_ += size_in_bytes;
    return ret;
}

class CPUMemoryContainer : public MemoryContainer {
public:
    CPUMemoryContainer() = default;

    ~CPUMemoryContainer() override {
        free(addr_);
    }

protected:
    void initMemory() override {
        addr_ = malloc(size_in_bytes_);
    }
};

#if USE_GPU
class GPUMemoryContainer : public MemoryContainer {
public:
    GPUMemoryContainer() = default;

    ~GPUMemoryContainer() override {
        cuda::MemoryPool &ins = cuda::MemoryPool::Ins();
        ins.Free(addr_);
    }

protected:
    void initMemory() override {
        cuda::MemoryPool &ins = cuda::MemoryPool::Ins();
        ins.Malloc(&addr_, size_in_bytes_);
    }
};
#endif

shared_ptr<MemoryContainer> memoryContainer(int size_in_bytes) {
    shared_ptr<MemoryContainer> ret;
#if USE_GPU
    ret = make_shared<GPUMemoryContainer>();
#else
    ret = make_shared<CPUMemoryContainer>();
#endif
    ret->init(size_in_bytes);
    return ret;
}

}
