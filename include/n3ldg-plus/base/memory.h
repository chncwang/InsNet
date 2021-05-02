#ifndef N3LDG_PLUS_MEMORY_H
#define N3LDG_PLUS_MEMORY_H

#include <memory>

namespace n3ldg_plus {

class MemoryContainer {
public:
    MemoryContainer() = default;

    void init(int size_in_bytes);

    virtual ~MemoryContainer() = default;

    void *allocate(int size_in_bytes);

protected:
    virtual void initMemory() = 0;

    void *addr_ = nullptr;
    int size_in_bytes_;

private:
    int offset_ = 0;
};

std::shared_ptr<MemoryContainer> memoryContainer(int size_in_bytes);

}

#endif
