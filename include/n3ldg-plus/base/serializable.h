#ifndef N3LDG_INCLUDE_SERIALIZABLE_H
#define N3LDG_INCLUDE_SERIALIZABLE_H

#include <iostream>
#include "n3ldg-plus/base/tensor.h"
#include "cereal/cereal.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal/types/unordered_map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"

namespace n3ldg_plus {

#if USE_GPU
class TransferableComponents : public n3ldg_cuda::Transferable
{
public:
    void copyFromHostToDevice() override {
        for (auto *t : transferablePtrs()) {
            t->copyFromHostToDevice();
        }
    }

    void copyFromDeviceToHost() override {
        for (auto *t : transferablePtrs()) {
            t->copyFromDeviceToHost();
        }
    }

    virtual std::vector<n3ldg_cuda::Transferable *> transferablePtrs() = 0;
};
#endif

}

#endif
