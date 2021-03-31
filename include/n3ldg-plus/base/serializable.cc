#include "n3ldg-plus/base/serializable.h"

namespace n3ldg_plus {

#if USE_GPU
void TransferableComponents::copyFromHostToDevice() override {
    for (auto *t : transferablePtrs()) {
        t->copyFromHostToDevice();
    }
}

void TransferableComponents::copyFromDeviceToHost() override {
    for (auto *t : transferablePtrs()) {
        t->copyFromDeviceToHost();
    }
}
#endif

}
