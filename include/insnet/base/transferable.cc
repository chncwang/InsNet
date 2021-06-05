#include "transferable.h"

namespace insnet {
namespace cuda {

#if USE_GPU
void TransferableComponents::copyFromHostToDevice() {
    for (auto *t : transferablePtrs()) {
        t->copyFromHostToDevice();
    }
}

void TransferableComponents::copyFromDeviceToHost() {
    for (auto *t : transferablePtrs()) {
        t->copyFromDeviceToHost();
    }
}
#endif

}
}
