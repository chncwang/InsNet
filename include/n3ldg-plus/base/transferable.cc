#include "transferable.h"

namespace n3ldg_plus {
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
