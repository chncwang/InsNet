#ifndef N3LDG_PLUS_TRANSFERABLE_H
#define N3LDG_PLUS_TRANSFERABLE_H

#include <vector>

namespace n3ldg_plus {
namespace cuda {

#if USE_GPU

class Transferable {
public:
    virtual void copyFromHostToDevice() = 0;
    virtual void copyFromDeviceToHost() = 0;
};

class TransferableComponents : public Transferable
{
public:
    void copyFromHostToDevice() override;
    void copyFromDeviceToHost() override;
    virtual ::std::vector<Transferable *> transferablePtrs() = 0;
};

#endif

}
}

#endif
