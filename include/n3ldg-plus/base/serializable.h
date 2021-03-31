#ifndef N3LDG_INCLUDE_SERIALIZABLE_H
#define N3LDG_INCLUDE_SERIALIZABLE_H

namespace n3ldg_plus {

#if USE_GPU
class TransferableComponents : public n3ldg_cuda::Transferable
{
public:
    void copyFromHostToDevice() override;
    void copyFromDeviceToHost() override;
    virtual std::vector<n3ldg_cuda::Transferable *> transferablePtrs() = 0;
};
#endif

}

#endif
