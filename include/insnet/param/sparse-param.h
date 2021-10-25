#ifndef INSNET_SPARSE_PARAM_H
#define INSNET_SPARSE_PARAM_H

#include "insnet/param/base-param.h"
#include "insnet/util/nrmat.h"
#include "insnet/cuda/insnet_cuda.h"

namespace insnet {

class SparseParam : public BaseParam { // TODO Don't use it currently, bugs remained.
public:
    SparseParam(const std::string &name = "sparse") : BaseParam(name) {}

    ~SparseParam();

#if USE_GPU
    void copyFromHostToDevice() override;

    void copyFromDeviceToHost() override;
#endif

    void init(int outDim, int inDim) override;

    bool initAndZeroGrad() override;

    void adagrad(dtype alpha, dtype reg, dtype eps) override;

    void adam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override;

    void adamW(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override {
        adam(belta1, belta2, alpha, reg, eps); // TODO
    }

    [[deprecated]]
    void randpoint(int& idx, int &idy) override;

    dtype gradSquareSum() override;

    void rescaleGrad(dtype scale) override;

    bool isSparse() override {
        return true;
    }

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(val_, aux_square_, aux_mean_);
    }

#if USE_GPU
    cuda::BoolArray *dIndexers = nullptr;
    cuda::IntArray *dIters = nullptr;
#endif

    nr::NRVec<bool> indexers;
    nr::NRVec<int> last_update; // TODO historical code which should be modified to use STL instead.
};

}

#endif
