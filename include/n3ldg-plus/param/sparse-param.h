#ifndef N3LDG_PLUS_SPARSE_PARAM_H
#define N3LDG_PLUS_SPARSE_PARAM_H

#include "n3ldg-plus/param/base-param.h"
#include "n3ldg-plus/util/nrmat.h"
#include "n3ldg-plus/cuda/n3ldg_plus_cuda.h"

namespace n3ldg_plus {

class SparseParam : public BaseParam {
public:
    SparseParam(const std::string &name = "sparse") : BaseParam(name) {}

    ~SparseParam();

#if USE_GPU
    void copyFromHostToDevice() override;

    void copyFromDeviceToHost() override;
#endif

    void init(int outDim, int inDim) override;

    void clearGrad() override;

    int outDim() override {
        return val_.row;
    }

    int inDim() override {
        return val_.col;
    }

    void adagrad(dtype alpha, dtype reg, dtype eps) override;

    void adam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override;

    void adamW(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override {
        adam(belta1, belta2, alpha, reg, eps); // TODO
    }

    [[deprecated]]
    void randpoint(int& idx, int &idy) override;

    dtype gradSquareSum() override;

    void rescaleGrad(dtype scale) override;

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(val_, aux_square_, aux_mean_);
    }

#if USE_GPU
    cuda::BoolArray *dIndexers = nullptr;
    cuda::IntArray *dIters = nullptr;
#endif

private:
    nr::NRVec<bool> indexers;
    nr::NRVec<int> last_update;
};

}

#endif
