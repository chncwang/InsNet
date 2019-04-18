/*
 * BaseParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef BasePARAM_H_
#define BasePARAM_H_

#include "boost/format.hpp"

#if USE_GPU
#include "N3LDG_cuda.h"
#endif

#include "MyTensor.h"

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
//            std::cout << boost::format("name:%1%") % t->name() << std::endl;
            t->copyFromDeviceToHost();
        }
    }

    virtual std::vector<n3ldg_cuda::Transferable *> transferablePtrs() = 0;
};
#endif

class BaseParam : public N3LDGSerializable
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    Tensor2D val;
    Tensor2D grad;

    BaseParam() = default;

    BaseParam(bool is_bias) : is_bias_(is_bias) {}

    bool isBias() const {
        return is_bias_;
    }

    virtual void init(int outDim, int inDim) = 0;
    virtual void updateAdagrad(dtype alpha, dtype reg, dtype eps) = 0;
    virtual void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) = 0;
    virtual int outDim() = 0;
    virtual int inDim() = 0;
    virtual void clearGrad() = 0;

    // Choose one point randomly
    virtual void randpoint(int& idx, int &idy) = 0;
    virtual dtype squareGradNorm() = 0;
    virtual void rescaleGrad(dtype scale) = 0;
#if USE_GPU
    virtual std::vector<n3ldg_cuda::Transferable *> transferablePtrs() {
        return {&val};
    }
#endif

private:
    bool is_bias_ = false;
};

#endif /* BasePARAM_H_ */
