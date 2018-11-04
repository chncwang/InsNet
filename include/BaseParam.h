/*
 * BaseParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef BasePARAM_H_
#define BasePARAM_H_

#if USE_GPU
#include "N3LDG_cuda.h"
#endif

#include "MyTensor.h"

struct BaseParam
#if USE_GPU
: public n3ldg_cuda::Transferable
#endif
{
    Tensor2D val;
    Tensor2D grad;
    int index;

    BaseParam() {
        static int s_index;
        index = s_index++;
    }

    virtual void initial(int outDim, int inDim) = 0;
    virtual void updateAdagrad(dtype alpha, dtype reg, dtype eps) = 0;
    virtual void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) = 0;
    virtual int outDim() = 0;
    virtual int inDim() = 0;
    virtual void clearGrad() = 0;

    // Choose one point randomly
    virtual void randpoint(int& idx, int &idy) = 0;
    virtual dtype squareGradNorm() = 0;
    virtual void rescaleGrad(dtype scale) = 0;
    virtual void save(std::ofstream &os)const = 0;
    virtual void load(std::ifstream &is) = 0;
#if USE_GPU
    virtual void copyFromHostToDevice() override {
        val.copyFromHostToDevice();
        grad.copyFromHostToDevice();
    }
    virtual void copyFromDeviceToHost() override {
        val.copyFromDeviceToHost();
        grad.copyFromDeviceToHost();
    }
#endif
};

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

#endif /* BasePARAM_H_ */
