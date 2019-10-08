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
typedef n3ldg_cuda::Tensor2D Tensor2D;
#else
typedef n3ldg_cpu::Tensor2D Tensor2D;
#endif

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

template <typename T>
class Tunable {
public:
    virtual std::vector<T *> tunableParams() = 0;
};

template <typename T>
class TunableAtom : public Tunable<T> {
public:
    std::vector<T *> tunableParams() override {
        return {static_cast<T*>(this)};
    }
};

template <typename T>
class TunableCombination : public Tunable<T> {
public:
    std::vector<T *> tunableParams() override {
        auto components = tunableComponents();
        std::vector<T*> result;
        for (Tunable<T> * t : components) {
            auto params = t->tunableParams();
            for (auto *p : params) {
                result.push_back(p);
            }
        }
        return result;
    }

protected:
    virtual std::vector<Tunable<T> *> tunableComponents() = 0;
};

class BaseParam : public N3LDGSerializable, public TunableAtom<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    Tensor2D val;
    Tensor2D grad;

    BaseParam(const std::string &name, bool is_bias = false) : is_bias_(is_bias), name_(name) {}

    bool isBias() const {
        return is_bias_;
    }

    virtual void init(int outDim, int inDim) = 0;
    virtual void updateAdagrad(dtype alpha, dtype reg, dtype eps) = 0;
    virtual void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) = 0;
    virtual void updateAdamW(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) = 0;
    virtual int outDim() = 0;
    virtual int inDim() = 0;
    virtual void clearGrad() = 0;
    virtual const std::string& getParamName() const {
        return name_;
    }

    // Choose one point randomly
    virtual void randpoint(int& idx, int &idy) = 0;
    virtual dtype squareGradNorm() = 0;
    virtual void rescaleGrad(dtype scale) = 0;

#if USE_GPU
    virtual std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&val};
    }
#endif

private:
    bool is_bias_ = false;
    std::string name_;
};

#endif /* BasePARAM_H_ */
