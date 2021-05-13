#ifndef BasePARAM_H_
#define BasePARAM_H_

#include "n3ldg-plus/base/tensor.h"

namespace n3ldg_plus {

#if USE_GPU
typedef cuda::Tensor2D Tensor2D;
#else
typedef cpu::Tensor2D Tensor2D;
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
        if (tunable_list_ == nullptr) {
            auto components = tunableComponents();
            std::vector<T*> result;
            for (Tunable<T> * t : components) {
                auto params = t->tunableParams();
                for (auto *p : params) {
                    result.push_back(p);
                }
            }
            tunable_list_ = std::make_unique<std::vector<T*>>(std::move(result));
        }
        return *tunable_list_;
    }

protected:
    virtual std::vector<Tunable<T> *> tunableComponents() = 0;

private:
    mutable std::unique_ptr<std::vector<T *>> tunable_list_ = nullptr;
};

class BaseParam : public TunableAtom<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
public:
    BaseParam(const std::string &name, bool is_bias = false) : is_bias_(is_bias), name_(name) {}

    BaseParam(bool is_bias = false) : is_bias_(is_bias) {}

    bool isBias() const {
        return is_bias_;
    }

    virtual void init(int outDim, int inDim) = 0;
    virtual void adagrad(dtype alpha, dtype reg, dtype eps) = 0;
    virtual void adam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) = 0;
    virtual void adamW(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) = 0;
    virtual int outDim() = 0;
    virtual int inDim() = 0;
    virtual const std::string& getParamName() const {
        return name_;
    }

    [[deprecated]]
    virtual void randpoint(int& idx, int &idy) = 0;

    virtual dtype gradSquareSum() = 0;

    virtual void rescaleGrad(dtype scale) = 0;

#if USE_GPU
    virtual std::vector<cuda::Transferable *> transferablePtrs() override {
        return {&val_};
    }
#endif

    Tensor2D &val() {
        return val_;
    }

    Tensor2D &grad() {
        return *grad_;
    }

    virtual void initAndZeroGrad();

    void releaseGrad() {
        grad_.reset();
    }

protected:
    bool is_bias_ = false;
    std::string name_;
    Tensor2D val_, aux_square_, aux_mean_;
    std::unique_ptr<Tensor2D> grad_ = nullptr;
};

typedef Tunable<BaseParam> TunableParam;
typedef TunableCombination<BaseParam> TunableParamCollection;

}

#endif /* BasePARAM_H_ */
