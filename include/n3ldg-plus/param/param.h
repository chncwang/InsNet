#ifndef N3LDG_PLUS_PARAM_H
#define N3LDG_PLUS_PARAM_H

#include "n3ldg-plus/param/base-param.h"
#include "fmt/core.h"

namespace n3ldg_plus {

enum InitDistribution {
    UNI = 0,
    NORM = 1
};

class Param : public BaseParam {
public:
    Param(const std::string &name, bool is_bias = false) : BaseParam(name, is_bias) {}

    virtual ~Param() = default;

    virtual void init(int outDim, int inDim) override {
        init(outDim, inDim, nullptr);
    }

    void init(int outDim, int inDim, const std::function<dtype(int, int)> *cal_bound,
            InitDistribution dist = InitDistribution::UNI);

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        auto v = BaseParam::transferablePtrs();
        v.push_back(&aux_square);
        v.push_back(&aux_mean);
        return v;
    }

    virtual std::string name() const {
        return "Param";
    }
#endif

    int outDim() override {
        return val_.row;
    }

    int inDim() override {
        return val_.col;
    }

    void clearGrad() override {
#if USE_GPU
        n3ldg_cuda::Memset(grad.value, grad.size, 0.0f);
#if TEST_CUDA
        grad.zero();
        n3ldg_cuda::Assert(grad.verify("Param clearGrad"));
#endif
#else
        grad_.zero();
#endif
    }

    void adagrad(dtype alpha, dtype reg, dtype eps) override;

    void adam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override;

    void adamW(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override;

    [[deprecated("Such util functions should not be a member method")]]
    void randpoint(int& idx, int &idy) override;

    dtype gradSquareSum() override;

    void rescaleGrad(dtype scale) override {
#if USE_GPU
        n3ldg_cuda::Rescale(grad.value, grad.size, scale);
#if TEST_CUDA
        grad.vec() = grad.vec() * scale;
        n3ldg_cuda::Assert(grad.verify("Param rescaleGrad"));
#endif
#else
        grad_.vec() = grad_.vec() * scale;
#endif
    }

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(val_, aux_square_, aux_mean_, iter_, is_fixed_);
    }

    [[deprecated]]
    void value(const int& featId, Tensor1D& out);

    [[deprecated]]
    void loss(const int& featId, const Tensor1D& loss);

private:
    Tensor2D aux_square_;
    Tensor2D aux_mean_;
    int iter_ = 0;
    bool is_fixed_ = false;
};

template<typename ParamType>
struct ParamArray : public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    ParamArray(const std::string &nam) : name(nam) {}

    std::vector<std::shared_ptr<ParamType>> params;
    std::string name;

    std::vector<ParamType *> ptrs() const {
        std::vector<ParamType *> results;
        for (auto &p : params) {
            results.push_back(p.get());
        }
        return results;
    }

    int size() const {
        return params.size();
    }

    void init(int layer, std::function<void(ParamType &, int)> &init_param) {
        for (int i = 0; i < layer; ++i) {
            std::shared_ptr<ParamType> param(new ParamType(name + std::to_string(i)));
            init_param(*param, i);
            params.push_back(param);
        }
    }

    template<typename Archive>
    void serialize(Archive &ar) {
        for (auto &param : params) {
            ar(*param);
        }
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        std::vector<n3ldg_cuda::Transferable *> results;
        for (auto &p : params) {
            results.push_back(p.get());
        }
        return results;
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam> *> tunableComponents() override {
        std::vector<Tunable<BaseParam> *> results;
        for (auto &p : params) {
            results.push_back(p.get());
        }
        return results;
    }
};

}

#endif
