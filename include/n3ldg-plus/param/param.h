#ifndef N3LDG_PLUS_PARAM_H
#define N3LDG_PLUS_PARAM_H

#include "n3ldg-plus/param/base-param.h"
#include "n3ldg-plus/base/transferable.h"
#include "fmt/core.h"

namespace n3ldg_plus {

enum InitDistribution {
    UNI = 0,
    NORM = 1
};

class Param : public BaseParam {
public:
    Param(const std::string &name, bool is_bias = false) : BaseParam(name, is_bias) {}

    Param(bool is_bias = false) : BaseParam(is_bias) {}

    virtual ~Param() = default;

    virtual void init(int outDim, int inDim) override {
        init(outDim, inDim, nullptr);
    }

    void init(int outDim, int inDim, const std::function<dtype(int, int)> *cal_bound,
            InitDistribution dist = InitDistribution::UNI);

#if USE_GPU
    std::vector<cuda::Transferable *> transferablePtrs() override {
        auto v = BaseParam::transferablePtrs();
        v.push_back(&aux_mean_);
        v.push_back(&aux_square_);
        return v;
    }
#endif

    int outDim() override {
        return val_.row;
    }

    int inDim() override {
        return val_.col;
    }

    void clearGrad() override;

    void adagrad(dtype alpha, dtype reg, dtype eps) override;

    void adam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override;

    void adamW(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override;

    [[deprecated("Such util functions should not be a member method")]]
    void randpoint(int& idx, int &idy) override;

    dtype gradSquareSum() override;

    void rescaleGrad(dtype scale) override;

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(val_, aux_square_, aux_mean_, iter_);
    }

private:
    int iter_ = 0;
};

template<typename ParamType>
struct ParamArray : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
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
    std::vector<cuda::Transferable *> transferablePtrs() override {
        std::vector<cuda::Transferable *> results;
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
