#ifndef N3LDG_PLUS_LAYER_NORMALIZATION_H
#define N3LDG_PLUS_LAYER_NORMALIZATION_H

#include "n3ldg-plus/operator/linear.h"

namespace n3ldg_plus {

class LayerNormParams : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents 
#endif
{
public:
    LayerNormParams(const std::string &name) : g_(name + "-g"), b_(name + "-b") {}

    void init(int dim) {
        g_.init(dim, 1);
        g_.val().assignAll(1);
        b_.initAsBias(dim);
    }

    Param &g() {
        return g_;
    }

    BiasParam &b() {
        return b_;
    }

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(g_, b_);
    }

#if USE_GPU
    std::vector<cuda::Transferable *> transferablePtrs() override {
        return {&g_, &b_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&g_, &b_};
    }

private:
    Param g_;
    BiasParam b_;
};

Node *layerNorm(Node &input, int row);

Node *layerNorm(Node &input, LayerNormParams &params);

Node *affine(Node &input, LayerNormParams &params);

}

#endif
