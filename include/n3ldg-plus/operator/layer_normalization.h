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

/// \ingroup operator
/// The row-wise layer normalization with the specified row number.
///
/// For Example, layerNorm([1.1, 0.9, -1.2, -0.8], 2) will return [1, -1, -1, 1].
///
/// **The operators with the equal row will be executed in batch.**
/// For example, layerNorm([0.1, 0.2, 0.3, 0.4], 2) and layerNorm([0.1, 0.2], 2) will be executed in batch, but layerNorm([0.1, 0.2, 0.3, 0.4], 2) and layerNorm([0.1, 0.2, 0.3, 0.4], 4) will not.
/// \param input The input tensor.
/// \param row The row number.
/// \return The normalized tensor with the mean value of 0 and the standard deviation of 1. Its size is equal to input.size().
Node *layerNorm(Node &input, int row);

/// \ingroup operator
/// The row-wise layer normalization with the parameters of the subsequent *affine* transformation.
///
/// For Example, supposing params.g() is [0.1, -0.1] and params.b() is [0, 0] layerNorm([1.1, 0.9, -1.2, -0.8], params) will return [0.1, 0.1, -0.1, -0.1].
///
/// **The operators with the same parameters will be executed in batch.** 
/// \param input The input tensor.
/// \param params g and b.
/// \return The affine transformed normalized tensor. Its size is equal to input.size().
Node *layerNorm(Node &input, LayerNormParams &params);

/// \ingroup operator
/// The affine transformation in layer normalization. \f$[{x_0}{g_0}, {x_1}{g_1}, ..., {x_n}{g_n}] + [b_0, b_1, ..., b_n]\f$
///
/// Following the Pytorch documentation's *LayerNorm*, we call this affine transformation, but it should actually be a simplified version. For Example, supposing params.g() is [0.1, -0.1] and params.b() is [0, 0] affine([1, -1, -1, 1], params) will return [0.1, 0.1, -0.1, -0.1].
///
/// **The operators with the same parameters will be executed in batch.** 
/// \param input The input tensor.
/// \param params g and b.
/// \return The affine transformed tensor. Its size is equal to input.size().
Node *affine(Node &input, LayerNormParams &params);

}

#endif
