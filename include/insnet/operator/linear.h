#ifndef INSNET_LINEAR_H
#define INSNET_LINEAR_H

#include "insnet/param/param.h"
#include "insnet/computation-graph/graph.h"

namespace insnet {

class LinearParams : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
public:
    LinearParams(const std::string &name = "") : b_(name + "-b", true), name_(name) {}

    ~LinearParams();

    void init(int out_dim, int in_dim, bool use_b = true,
            const std::function<dtype(int, int)> *bound = nullptr,
            InitDistribution dist = InitDistribution::UNI);

    void init(Param &W);

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(bias_enabled_, *W_);
        if (bias_enabled_) {
            ar(b_);
        }
    }

#if USE_GPU
    std::vector<cuda::Transferable *> transferablePtrs() override;
#endif

    Param &W() {
        return *W_;
    }

    Param &b() {
        return b_;
    }

    bool biasEnabled() const {
        return bias_enabled_;
    }

protected:
    std::vector<Tunable<BaseParam>*> tunableComponents() override;

private:
    Param *W_ = nullptr;
    Param b_;
    std::string name_;
    bool bias_enabled_ = true;
    bool is_W_owner_ = true;
};

class BiasParam : public Param {
public:
    BiasParam(const std::string &name) : Param(name, true) {}

    void init(int outDim, int inDim) override {
        std::cerr << "BiasParam::init - unsupported method" << std::endl;
        abort();
    }

    void initAsBias(int dim) {
        Param::init(dim, 1);
    }
};

/// \ingroup operator
/// The linear transformation with bias. \f${W^T}{X} + [b b .. b]\f$.
///
/// You can disable the bias term when initializing *params*.
///
/// **The operators with the same parameters will be executed in batch.**
/// For example, supposing we have *params* with a 2x2 weight matrix, linear([0.1, 0.2, 0.3, 0.4], params) and linear([0.1, 0.2], params) will be executed in batch.
/// \param X The input tensor.
/// \param params W and b.
/// \return The transformed tensor. its size is equal to X.size() / params.W.row() * params.W.col().
Node *linear(Node &X, LinearParams &params);

/// \ingroup operator
/// The linear transformation. \f${W^T}{X}\f$.
///
/// This operator is especially useful when you want to share the weight matrix with another component. For example, to tie the input and output embeddings, call this operator like *linear(h, emb_table.param())*.
///
/// **The operators with the same weight matrix will be executed in batch.**
/// \param X The input tensor.
/// \param W The weight matrix.
/// \return The transformed tensor. its size is equal to X.size() / W.row() * W.col().
Node *linear(Node &X, Param &W);

/// \ingroup operator
/// Add the bias iterm to the input tensor. \f$X + [b b .. b]\f$.
///
/// **The operators with the same bias and X.size() will be executed in batch.** Note that this batching rule needs to be loosed in the future version.
/// \param X The input tensor.
/// \param b The bias parameters.
/// \Return The result tensor. Its size is equal to X.size().
Node *bias(Node &X, BiasParam &b);

}

#endif
