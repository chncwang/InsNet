#ifndef N3LDG_PLUS_LINEAR_H
#define N3LDG_PLUS_LINEAR_H

#include "n3ldg-plus/param/param.h"
#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

class LinearParam : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
public:
    LinearParam(const ::std::string &name) : b_(name + "-b", true), name_(name) {}

    ~LinearParam();

    void init(int out_dim, int in_dim, bool use_b = true,
            const ::std::function<dtype(int, int)> *bound = nullptr,
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
    ::std::vector<cuda::Transferable *> transferablePtrs() override;
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
    ::std::vector<Tunable<BaseParam>*> tunableComponents() override;

private:
    Param *W_ = nullptr;
    Param b_;
    ::std::string name_;
    bool bias_enabled_ = true;
    bool is_W_owner_ = true;
};

class BiasParam : public Param {
public:
    BiasParam(const ::std::string &name) : Param(name, true) {}

    void init(int outDim, int inDim) override {
        ::std::cerr << "BiasParam::init - unsupported method" << ::std::endl;
        abort();
    }

    void initAsBias(int dim) {
        Param::init(dim, 1);
    }
};

Node *linear(Graph &graph, Node &input, LinearParam &params);

Node *linear(Graph &graph, Node &input, Param &param);

Node *bias(Graph &graph, Node &input, BiasParam &param);

BatchedNode *bias(Graph &graph, BatchedNode &input, BiasParam &param);

}

#endif
