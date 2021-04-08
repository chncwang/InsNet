#ifndef N3LDG_PLUS_LAYER_NORMALIZATION_H
#define N3LDG_PLUS_LAYER_NORMALIZATION_H

#include "n3ldg-plus/operator/linear.h"

namespace n3ldg_plus {

class LayerNormalizationParams : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents 
#endif
{
public:
    LayerNormalizationParams(const ::std::string &name) : g_(name + "-g"), b_(name + "-b") {}

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
    ::std::vector<cuda::Transferable *> transferablePtrs() override {
        return {&g_, &b_};
    }
#endif

protected:
    virtual ::std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&g_, &b_};
    }

private:
    Param g_;
    BiasParam b_;
};

Node *layerNormalization(Graph &graph, LayerNormalizationParams &params, Node &input_layer,
        int col = 1);

BatchedNode *layerNormalization(Graph &graph, LayerNormalizationParams &params,
        BatchedNode &input_layer);

::std::vector<Node *> layerNormalization(Graph &graph, LayerNormalizationParams &params,
        const ::std::vector<Node *> &input_layer);

}

#endif
