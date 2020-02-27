#ifndef N3LDG_PLUS_LAYER_NORMALIZATION_H
#define N3LDG_PLUS_LAYER_NORMALIZATION_H

#include "MyLib.h"
#include "LookupTable.h"
#include "Node.h"
#include "Div.h"
#include "AtomicOP.h"
#include "Graph.h"
#include "Pooling.h"
#include "Sub.h"
#include "PMultiOP.h"
#include "Param.h"
#include "UniOP.h"

class LayerNormalizationParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents 
#endif
{
public:
    void init(int dim) {
        g_.init(dim, 1);
        g_.val.assignAll(1);
    }

    Param &g() {
        return g_;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["g"] = g_.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        g_.fromJson(json["g"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&g_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&g_};
    }

private:
    Param g_;
};

vector<Node *> layerNormalization(Graph &graph, LayerNormalizationParams &params,
        vector<Node *> &input_layer) {
    using namespace n3ldg_plus;
    Node *avg = averagePool(graph, input_layer);
    int len = input_layer.size();
    vector<Node *> square_nodes;
    square_nodes.reserve(len);
    vector<Node *> zeros_around;
    for (Node *input : input_layer) {
        Node *sub = n3ldg_plus::sub(graph, *input, *avg);
        zeros_around.push_back(sub);
        Node *square = pointwiseMultiply(graph, *sub, *sub);
        square_nodes.push_back(square);
    }
    Node *var = averagePool(graph, square_nodes);
    Node *standard_deviation = sqrt(graph, *var);
    Node *g = embedding(graph, params.g(), 0, true);
    Node *factor = fullDiv(graph, *g, *standard_deviation);

    vector<Node *> normalized_layer;
    for (Node *e : zeros_around) {
        Node *scaled = pointwiseMultiply(graph, *factor, *e);
        normalized_layer.push_back(scaled);
    }
    return normalized_layer;
}

#endif
