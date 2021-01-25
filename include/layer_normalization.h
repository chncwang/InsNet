#ifndef N3LDG_PLUS_LAYER_NORMALIZATION_H
#define N3LDG_PLUS_LAYER_NORMALIZATION_H

#include "MyLib.h"
#include "LookupTable.h"
#include "Node.h"
#include "PAddOP.h"
#include "BucketOP.h"
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
    LayerNormalizationParams(const string &name) : g_(name + "-g"), b_(name + "-b") {}

    void init(int dim) {
        g_.init(dim, 1);
        g_.val.assignAll(1);
        b_.initAsBias(dim);
    }

    Param &g() {
        return g_;
    }

    BiasParam &b() {
        return b_;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["g"] = g_.toJson();
        json["b"] = b_.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        g_.fromJson(json["g"]);
        b_.fromJson(json["b"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
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

AtomicNode *layerNormalization(Graph &graph, LayerNormalizationParams &params,
        AtomicNode &input_layer) {
    using namespace n3ldg_plus;
    AtomicNode *sum = vectorSum(graph, input_layer, 1);
    AtomicNode *avg = scaled(graph, *sum, 1.0 / input_layer.getDim());
    AtomicNode *avg_vector = scalarToVector(graph, input_layer.getDim(), *avg);
    AtomicNode *zeros_around = sub(graph, input_layer, *avg_vector);
    AtomicNode *square = pointwiseMultiply(graph, *zeros_around, *zeros_around);
    AtomicNode *square_sum = vectorSum(graph, *square, 1);
    AtomicNode *eps = bucket(graph, 1, 1e-6);
    square_sum = add(graph, {square_sum, eps});
    AtomicNode *var = scaled(graph, *square_sum, 1.0 / input_layer.getDim());
    AtomicNode *standard_deviation = sqrt(graph, *var);
    standard_deviation = scalarToVector(graph, input_layer.getDim(), *standard_deviation);
    AtomicNode *g = embedding(graph, params.g(), 0, true);
    AtomicNode *factor = fullDiv(graph, *g, *standard_deviation);

    AtomicNode *scaled = pointwiseMultiply(graph, *factor, *zeros_around);
    AtomicNode *biased = bias(graph, params.b(), *scaled);
    return biased;
}

vector<AtomicNode *> layerNormalization(Graph &graph, LayerNormalizationParams &params,
        const vector<AtomicNode *> &input_layer) {
    vector<AtomicNode *> results;
    results.reserve(input_layer.size());
    for (AtomicNode *x : input_layer) {
        results.push_back(layerNormalization(graph, params, *x));
    }
    return results;
}

#endif
