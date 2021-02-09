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

Node *layerNormalization(Graph &graph, LayerNormalizationParams &params,
        Node &input_layer) {
    using namespace n3ldg_plus;
    Node *sum = vectorSum(graph, input_layer, 1);
    Node *avg = scaled(graph, *sum, 1.0 / input_layer.getDim());
    Node *avg_vector = scalarToVector(graph, *avg, input_layer.getDim());
    Node *zeros_around = sub(graph, input_layer, *avg_vector);
    Node *square = pointwiseMultiply(graph, *zeros_around, *zeros_around);
    Node *square_sum = vectorSum(graph, *square, 1);
    Node *eps = bucket(graph, 1, 1e-6);
    square_sum = add(graph, {square_sum, eps});
    Node *var = scaled(graph, *square_sum, 1.0 / input_layer.getDim());
    Node *standard_deviation = sqrt(graph, *var);
    standard_deviation = scalarToVector(graph, *standard_deviation, input_layer.getDim());
    Node *g = embedding(graph, params.g(), 0, true);
    Node *factor = fullDiv(graph, *g, *standard_deviation);

    Node *scaled = pointwiseMultiply(graph, *factor, *zeros_around);
    Node *biased = bias(graph, *scaled, params.b());
    return biased;
}

BatchedNode *layerNormalization(Graph &graph, LayerNormalizationParams &params,
        BatchedNode &input_layer) {
    using namespace n3ldg_plus;
//    return bias(graph, input_layer, params.b());
    BatchedNode *sum = vectorSum(graph, input_layer, 1);
    BatchedNode *avg = scaled(graph, *sum, 1.0 / input_layer.getDim());
    BatchedNode *avg_vector = scalarToVector(graph, *avg, input_layer.getDim());
    BatchedNode *zeros_around = sub(graph, input_layer, *avg_vector);
    BatchedNode *square = pointwiseMultiply(graph, *zeros_around, *zeros_around);
    BatchedNode *square_sum = vectorSum(graph, *square, 1);
    BatchedNode *eps = bucket(graph, 1, input_layer.batch().size(), 1e-6);
    square_sum = addInBatch(graph, {square_sum, eps});
    BatchedNode *var = scaled(graph, *square_sum, 1.0 / input_layer.getDim());
    BatchedNode *standard_deviation = sqrt(graph, *var);
    standard_deviation = scalarToVector(graph, *standard_deviation, input_layer.getDim());
    BatchedNode *g = embedding(graph, params.g(), 0, input_layer.batch().size(), true);
    BatchedNode *factor = fullDiv(graph, *g, *standard_deviation);

    BatchedNode *scaled = pointwiseMultiply(graph, *factor, *zeros_around);
    BatchedNode *biased = bias(graph, *scaled, params.b());
    return biased;
}

vector<Node *> layerNormalization(Graph &graph, LayerNormalizationParams &params,
        const vector<Node *> &input_layer) {
    vector<Node *> results;
    results.reserve(input_layer.size());
    for (Node *x : input_layer) {
        results.push_back(layerNormalization(graph, params, *x));
    }
    return results;
}

#endif
