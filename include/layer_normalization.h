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

class StandardLayerNormNode : public UniInputNode, public Poolable<StandardLayerNormNode> {
public:
    StandardLayerNormNode() : UniInputNode("standard-layernorm") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        Node::setDim(dim);
    }

    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }

    PExecutor generate() override;

    string typeSignature() const override {
        return Node::typeSignature();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();;
    }

private:
    friend class LayerNormExecutor;
};

class BatchedStandardLayerNormNode : public BatchedNodeImpl<StandardLayerNormNode> {
public:
    void init(Graph &graph, BatchedNode &input, LayerNormalizationParams &params) {
        allocateBatch(input.getDim(), input.batch().size());
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
};

#if USE_GPU
class LayerNormExecutor : public UniInputExecutor {
public:
    void forward() override {
        int count = batch.size();
        Tensor1D means;
        means.init(count);
        sds_.init(count);
        vector<dtype *> in_vals(count), vals(count);
        int i = 0;
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            in_vals.at(i) = s.getInput()->getVal().value;
            vals.at(i++) = s.getVal().value;
        }
        n3ldg_cuda::StandardLayerNormForward(in_vals, count, getDim(), vals, sds_.value);
        vals_ = move(vals);
#if TEST_CUDA
        cpu_sds_.init(batch.size());
        i = 0;
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            auto &input = s.getInput()->getVal();
            dtype mean = input.mat().sum() / s.getDim();
            Tensor1D x;
            x.init(s.getDim());
            x.vec() = (input.vec() - mean).square();
            dtype sd = sqrt(x.mat().sum() / s.getDim());
            sds_[i++] = sd;
            s.val().vec() = ((input.vec() - mean) / sd);
        }
        verifyForward();
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype *> in_grads(count), grads(count);
        int i = 0;
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            in_grads.at(i) = s.getInput()->getLoss().value;
            grads.at(i++) = s.getLoss().value;
        }
        n3ldg_cuda::StandardLayerNormBackward(grads, count, getDim(), vals_, sds_.value, in_grads);
#if TEST_CUDA
        verifyBackward();
#endif
    }

private:
    Tensor1D sds_;
    vector<dtype *> vals_;
#if TEST_CUDA
    n3ldg_cpu::Tensor1D cpu_sds_;
#endif
};
#else
class LayerNormExecutor : public UniInputExecutor {
public:
    void forward() override {
        sds_.init(batch.size());
        int i = 0;
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            auto &input = s.getInput()->getVal();
            dtype mean = input.mat().sum() / s.getDim();
            Tensor1D x;
            x.init(s.getDim());
            x.vec() = (input.vec() - mean).square();
            dtype sd = sqrt(x.mat().sum() / s.getDim());
            sds_[i++] = sd;
            s.val().vec() = ((input.vec() - mean) / sd);
        }
    }

    void backward() override {
        int i = 0;
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            int n = s.getDim();
            dtype c = 1.0 / (n * sds_[i]);
            auto y2 = s.getVal().vec().square();
            Tensor1D m;
            m.init(n);
            m.vec() = s.getLoss().vec() * s.getVal().vec();
            s.getInput()->loss().vec() += c * ((n - 1 - y2) * s.getLoss().vec() -
                    ((m.mat().sum() - m.vec()) * s.getVal().vec() + s.getLoss().mat().sum() -
                     s.getLoss().vec()));
            ++i;
        }
    }

    int calculateFLOPs() override {
        return 0; // TODO
    }

private:
    Tensor1D sds_;
};
#endif

Executor *StandardLayerNormNode::generate() {
    return new LayerNormExecutor;
}


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
//    BatchedNode *sum = vectorSum(graph, input_layer, 1);
//    BatchedNode *avg = scaled(graph, *sum, 1.0 / input_layer.getDim());
//    BatchedNode *avg_vector = scalarToVector(graph, *avg, input_layer.getDim());
//    BatchedNode *zeros_around = sub(graph, input_layer, *avg_vector);
//    BatchedNode *square = pointwiseMultiply(graph, *zeros_around, *zeros_around);
//    BatchedNode *square_sum = vectorSum(graph, *square, 1);
//    BatchedNode *eps = bucket(graph, 1, input_layer.batch().size(), 1e-6);
//    square_sum = addInBatch(graph, {square_sum, eps});
//    BatchedNode *var = scaled(graph, *square_sum, 1.0 / input_layer.getDim());
//    BatchedNode *standard_deviation = sqrt(graph, *var);
//    standard_deviation = scalarToVector(graph, *standard_deviation, input_layer.getDim());
//    BatchedNode *g = embedding(graph, params.g(), 0, input_layer.batch().size(), true);
//    BatchedNode *factor = fullDiv(graph, *g, *standard_deviation);

//    BatchedNode *scaled = pointwiseMultiply(graph, *factor, *zeros_around);
//    BatchedNode *biased = bias(graph, *scaled, params.b());
//    return biased;
    BatchedStandardLayerNormNode *node = new BatchedStandardLayerNormNode;
    node->init(graph, input_layer, params);
    return node;
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
