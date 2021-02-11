#ifndef N3LDG_PLUS_SOFTMAX_NODE
#define N3LDG_PLUS_SOFTMAX_NODE

#include "AtomicOP.h"
#include "Sub.h"
#include "AtomicOP.h"
#include "Div.h"
#include "Split.h"

#include <boost/format.hpp>

class SoftmaxNode : public UniInputNode, public Poolable<SoftmaxNode> {
public:
    SoftmaxNode() : UniInputNode("softmax") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    Executor *generate() override;

    void compute() override {
        dtype max = getInput()->getVal().mat().maxCoeff();
        Tensor1D x_exp, x;
        x.init(getDim());
        x_exp.init(getDim());
        x.vec() = getInput()->getVal().vec() - max;
        x_exp.vec() = x.vec().exp();
        dtype sum = x_exp.mat().sum();
        val().vec() = x_exp.vec() / sum;
    }

    void backward() override {
        Tensor1D a;
        a.init(getDim());
        a.vec() = getLoss().vec() * getVal().vec();
        dtype z = a.mat().sum();
        Tensor1D b;
        b.init(getDim());
        b.vec() = z - a.vec();
        getInput()->loss().vec() += getVal().vec() *
            ((1 - getVal().vec()) * getLoss().vec() - b.vec());
    }

    string typeSignature() const override {
        return Node::getNodeType();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return true;
    }
};

class BatchedSoftmaxNode : public BatchedNodeImpl<SoftmaxNode> {
public:
    void init(Graph &graph, BatchedNode &input) {
        allocateBatch(input.getDims());
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
};

#if USE_GPU
#else
class SoftmaxExecutor : public UniInputExecutor {
public:
    int calculateFLOPs() override {
        return 0; // TODO
    }
};
#endif

Executor *SoftmaxNode::generate() {
    return new SoftmaxExecutor;
}

namespace n3ldg_plus {

Node *minusMaxScalar(Graph &graph, Node &input, int input_col) {
    using namespace n3ldg_plus;

    Node *max_scalar = maxScalar(graph, input, input_col);
    int input_row = input.getDim() / input_col;
    Node *scalar_to_vector = scalarToVector(graph, *max_scalar, input_row);
    Node *subtracted = sub(graph, input, *scalar_to_vector);
    return subtracted;
}

BatchedNode *minusMaxScalar(Graph &graph, BatchedNode &input, int input_col) {
    using namespace n3ldg_plus;

    BatchedNode *max_scalar = maxScalar(graph, input, input_col);
    vector<int> input_rows;
    for (int dim : input.getDims()) {
        input_rows.push_back(dim / input_col);
    }
    BatchedNode *scalar_to_vector = scalarToVector(graph, *max_scalar, input_rows);
    BatchedNode *subtracted = sub(graph, input, *scalar_to_vector);
    return subtracted;
}

Node* softmax(Graph &graph, Node &input, int input_col) {
    using namespace n3ldg_plus;
    Node *subtracted = minusMaxScalar(graph, input, input_col);
    Node *exp = n3ldg_plus::exp(graph, *subtracted);
    Node *sum = vectorSum(graph, *exp, input_col);
    int input_row = input.getDim() / input_col;
    sum = scalarToVector(graph, *sum, input_row);
    Node *div = n3ldg_plus::fullDiv(graph, *exp, *sum);
    return div;
}

BatchedNode* softmax(Graph &graph, BatchedNode &input, int input_col = 1) {
    using namespace n3ldg_plus;
    BatchedSoftmaxNode *node = new BatchedSoftmaxNode;
    node->init(graph, input);
    return node;

//    BatchedNode *subtracted = minusMaxScalar(graph, input, input_col);
//    BatchedNode *exp = n3ldg_plus::exp(graph, *subtracted);
//    BatchedNode *sum = vectorSum(graph, *exp, input_col);
//    vector<int> input_rows;
//    for (int dim : input.getDims()) {
//        input_rows.push_back(dim / input_col);
//    }
//    sum = scalarToVector(graph, *sum, input_rows);
//    BatchedNode *div = n3ldg_plus::fullDiv(graph, *exp, *sum);
//    return div;
}

};

#endif
