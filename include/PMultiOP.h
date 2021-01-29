#ifndef PMultiOP
#define PMultiOP

/*
*  PMultiOP.h:
*  pointwise multiplication
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class PMultiNode : public Node, public Poolable<PMultiNode> {
public:
    Node *in1, *in2;

    PMultiNode() : Node("point-multiply") {
        in1 = nullptr;
        in2 = nullptr;
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void forward(Graph &graph, Node &input1, Node &input2) {
        setInputs({&input1, &input2});
        afterForward(graph, {&input1, &input2});
    }

    void setInputs(const vector<Node *> &inputs) override {
        in1 = inputs.at(0);
        in2 = inputs.at(1);
        if (in1->getDim() != getDim() || in2->getDim() != getDim()) {
            cerr << boost::format("PMultiNode setInputs dim error a:%1% b:%2% self:%3%") %
                in1->getDim() % in2->getDim() % getDim() << endl;
            abort();
        }
    }

    void compute() override {
        val().vec() = in1->val().vec() * in2->val().vec();
    }

    void backward() override {
        in1->loss().vec() += loss().vec() * in2->val().vec();
        in2->loss().vec() += loss().vec() * in1->val().vec();
    }

    Executor * generate() override;
};

class BatchedPMultiNode : public BatchedNodeImpl<PMultiNode> {
public:
    void init(Graph &graph, BatchedNode &a, BatchedNode &b) {
        allocateBatch(a.getDim(), a.batch().size());
        setInputsPerNode({&a, &b});
        afterInit(graph, {&a, &b});
    }
};

class PMultiExecutor :public Executor {
public:
    std::vector<dtype*> in_vals1;
    std::vector<dtype*> in_vals2;
    std::vector<dtype*> vals;
    Tensor1D y, x1, x2;
    int sumDim;


#if !USE_GPU
    int calculateFLOPs() override {
        return defaultFLOPs();
    }
#endif

#if USE_GPU
    void  forward() {
        int count = batch.size();
        for (Node *n : batch) {
            PMultiNode *pmulti = static_cast<PMultiNode*>(n);
            in_vals1.push_back(pmulti->in1->val().value);
            in_vals2.push_back(pmulti->in2->val().value);
            vals.push_back(pmulti->val().value);
        }
        n3ldg_cuda::PMultiForward(in_vals1, in_vals2, count, getDim(), vals);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch[idx]->val().verify("PMultiExecutor forward"));
        }
#endif
    }

    void backward() {
        int count = batch.size();
        std::vector<dtype*> losses, vals1, vals2, losses1, losses2;
        losses.reserve(count);
        vals1.reserve(count);
        vals2.reserve(count);
        losses1.reserve(count);
        losses2.reserve(count);
        for (Node *n : batch) {
            PMultiNode *pmulti = static_cast<PMultiNode*>(n);
            losses.push_back(pmulti->loss().value);
            vals1.push_back(pmulti->in1->val().value);
            vals2.push_back(pmulti->in2->val().value);
            losses1.push_back(pmulti->in1->loss().value);
            losses2.push_back(pmulti->in2->loss().value);
        }
        n3ldg_cuda::PMultiBackward(losses, vals1, vals2, count, getDim(), losses1, losses2);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }
        for (Node *n : batch) {
            PMultiNode *pmulti = static_cast<PMultiNode*>(n);
            n3ldg_cuda::Assert(pmulti->in1->loss().verify(
                        "PMultiExecutor backward in1 loss"));
            n3ldg_cuda::Assert(pmulti->in2->loss().verify(
                        "PMultiExecutor backward in2 loss"));
        }
#endif
    }
#endif
};

Executor * PMultiNode::generate() {
    PMultiExecutor* exec = new PMultiExecutor();
    return exec;
};

namespace n3ldg_plus {

Node *pointwiseMultiply(Graph &graph, Node &a, Node &b) {
    if (a.getDim() != b.getDim()) {
        cerr << boost::format("a dim:%1% b dim:%2%") % a.getDim() % b.getDim() << endl;
        abort();
    }
    PMultiNode *node = PMultiNode::newNode(a.getDim());
    node->forward(graph, a, b);
    return node;
}

BatchedNode *pointwiseMultiply(Graph &graph, BatchedNode &a, BatchedNode &b) {
    if (a.getDim() != b.getDim()) {
        cerr << boost::format("a dim:%1% b dim:%2%") % a.getDim() % b.getDim() << endl;
        abort();
    }
    BatchedPMultiNode *node = new BatchedPMultiNode;
    node->init(graph, a, b);
    return node;
}

}
#endif
