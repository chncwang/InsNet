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
        this->forward(&graph, &input1, &input2);
    }

    void forward(Graph *cg, Node * x1, Node * x2) {
        in1 = x1;
        in2 = x2;
        x1->addParent(this);
        x2->addParent(this);
        cg->addNode(this);
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

}
#endif
