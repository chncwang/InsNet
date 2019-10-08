#ifndef PAddOP
#define PAddOP

/*
*  PAddOP.h:
*  (pointwise) add
*
*  Created on: June 13, 2017
*      Author: mszhang
*/

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class PAddNode : public Node {
public:
    vector<PNode> ins;

    PAddNode() : Node("point-add") {}

    void forward(Graph &graph, Node &input1, Node &input2) {
        vector<Node *> inputs = {&input1, &input2};
        this->forward(graph, inputs);
    }

    void forward(Graph &cg, vector<PNode>& x) {
        if (x.empty()) {
            std::cerr << "empty inputs for add" << std::endl;
            abort();
        }

        for (int i = 0; i < x.size(); i++) {
            if (x[i]->val().dim != getDim()) {
                std::cerr << "dim does not match" << std::endl;
                abort();
            }
        }

        ins = x;

        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins.at(i)->addParent(this);
        }

        cg.addNode(this);
    }

    void compute() override {
        int nSize = ins.size();
        val().zero();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < getDim(); idx++) {
                val()[idx] += ins.at(i)->val()[idx];
            }
        }
    }


    void backward() override {
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < getDim(); idx++) {
                ins.at(i)->loss()[idx] += loss()[idx];
            }
        }
    }


public:
    PExecutor generate() override;

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) override {
        if (!Node::typeEqual(other)) {
            return false;
        }
        PAddNode *add = static_cast<PAddNode*>(other);
        return ins.size() == add->ins.size();
    }

    string typeSignature() const override {
        return Node::typeSignature() + "-" + to_string(ins.size());
    }
};

namespace n3ldg_plus {
    Node *add(Graph &graph, vector<Node*> inputs) {
        for (int i = 1; i < inputs.size(); ++i) {
            if (inputs.front()->getDim() != inputs.at(i)->getDim()) {
                cerr << "input dims not equal" << endl;
                abort();
            }
        }

        int dim = inputs.front()->getDim();
        PAddNode *result = new PAddNode;
        result->init(dim);
        result->forward(graph, inputs);
        return result;
    }
}

class PAddExecutor : public Executor {
public:
    int in_count;
    int dim;

public:
    Tensor1D x, y;
    int sumDim;

#if USE_GPU
    void  forward() {
        int count = batch.size();

        std::vector<std::vector<dtype*>> in_vals;
        in_vals.reserve(in_count);
        for (int i = 0; i < in_count; ++i) {
            std::vector<dtype*> ins;
            ins.reserve(count);
            for (PNode n : batch) {
                PAddNode *padd = static_cast<PAddNode*>(n);
                ins.push_back(padd->ins.at(i)->val().value);
            }
            in_vals.push_back(ins);
        }
        std::vector<dtype *> outs;
        outs.reserve(count);
        for (PNode n : batch) {
            PAddNode *padd = static_cast<PAddNode*>(n);
            outs.push_back(padd->val().value);
        }
        n3ldg_cuda::PAddForward(in_vals, count, dim, in_count, outs);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
        }
        for (Node *n : batch) {
            n3ldg_cuda::Assert(n->val().verify("PAdd forward"));
        }
#endif
    }
#else
    void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
        }
    }
#endif

#if USE_GPU
    void backward() {
        int count = batch.size();
        std::vector<std::vector<dtype*>> in_losses;
        in_losses.reserve(in_count);
        for (int i = 0; i < in_count; ++i) {
            std::vector<dtype*> ins;
            ins.reserve(count);
            for (PNode n : batch) {
                PAddNode *padd = static_cast<PAddNode*>(n);
                ins.push_back(padd->ins.at(i)->loss().value);
            }
            in_losses.push_back(ins);
        }
        std::vector<dtype *> out_losses;
        out_losses.reserve(count);
        for (PNode n : batch) {
            PAddNode *padd = static_cast<PAddNode*>(n);
            out_losses.push_back(padd->loss().value);
        }
        n3ldg_cuda::PAddBackward(out_losses, count, dim, in_count, in_losses);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        for (Node *n : batch) {
            PAddNode *add = static_cast<PAddNode*>(n);
            for (Node *in : add->ins) {
                n3ldg_cuda::Assert(in->loss().verify("PAddExecutor backward"));
            }
        }
#endif
    }
#else
    void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }
    }
#endif
};

PExecutor PAddNode::generate() {
    PAddExecutor* exec = new PAddExecutor();
    exec->batch.push_back(this);
    exec->in_count = ins.size();
    exec->dim = getDim();
    return exec;
}

#endif
