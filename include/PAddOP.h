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

    ~PAddNode() {
        ins.clear();
    }
public:
    PAddNode() : Node() {
        ins.clear();
        node_type = "point-add";
    }

    void forward(Graph &graph, Node &input1, Node &input2) {
        vector<Node *> inputs = {&input1, &input2};
        this->forward(&graph, inputs);
    }

    void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for add" << std::endl;
            return;
        }

        ins.clear();
        for (int i = 0; i < x.size(); i++) {
            if (x[i]->val.dim == dim) {
                ins.push_back(x[i]);
            }
            else {
                std::cout << "dim does not match" << std::endl;
            }
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x4->dim == dim) {
            ins.push_back(x4);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x4->dim == dim) {
            ins.push_back(x4);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x5->dim == dim) {
            ins.push_back(x5);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x4->dim == dim) {
            ins.push_back(x4);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x5->dim == dim) {
            ins.push_back(x5);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x6->dim == dim) {
            ins.push_back(x6);
        }
        else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

public:
    void compute() {
        int nSize = ins.size();
        val.zero();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < dim; idx++) {
                val[idx] += ins[i]->val[idx];
            }
        }
    }


    void backward() {
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < dim; idx++) {
                ins[i]->loss[idx] += loss[idx];
            }
        }
    }


public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        if (!Node::typeEqual(other)) {
            return false;
        }
        PAddNode *add = static_cast<PAddNode*>(other);
        return ins.size() == add->ins.size();
    }

    size_t typeHashCode() const override {
        return (std::hash<int>{}(ins.size()) << 1) ^ Node::typeHashCode();
    }
};


class PAddExecute : public Execute {
public:
    int in_count;
    int dim;
    Tensor2D drop_mask;

public:
    Tensor1D x, y;
    int sumDim;

#if USE_GPU
    void  forward() {
        int count = batch.size();

        drop_mask.init(dim, count);
        CalculateDropMask(count, dim, drop_mask);

        std::vector<std::vector<dtype*>> in_vals;
        in_vals.reserve(in_count);
        for (int i = 0; i < in_count; ++i) {
            std::vector<dtype*> ins;
            ins.reserve(count);
            for (PNode n : batch) {
                PAddNode *padd = static_cast<PAddNode*>(n);
                ins.push_back(padd->ins.at(i)->val.value);
            }
            in_vals.push_back(ins);
        }
        std::vector<dtype *> outs;
        outs.reserve(count);
        for (PNode n : batch) {
            PAddNode *padd = static_cast<PAddNode*>(n);
            outs.push_back(padd->val.value);
        }
        n3ldg_cuda::PAddForward(in_vals, count, dim, in_count, drop_mask.value,
            drop_factor, outs);
#if TEST_CUDA
        drop_mask.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < dim; ++j) {
                dtype v = drop_mask[i][j];
                batch[i]->drop_mask[j] = v <= drop_factor ? 0 : 1;
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
        for (Node *n : batch) {
            n3ldg_cuda::Assert(n->val.verify("PAdd forward"));
        }
#endif
    }
#else
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
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
                ins.push_back(padd->ins.at(i)->loss.value);
            }
            in_losses.push_back(ins);
        }
        std::vector<dtype *> out_losses;
        out_losses.reserve(count);
        for (PNode n : batch) {
            PAddNode *padd = static_cast<PAddNode*>(n);
            out_losses.push_back(padd->loss.value);
        }
        n3ldg_cuda::PAddBackward(out_losses, count, dim, in_count,
            drop_mask.value, drop_factor, in_losses);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }

        for (Node *n : batch) {
            PAddNode *add = static_cast<PAddNode*>(n);
            for (Node *in : add->ins) {
                n3ldg_cuda::Assert(in->loss.verify("PAddExecute backward"));
            }
        }
#endif
    }
#else
    void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
#endif
};


PExecute PAddNode::generate(bool bTrain, dtype cur_drop_factor) {
    PAddExecute* exec = new PAddExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor * drop_value;
    exec->in_count = ins.size();
    exec->dim = dim;
    return exec;
}


#endif
