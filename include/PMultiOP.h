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

class PMultiNode : public Node {
  public:
    PNode in1, in2;
  public:
    PMultiNode() : Node() {
        in1 = NULL;
        in2 = NULL;
        node_type = "point-multiply";
    }
  public:
    virtual inline void clearValue() {
        Node::clearValue();
        in1 = NULL;
        in2 = NULL;
    }

  public:
    void forward(Graph *cg, PNode x1, PNode x2) {
        in1 = x1;
        in2 = x2;
        degree = 0;
        x1->addParent(this);
        x2->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        val.vec() = in1->val.vec() * in2->val.vec();
    }

    void backward() {
        in1->loss.vec() += loss.vec() * in2->val.vec();
        in2->loss.vec() += loss.vec() * in1->val.vec();
    }

  public:
    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

    inline PExecute generate(bool bTrain, dtype cur_drop_factor);
};

class PMultiExecute :public Execute {
public:
    Tensor2D drop_mask;
    std::vector<dtype*> in_vals1;
    std::vector<dtype*> in_vals2;
    std::vector<dtype*> vals;
    int dim;
public:
    Tensor1D y, x1, x2;
    int sumDim;
    bool bTrain;

public:
#if USE_GPU
    void  forward() {
        int count = batch.size();
        drop_mask.init(dim, count);
        CalculateDropMask(count, dim, drop_mask);
        for (Node *n : batch) {
            PMultiNode *pmulti = static_cast<PMultiNode*>(n);
            in_vals1.push_back(pmulti->in1->val.value);
            in_vals2.push_back(pmulti->in2->val.value);
            vals.push_back(pmulti->val.value);
        }
        n3ldg_cuda::PMultiForward(in_vals1, in_vals2, count, dim, bTrain,
                drop_mask.value, dynamicDropValue(), vals);
#if TEST_CUDA
        drop_mask.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < dim; ++j) {
                dtype v = drop_mask[j][i];
                batch[i]->drop_mask[j] = v <= dynamicDropValue() ? 0 : 1;
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
            n3ldg_cuda::Assert(batch[idx]->val.verify(
                        "PMultiExecute forward"));
        }
#endif
    }
#else
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
            sumDim += batch[idx]->dim;
        }
        y.init(sumDim);
        x1.init(sumDim);
        x2.init(sumDim);
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                x1[offset + idy] = ptr->in1->val[idy];
                x2[offset + idy] = ptr->in2->val[idy];
            }
            offset += ptr->dim;
        }
        y.vec() = x1.vec() * x2.vec();

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->val[idy] = y[offset + idy];
            }
            offset += ptr->dim;
            ptr->forward_drop(bTrain,1);
        }
    }
#endif

#if USE_GPU
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
            losses.push_back(pmulti->loss.value);
            vals1.push_back(pmulti->in1->val.value);
            vals2.push_back(pmulti->in2->val.value);
            losses1.push_back(pmulti->in1->loss.value);
            losses2.push_back(pmulti->in2->loss.value);
        }
        n3ldg_cuda::PMultiBackward(losses, vals1, vals2, count, dim,
                drop_mask.value, dynamicDropValue(), losses1, losses2);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
        for (Node *n : batch) {
            PMultiNode *pmulti = static_cast<PMultiNode*>(n);
            n3ldg_cuda::Assert(pmulti->in1->loss.verify(
                        "PMultiExecute backward in1 loss"));
            n3ldg_cuda::Assert(pmulti->in2->loss.verify(
                        "PMultiExecute backward in2 loss"));
        }
#endif
    }
#else
    void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        Tensor1D ly, lx1, lx2;
        ly.init(sumDim);
        lx1.init(sumDim);
        lx2.init(sumDim);
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < ptr->dim; idy++) {
                ly[offset + idy] = ptr->loss[idy];
            }
            offset += ptr->dim;
        }
        lx1.vec() = ly.vec() * x2.vec();
        lx2.vec() = ly.vec() * x1.vec();
        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->in1->loss[idy] += lx1[offset + idy];
                ptr->in2->loss[idy] += lx2[offset + idy];
            }
            offset += ptr->dim;
        }
    }
#endif
};

inline PExecute PMultiNode::generate(bool bTrain, dtype cur_drop_factor) {
    PMultiExecute* exec = new PMultiExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    exec->dim = dim;
    return exec;
};


#endif
