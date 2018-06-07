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

    inline PExecute generate(bool bTrain);
};


#if USE_GPU
class PMultiExecute :public Execute {
  public:
    Tensor1D y, x1, x2;
    int sumDim;
    bool bTrain;

  public:
    inline void  forward() {
        int count = batch.size();
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
            ptr->forward_drop(bTrain);
        }
    }

    inline void  backward() {
        Tensor1D ly, lx1, lx2;
        ly.init(sumDim);
        lx1.init(sumDim);
        lx2.init(sumDim);

        int count = batch.size();
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

};

inline PExecute PMultiNode::generate(bool bTrain) {
    PMultiExecute* exec = new PMultiExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}

#elif USE_BASE
class PMultiExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute PMultiNode::generate(bool bTrain) {
    PMultiExecute* exec = new PMultiExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};

#else
class PMultiExecute :public Execute {
  public:
    Tensor1D y, x1, x2;
    int sumDim;
    bool bTrain;

  public:
    inline void  forward() {
        int count = batch.size();
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
            ptr->forward_drop(bTrain);
        }
    }

    inline void  backward() {
        Tensor1D ly, lx1, lx2;
        ly.init(sumDim);
        lx1.init(sumDim);
        lx2.init(sumDim);

        int count = batch.size();
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

};

inline PExecute PMultiNode::generate(bool bTrain) {
    PMultiExecute* exec = new PMultiExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}
#endif


#endif
