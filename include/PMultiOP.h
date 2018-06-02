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
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};

inline PExecute PMultiNode::generate(bool bTrain, dtype cur_drop_factor) {
    PMultiExecute* exec = new PMultiExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};


#endif
