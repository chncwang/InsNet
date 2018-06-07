#ifndef BucketOP
#define BucketOP

/*
*  BucketOP.h:
*  a bucket operation, for padding mainly
*  usually an inputleaf node, degree = 0
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

using namespace Eigen;



class BucketNode : public Node {
  public:
    BucketNode() : Node() {
        node_type = "bucket";
    }
  public:
    virtual inline void clearValue() {
        //Node::clearValue();
        loss = 0;
        degree = 0;
        if (drop_value > 0)drop_mask = 1;
        parents.clear();
    }

    virtual inline void init(int ndim, dtype dropout) {
        Node::init(ndim, -1);
    }

  public:
    void forward(Graph *cg, dtype value) {
        val = value;
        loss = 0;
        degree = 0;
        cg->addNode(this);
    }

    //value already assigned
    void forward(Graph *cg) {
        loss = 0;
        degree = 0;
        cg->addNode(this);
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

};

//#if USE_GPU
//class BucketExecute : public Execute {
//public:
//  bool bTrain;
//public:
//  inline void  forward() {
//    int count = batch.size();
//    for (int idx = 0; idx < count; idx++) {
//      BucketNode* ptr = (BucketNode*)batch[idx];
//      ptr->forward_drop(bTrain);
//    }
//  }
//
//  inline void backward() {
//    int count = batch.size();
//    for (int idx = 0; idx < count; idx++) {
//      BucketNode* ptr = (BucketNode*)batch[idx];
//      ptr->backward_drop();
//    }
//  }
//};
//
//inline PExecute BucketNode::generate(bool bTrain) {
//  BucketExecute* exec = new BucketExecute();
//  exec->batch.push_back(this);
//  exec->bTrain = bTrain;
//  return exec;
//}
//#else
class BucketExecute : public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            BucketNode* ptr = (BucketNode*)batch[idx];
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            BucketNode* ptr = (BucketNode*)batch[idx];
            ptr->backward_drop();
        }
    }
};

inline PExecute BucketNode::generate(bool bTrain) {
    BucketExecute* exec = new BucketExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}
//#endif

#endif
