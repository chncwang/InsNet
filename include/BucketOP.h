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
    virtual void init(int ndim, dtype dropout) {
        Node::init(ndim, -1);
    }

    void forward(Graph &graph, dtype value) {
        this->forward(&graph, value);
    }

    void forward(Graph *cg, dtype value) {
#if TEST_CUDA
        val  = value;
        loss = 0;
#endif
#if USE_GPU
        n3ldg_cuda::Memset(val.value, dim, value);
        n3ldg_cuda::Memset(loss.value, dim, 0.0f);
#if TEST_CUDA
        n3ldg_cuda::Assert(val.verify("bucket forward"));
        n3ldg_cuda::Assert(loss.verify("loss verify"));
#endif
#else
        val = value;
        loss = 0;
#endif
        degree = 0;
        cg->addNode(this);
    }

    void forward(Graph &graph) {
        this->forward(&graph);
    }

    //value already assigned
    void forward(Graph *cg) {
#if USE_GPU
        n3ldg_cuda::Memset(loss.value, dim, 0.0f);
#else
        loss = 0;
#endif
        degree = 0;
        cg->addNode(this);
    }

    void forwardArr(Graph *cg, dtype *value) {
#if USE_GPU
      abort();
#else
      Vec(val.v, dim) = Vec(value, dim);
      degree = 0;
      cg->addNode(this);
#endif
    }

    void compute() {

    }

    void backward() {

    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

};

#if USE_GPU
class BucketExecute : public Execute {
  public:
    void  forward() override {}

    void backward() override {}
};

#else
class BucketExecute : public Execute {
  public:
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
        }
    }
};
#endif

PExecute BucketNode::generate(bool bTrain, dtype cur_drop_factor) {
    BucketExecute* exec = new BucketExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
}

#endif
