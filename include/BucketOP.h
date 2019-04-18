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

    virtual void init(int ndim) {
#if USE_GPU
        Node::initOnHostAndDevice(ndim);
#else
        Node::init(ndim);
#endif
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
      degree = 0;
      Vec(val.v, dim) = Vec(value, dim);
      cg->addNode(this);
    }

    void compute() {}

    void backward() {}

    PExecute generate();

    bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

};

class BucketExecute : public Execute {
};

PExecute BucketNode::generate() {
    BucketExecute* exec = new BucketExecute();
    exec->batch.push_back(this);
    return exec;
}

#endif
