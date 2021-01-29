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
#include <vector>
#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

using namespace Eigen;
using std::vector;

class BucketNode : public Node, public Poolable<BucketNode> {
public:
    BucketNode() : Node("bucket") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void forward(Graph &graph, const vector<dtype> &input) {
        if (input.size() != getDim()) {
            cerr << boost::format("input size %1% is not equal to dim %2%") % input.size() %
                getDim() << endl;
            abort();
        }
        input_ = input;
        graph.addNode(this);
    }

    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }

    PExecutor generate() override;

    void setVals(const vector<dtype> &vals) {
        input_ = vals;
    }

private:
    vector<dtype> input_;
    friend class BucketExecutor;
};

class BatchedBucketNode : public BatchedNodeImpl<BucketNode> {
public:
    void init(Graph &graph, const vector<dtype> &vals, int batch_size) {
        allocateBatch(vals.size(), batch_size);
        for (Node *node : batch()) {
            BucketNode *b = dynamic_cast<BucketNode *>(node);
            b->setVals(vals);
        }
        graph.addNode(this);
    }
};

namespace n3ldg_plus {

Node *bucket(Graph &graph, int dim, float v) {
    vector<dtype> vals;
    vals.reserve(dim);
    for (int i = 0; i < dim; ++i) {
        vals.push_back(v);
    }
    BucketNode *bucket = BucketNode::newNode(dim);
    bucket->forward(graph, vals);
    return bucket;
}

Node *bucket(Graph &graph, const vector<float> &v) {
    BucketNode *bucket = BucketNode::newNode(v.size());
    bucket->forward(graph, v);
    return bucket;
}

BatchedBucketNode *bucket(Graph &graph, int batch_size, const vector<dtype> &v) {
    BatchedBucketNode *node = new BatchedBucketNode;
    node->init(graph, v, batch_size);
    return node;
}

BatchedBucketNode *bucket(Graph &graph, int dim, int batch_size, dtype v) {
    vector<dtype> vals;
    vals.reserve(dim);
    for (int i = 0; i < dim; ++i) {
        vals.push_back(v);
    }
    BatchedBucketNode *node = new BatchedBucketNode;
    node->init(graph, vals, batch_size);
    return node;
}

}

class BucketExecutor : public Executor {
public:
#if !USE_GPU
    int calculateFLOPs() override {
        return 0;
    }
#endif

    void forward() override {
#if USE_GPU
        int count = batch.size();
        vector<dtype*> ys;
        vector<dtype> cpu_x;
        cpu_x.reserve(getDim() * count);
        for (Node *node : batch) {
            BucketNode *bucket = static_cast<BucketNode*>(node);
            ys.push_back(bucket->val().value);
            for (int i = 0; i < getDim(); ++i) {
                cpu_x.push_back(bucket->input_.at(i));
            }
        }
        n3ldg_cuda::BucketForward(cpu_x, count, getDim(), ys);
#if TEST_CUDA
        for (Node *node : batch) {
            BucketNode *bucket = static_cast<BucketNode*>(node);
            dtype *v = node->val().v;
            for (int i = 0; i < getDim(); ++i) {
                v[i] = bucket->input_.at(i);
            }
            n3ldg_cuda::Assert(node->val().verify("bucket forward"));
        }
#endif
#else
        for (Node *node : batch) {
            BucketNode *bucket = static_cast<BucketNode*>(node);
            node->val() = bucket->input_;
        }
#endif
    }

    void backward() override {}
};

PExecutor BucketNode::generate() {
    return new BucketExecutor();
}

#endif
