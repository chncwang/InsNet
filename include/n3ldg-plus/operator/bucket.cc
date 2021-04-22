#include "n3ldg-plus/operator/bucket.h"

using std::vector;
using std::cerr;

namespace n3ldg_plus {

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
            cerr << fmt::format("input size {} is not equal to dim {}\n", input.size(),
                getDim());
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

    Executor* generate() override;

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

Node *bucket(Graph &graph, int dim, float v) {
    vector<dtype> vals(dim);
    for (int i = 0; i < dim; ++i) {
        vals.at(i) = v;
    }
    BucketNode *bucket = BucketNode::newNode(dim);
    bucket->forward(graph, vals);
    return bucket;
}

Node *bucket(Graph &graph, const vector<dtype> &v) {
    BucketNode *bucket = BucketNode::newNode(v.size());
    bucket->forward(graph, v);
    return bucket;
}

BatchedNode *bucket(Graph &graph, int batch_size, const vector<dtype> &v) {
    BatchedBucketNode *node = new BatchedBucketNode;
    node->init(graph, v, batch_size);
    return node;
}

BatchedNode *bucket(Graph &graph, int dim, int batch_size, dtype v) {
    vector<dtype> vals(dim);
    for (int i = 0; i < dim; ++i) {
        vals.at(i) = v;
    }
    BatchedBucketNode *node = new BatchedBucketNode;
    node->init(graph, vals, batch_size);
    return node;
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
        vector<dtype*> ys(batch.size());
        vector<dtype> cpu_x(batch.size() * getDim());
        int batch_i = 0;
        int j = 0;
        for (Node *node : batch) {
            BucketNode *bucket = dynamic_cast<BucketNode*>(node);
            ys.at(batch_i++) = bucket->val().value;
            for (int i = 0; i < getDim(); ++i) {
                cpu_x.at(j++) = bucket->input_.at(i);
            }
        }
        cuda::BucketForward(cpu_x, count, getDim(), ys);
#if TEST_CUDA
        for (Node *node : batch) {
            BucketNode *bucket = dynamic_cast<BucketNode*>(node);
            dtype *v = node->val().v;
            for (int i = 0; i < getDim(); ++i) {
                v[i] = bucket->input_.at(i);
            }
            cuda::Assert(node->val().verify("bucket forward"));
        }
#endif
#else
        for (Node *node : batch) {
            BucketNode *bucket = dynamic_cast<BucketNode*>(node);
            node->val() = bucket->input_;
        }
#endif
    }

    void backward() override {}
};

Executor* BucketNode::generate() {
    return new BucketExecutor();
}

}
