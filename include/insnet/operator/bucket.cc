#include "insnet/operator/bucket.h"

using std::vector;
using std::cerr;

namespace insnet {

class BucketNode : public Node, public Poolable<BucketNode> {
public:
    BucketNode() : Node("bucket") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void forward(Graph &graph, const vector<dtype> &input) {
        if (input.size() != size()) {
            cerr << fmt::format("input size {} is not equal to dim {}\n", input.size(),
                size());
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

protected:
    int forwardOnlyInputValSize() override {
        return 0;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    vector<dtype> input_;
    friend class BucketExecutor;
};

Node *tensor(Graph &graph, int dim, dtype v) {
    vector<dtype> vals(dim);
    for (int i = 0; i < dim; ++i) {
        vals.at(i) = v;
    }
    BucketNode *bucket = BucketNode::newNode(dim);
    bucket->forward(graph, vals);
    return bucket;
}

Node *tensor(Graph &graph, const vector<dtype> &v) {
    BucketNode *bucket = BucketNode::newNode(v.size());
    bucket->forward(graph, v);
    return bucket;
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
        vector<dtype> cpu_x(batch.size() * size());
        int batch_i = 0;
        int j = 0;
        for (Node *node : batch) {
            BucketNode *bucket = dynamic_cast<BucketNode*>(node);
            ys.at(batch_i++) = bucket->val().value;
            for (int i = 0; i < size(); ++i) {
                cpu_x.at(j++) = bucket->input_.at(i);
            }
        }
        cuda::BucketForward(cpu_x, count, size(), ys);
#if TEST_CUDA
        for (Node *node : batch) {
            BucketNode *bucket = dynamic_cast<BucketNode*>(node);
            dtype *v = node->val().v;
            for (int i = 0; i < size(); ++i) {
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
