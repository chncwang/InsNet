#include "n3ldg-plus/operator/broadcast.h"

using std::string;
using std::to_string;
using std::vector;

namespace n3ldg_plus {

class BroadcastNode : public UniInputNode, public Poolable<BroadcastNode> {
public:
    BroadcastNode() : UniInputNode("broadcast") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    Executor *generate() override;

    void compute() override {
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            Vec(val().v + row * i, row) = getInput().getVal().vec();
        }
    }

    void backward() override {
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            getInput().loss().vec() += Vec(loss().v + row * i, row);
        }
    }

    string typeSignature() const override {
        return Node::getNodeType() + to_string(getDim() / getColumn());
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return getColumn() * input.getDim() == getDim();
    }
};

#if USE_GPU

class BroadcastExecutor : public UniInputExecutor {
public:
    void forward() override {
        vector<dtype *> in_vals, vals;
        vector<int> ns;
        int count = batch.size();
        in_vals.reserve(count);
        vals.reserve(count);
        ns.reserve(count);

        for (Node *node : batch) {
            BroadcastNode &b = dynamic_cast<BroadcastNode &>(*node);
            in_vals.push_back(b.getInput().getVal().value);
            vals.push_back(b.getVal().value);
            ns.push_back(b.getColumn());
            in_dim_ = b.getDim() /  b.getColumn();
        }
        cuda::NumberPointerArray in_val_arr, val_arr;
        in_val_arr.init(in_vals.data(), count);
        val_arr.init(vals.data(), count);
        ns_arr_.init(ns.data(), count);
        max_n_ = *max_element(ns.begin(), ns.end());
        cuda::BroadcastForward(in_val_arr.value, count, in_dim_, ns_arr_.value, max_n_,
                val_arr.value);
#if TEST_CUDA
        UniInputExecutor::testForward();
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype *> grads, in_grads;
        grads.reserve(count);
        in_grads.reserve(count);
        for (Node *node : batch) {
            BroadcastNode &b = dynamic_cast<BroadcastNode &>(*node);
            grads.push_back(b.getLoss().value);
            in_grads.push_back(b.getInput().getLoss().value);
        }

        cuda::NumberPointerArray grad_arr, in_grad_arr;
        grad_arr.init(grads.data(), count);
        in_grad_arr.init(in_grads.data(), count);
        cuda::BroadcastBackward(grad_arr.value, count, in_dim_, ns_arr_.value, max_n_,
                in_grad_arr.value);
#if TEST_CUDA
        UniInputExecutor::testBackward();
#endif
    }

private:
    cuda::IntArray ns_arr_;
    int max_n_;
    int in_dim_;
};

#else

class BroadcastExecutor : public UniInputExecutor {
public:
    int calculateFLOPs() override {
        return 0; // TODO
    }
};

#endif

Executor *BroadcastNode::generate() {
    return new BroadcastExecutor;
}

Node *broadcast(Node &input, int count) {
    BroadcastNode *result = BroadcastNode::newNode(input.getDim() * count);
    result->setColumn(count);
    result->connect(input);
    return result;
}

}
