#include "n3ldg-plus/operator/sub.h"

using std::string;
using std::vector;
using std::cerr;
using std::pair;
using std::make_pair;

namespace n3ldg_plus {

class SubNode : public Node, public Poolable<SubNode> {
public:
    SubNode() : Node("sub") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    string typeSignature() const override {
        return getNodeType();
    }

    void setInputs(const vector<Node *> &inputs) override {
        Node &minuend = *inputs.at(0);
        Node &subtrahend = *inputs.at(1);
        if (getDim() != minuend.getDim() || getDim() != subtrahend.getDim()) {
            cerr << fmt::format("dim:{} minuend:{} subtrahend:{}\n", getDim(),
                minuend.getDim(), subtrahend.getDim());
            abort();
        }
        minuend_ = &minuend;
        subtrahend_ = &subtrahend;
    }

    void connect(Node &minuend, Node &subtrahend) {
        vector<Node*> ins = {&minuend, &subtrahend};
        setInputs(ins);
        afterConnect(ins);
    }

    void compute() override {
        val().vec() = minuend_->getVal().vec() - subtrahend_->getVal().vec();
    }

    void backward() override {
        minuend_->loss().vec() += loss().vec();
        subtrahend_->loss().vec() -= loss().vec();
    }

    Executor *generate() override;

private:
    Node *minuend_;
    Node *subtrahend_;

    friend class SubExecutor;
    friend vector<pair<Node *, string>> getInput(Node &node);
};

class BatchedSubNode : public BatchedNodeImpl<SubNode> {
public:
    void init(BatchedNode &minuend, BatchedNode &subtrahend) {
        allocateBatch(minuend.getDims());
        setInputsPerNode({&minuend, &subtrahend});
        afterInit({&minuend, &subtrahend});
    }
};

Node *sub(Node &minuend, Node &subtrahend) {
    SubNode *result = SubNode::newNode(minuend.getDim());
    result->connect(minuend, subtrahend);
    return result;
}

BatchedNode *sub(BatchedNode &minuend, BatchedNode &subtrahend) {
    BatchedSubNode *node = new BatchedSubNode;
    node->init(minuend, subtrahend);
    return node;
}

vector<pair<Node *, string>> getInput(Node &node) {
    SubNode &sub = dynamic_cast<SubNode&>(node);
    vector<pair<Node*, string>> inputs = {make_pair(sub.minuend_, "minuend"),
        make_pair(sub.subtrahend_, "subtrahend")};
    return inputs;
}

#if USE_GPU
class SubExecutor : public Executor {
    void forward() override {
        vector<dtype*> minuend, subtrahend;
        vector<dtype*> results;
#if TEST_CUDA
        testForwardInpputs(getInput);
        for (Node *node : batch) {
            SubNode *sub = static_cast<SubNode*>(node);
            sub->minuend_->val().copyFromHostToDevice();
            sub->subtrahend_->val().copyFromHostToDevice();
        }
#endif

        for (Node *node : batch) {
            SubNode *sub = static_cast<SubNode*>(node);
            minuend.push_back(sub->minuend_->getVal().value);
            subtrahend.push_back(sub->subtrahend_->getVal().value);
            results.push_back(sub->getVal().value);
            dims_.push_back(node->getDim());
        }

        cuda::SubForward(minuend, subtrahend, batch.size(), dims_, results);
#if TEST_CUDA
        testForward();
        cout << "sub forward tested" << endl;
#endif
    }

    void backward() override {
        std::vector<dtype*> losses;
        std::vector<dtype*> minuend_losses, subtrahend_losses;
        for (Node *n : batch) {
            SubNode *sub = static_cast<SubNode*>(n);
            losses.push_back(sub->loss().value);
            minuend_losses.push_back(sub->minuend_->loss().value);
            subtrahend_losses.push_back(sub->subtrahend_->loss().value);
        }
#if TEST_CUDA
        cout << "test before sub backward..." << endl;
        testBeforeBackward(getInput);
#endif
        int count = batch.size();
        cuda::SubBackward(losses, count, dims_, minuend_losses, subtrahend_losses);
#if TEST_CUDA
        cout << "test sub backward..." << endl;
        Executor::testBackward(getInput);
        cout << "sub tested" << endl;
#endif
    }

private:
    vector<int> dims_;
};
#else
class SubExecutor : public Executor {
    int calculateFLOPs() override {
        return defaultFLOPs();
    }
};
#endif

Executor *SubNode::generate() {
    SubExecutor * executor = new SubExecutor();
    return executor;
}

}

