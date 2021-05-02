#include "n3ldg-plus/operator/sub.h"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;

namespace n3ldg_plus {

constexpr int MINUEND = 0;
constexpr int SUBTRAHEND = 1;

class SubNode : public Node, public Poolable<SubNode> {
public:
    SubNode() : Node("sub") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    string typeSignature() const override {
        return getNodeType();
    }

    void setInputs(const vector<Node *> &inputs) override {
        Node &minuend = *inputs.at(MINUEND);
        Node &subtrahend = *inputs.at(SUBTRAHEND);
        if (getDim() != minuend.getDim() || getDim() != subtrahend.getDim()) {
            cerr << fmt::format("dim:{} minuend:{} subtrahend:{}\n", getDim(),
                minuend.getDim(), subtrahend.getDim());
            abort();
        }
        Node::setInputs(inputs);
    }

    void connect(Node &minuend, Node &subtrahend) {
        vector<Node*> ins = {&minuend, &subtrahend};
        setInputs(ins);
        afterConnect(ins);
    }

    void compute() override {
        val().vec() = input_vals_.at(MINUEND)->vec() - input_vals_.at(SUBTRAHEND)->vec();
    }

    void backward() override {
        input_grads_.at(MINUEND)->vec() += grad().vec();
        input_grads_.at(SUBTRAHEND)->vec() -= grad().vec();
    }

    Executor *generate() override;

protected:
    int forwardOnlyInputValSize() override {
        return 0;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
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

#if USE_GPU
class SubExecutor : public Executor {
    void forward() override {
        vector<dtype*> minuend, subtrahend;
        vector<dtype*> results;
#if TEST_CUDA
        testForwardInpputs();
        for (Node *node : batch) {
            SubNode *sub = static_cast<SubNode*>(node);
            for (Tensor1D *val : sub->input_vals_) {
                val->copyFromHostToDevice();
            }
        }
#endif

        for (Node *node : batch) {
            SubNode *sub = static_cast<SubNode*>(node);
            minuend.push_back(sub->input_vals_.at(MINUEND)->value);
            subtrahend.push_back(sub->input_vals_.at(SUBTRAHEND)->value);
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
            losses.push_back(sub->grad().value);
            minuend_losses.push_back(sub->input_grads_.at(MINUEND)->value);
            subtrahend_losses.push_back(sub->input_grads_.at(SUBTRAHEND)->value);
        }
#if TEST_CUDA
        cout << "test before sub backward..." << endl;
        testBeforeBackward();
#endif
        int count = batch.size();
        cuda::SubBackward(losses, count, dims_, minuend_losses, subtrahend_losses);
#if TEST_CUDA
        cout << "test sub backward..." << endl;
        Executor::testBackward();
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

