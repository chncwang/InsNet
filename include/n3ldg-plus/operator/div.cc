#include "n3ldg-plus/operator/div.h"

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;

namespace n3ldg_plus {

namespace {

const int NUMERATOR = 0;
const int DENOMINATOR = 1;

}

class FullDivNode : public Node, public Poolable<FullDivNode>  {
public:
    FullDivNode() : Node("full_div") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    string typeSignature() const override {
        return getNodeType();
    }

    void setInputs(const vector<Node *> &inputs) override {
        if (size() != inputs.at(0)->size() || size() != inputs.at(1)->size()) {
            cerr << fmt::format("dim:{} minuend:{} subtrahend:{}\n", size(),
                inputs.at(0)->size(), inputs.at(1)->size());
            abort();
        }

        Node::setInputs(inputs);
    }

    void connect(Node &numerator, Node &denominator) {
        vector<Node*> ins = {&numerator, &denominator};
        setInputs(ins);
        afterConnect(ins);
    }

    void compute() override {
        val().vec() = input_vals_.at(NUMERATOR)->vec() / input_vals_.at(DENOMINATOR)->vec();
    }

    void backward() override {
        input_grads_.at(NUMERATOR)->vec() += getGrad().vec() / input_vals_.at(DENOMINATOR)->vec();
        input_grads_.at(DENOMINATOR)->vec() -= getGrad().vec() * input_vals_.at(NUMERATOR)->vec() /
            input_vals_.at(DENOMINATOR)->vec().square();
    }

    Executor* generate() override;

protected:
    int forwardOnlyInputValSize() override {
        return 0;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    friend class FullDivExecutor;
};

class BatchedFullDivNode : public BatchedNodeImpl<FullDivNode> {
public:
    void init(BatchedNode &numerator, BatchedNode &denominator) {
        allocateBatch(numerator.sizes());
        auto ins = {&numerator, &denominator};
        setInputsPerNode(ins);
        afterInit(ins);
    }
};

#if USE_GPU
class FullDivExecutor : public Executor {
public:
    void forward() override {
        vector<dtype*> results;
        results.reserve(batch.size());
        numerators.reserve(batch.size());
        denominators.reserve(batch.size());
        dims.reserve(batch.size());
        for (Node *node : batch) {
            FullDivNode *div = dynamic_cast<FullDivNode*>(node);
            numerators.push_back(div->input_vals_.at(NUMERATOR)->value);
            denominators.push_back(div->input_vals_.at(DENOMINATOR)->value);
            results.push_back(div->getVal().value);
            dims.push_back(node->size());
        }

        cuda::FullDivForward(numerators, denominators, batch.size(), dims, results);
#if TEST_CUDA
        Executor::testForward();
        cout << "div tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype*> losses;
        vector<dtype*> numerator_losses, denominator_losses;
        losses.reserve(batch.size());
        numerator_losses.reserve(batch.size());
        denominator_losses.reserve(batch.size());
        for (Node *node : batch) {
            FullDivNode *div = dynamic_cast<FullDivNode*>(node);
            losses.push_back(node->getGrad().value);
            numerator_losses.push_back(div->input_grads_.at(NUMERATOR)->value);
            denominator_losses.push_back(div->input_grads_.at(DENOMINATOR)->value);
        }

        cuda::FullDivBackward(losses, denominators, numerators, batch.size(), dims,
                numerator_losses, denominator_losses);
#if TEST_CUDA
        Executor::testBackward();
        cout << "div backward tested" << endl;
#endif
    }

private:
    vector<dtype*> numerators;
    vector<dtype*> denominators;
    vector<int> dims;
};
#else
class FullDivExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return defaultFLOPs();
    }
};
#endif

Executor *FullDivNode::generate() {
    return new FullDivExecutor();
}

Node *fullDiv(Node &numerator, Node &denominator) {
    FullDivNode *result = FullDivNode::newNode(numerator.size());
    result->connect(numerator, denominator);
    return result;
}

BatchedNode *fullDiv(BatchedNode &numerator, BatchedNode &denominator) {
    BatchedFullDivNode *node = new BatchedFullDivNode;
    node->init(numerator, denominator);
    return node;
}

}
