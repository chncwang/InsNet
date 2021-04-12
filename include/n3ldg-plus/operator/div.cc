#include "n3ldg-plus/operator/div.h"

using std::string;
using std::vector;
using std::cerr;

namespace n3ldg_plus {

class FullDivNode : public Node, public Poolable<FullDivNode>  {
public:
    FullDivNode() : Node("full_div") {}

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
        if (getDim() != inputs.at(0)->getDim() || getDim() != inputs.at(1)->getDim()) {
            cerr << fmt::format("dim:{} minuend:{} subtrahend:{}\n", getDim(),
                inputs.at(0)->getDim(), inputs.at(1)->getDim());
            abort();
        }
        numerator_ = inputs.at(0);
        denominator_ = inputs.at(1);
    }

    void connect(Node &numerator, Node &denominator) {
        vector<Node*> ins = {&numerator, &denominator};
        setInputs(ins);
        afterConnect(ins);
    }

    void compute() override {
        val().vec() = numerator_->getVal().vec() / denominator_->getVal().vec();
    }

    void backward() override {
        numerator_->loss().vec() += getLoss().vec() / denominator_->getVal().vec();
        denominator_->loss().vec() -= getLoss().vec() * numerator_->getVal().vec() /
            denominator_->getVal().vec().square();
    }

    Executor* generate() override;

private:
    Node *numerator_;
    Node *denominator_;
    friend class FullDivExecutor;
};

class BatchedFullDivNode : public BatchedNodeImpl<FullDivNode> {
public:
    void init(BatchedNode &numerator, BatchedNode &denominator) {
        allocateBatch(numerator.getDims());
        auto ins = {&numerator, &denominator};
        setInputsPerNode(ins);
        afterInit(ins);
    }
};

#if USE_GPU
class FullDivExecutor : public Executor {
public:
    vector<dtype*> numerators;
    vector<dtype*> denominators;
    vector<int> dims;

    void forward() override {
        vector<dtype*> results;
        results.reserve(batch.size());
        numerators.reserve(batch.size());
        denominators.reserve(batch.size());
        dims.reserve(batch.size());
        for (Node *node : batch) {
            FullDivNode *div = dynamic_cast<FullDivNode*>(node);
            numerators.push_back(div->numerator_->getVal().value);
            denominators.push_back(div->denominator_->getVal().value);
            results.push_back(div->getVal().value);
            dims.push_back(node->getDim());
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
            losses.push_back(node->getLoss().value);
            numerator_losses.push_back(div->numerator_->getLoss().value);
            denominator_losses.push_back(div->denominator_->getLoss().value);
        }

        cuda::FullDivBackward(losses, denominators, numerators, batch.size(), dims,
                numerator_losses, denominator_losses);
#if TEST_CUDA
        auto get_inputs = [](Node &node) {
            FullDivNode &div = dynamic_cast<FullDivNode&>(node);
            vector<pair<Node*, string>> results = {make_pair(div.denominator_, "denominator"),
                    make_pair(div.numerator_, "numerator")};
            return results;
        };
        Executor::testBackward(get_inputs);
        cout << "div backward tested" << endl;
#endif
    }
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
    FullDivNode *result = FullDivNode::newNode(numerator.getDim());
    result->connect(numerator, denominator);
    return result;
}

BatchedNode *fullDiv(BatchedNode &numerator, BatchedNode &denominator) {
    BatchedFullDivNode *node = new BatchedFullDivNode;
    node->init(numerator, denominator);
    return node;
}

}
