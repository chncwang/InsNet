#ifndef DivOP
#define DivOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class FullDivNode : public Node, public Poolable<FullDivNode>  {
public:
    FullDivNode() : Node("full_div") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    bool typeEqual(Node *other) override {
        return getNodeType() == other->getNodeType();
    }

    string typeSignature() const override {
        return getNodeType();
    }

    void forward(Graph &graph, Node &numerator, Node &denominator) {
        if (getDim() != numerator.getDim() || getDim() != denominator.getDim()) {
            cerr << boost::format("dim:%1% minuend:%2% subtrahend:%3%") % getDim() %
                numerator.getDim() % denominator.getDim() << endl;
            abort();
        }
        numerator_ = &numerator;
        denominator_ = &denominator;
        vector<Node*> ins = {numerator_, denominator_};
        afterForward(graph, ins);
    }

    void compute() override {
        val().vec() = numerator_->getVal().vec() / denominator_->getVal().vec();
    }

    void backward() override {
        numerator_->loss().vec() += getLoss().vec() / denominator_->getVal().vec();
        denominator_->loss().vec() -= getLoss().vec() * numerator_->getVal().vec() /
            denominator_->getVal().vec().square();
    }

    PExecutor generate() override;

private:
    Node *numerator_;
    Node *denominator_;
    friend class FullDivExecutor;
};

#if USE_GPU
class FullDivExecutor : public Executor { // TODO
public:
    vector<dtype*> numerators;
    vector<dtype*> denominators;

    void forward() override {
        vector<dtype*> results;
        for (Node *node : batch) {
            FullDivNode *div = static_cast<FullDivNode*>(node);
            numerators.push_back(div->numerator_->getVal().value);
            denominators.push_back(div->denominator_->getVal().value);
            results.push_back(div->getVal().value);
        }

        n3ldg_cuda::FullDivForward(numerators, denominators, batch.size(), getDim(), results);
#if TEST_CUDA
        Executor::testForward();
        cout << "div tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype*> losses;
        vector<dtype*> numerator_losses, denominator_losses;
        for (Node *node : batch) {
            FullDivNode *div = static_cast<FullDivNode*>(node);
            losses.push_back(node->getLoss().value);
            numerator_losses.push_back(div->numerator_->getLoss().value);
            denominator_losses.push_back(div->denominator_->getLoss().value);
        }

        n3ldg_cuda::FullDivBackward(losses, denominators, numerators, batch.size(), getDim(),
                numerator_losses, denominator_losses);
#if TEST_CUDA
        auto get_inputs = [](Node &node) {
            FullDivNode &div = static_cast<FullDivNode&>(node);
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

namespace n3ldg_plus {
    Node *fullDiv(Graph &graph, Node &numerator, Node &denominator) {
        FullDivNode *result = FullDivNode::newNode(numerator.getDim());
        result->forward(graph, numerator, denominator);
        return result;
    }
}

#endif
