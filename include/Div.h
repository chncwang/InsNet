#ifndef DivOP
#define DivOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class DivNode : public Node {
public:
    DivNode() : Node("div_node") {}

    void forward(Graph &graph, Node &numerator, Node &denominator) {
        if (getDim() != numerator.getDim() || 1 != denominator.getDim()) {
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
        val().vec() = numerator_->getVal().vec() / denominator_->getVal()[0];
    }

    void backward() override {
        numerator_->loss().vec() += getLoss().vec() / denominator_->getVal()[0];

        dtype square = denominator_->getVal()[0] * denominator_->getVal()[0];

        for (int i = 0; i < getDim(); ++i) {
            denominator_->loss()[0] -= getLoss()[i] * numerator_->getVal()[i] / square;
        }
    }

    PExecutor generate() override;
private:
    Node *numerator_;
    Node *denominator_;
    friend class DivExecutor;
};

#if USE_GPU
class DivExecutor : public Executor {
    void forward() override {
        vector<const dtype*> numerators, denominators;
        vector<dtype*> results;
        for (Node *node : batch) {
            DivNode *div = static_cast<DivNode*>(node);
            numerators.push_back(div->numerator_->getVal().value);
            denominators.push_back(div->denominator_->getVal().value);
            results.push_back(div->getVal().value);
        }

        n3ldg_cuda::DivForwartd(numerators, denominators, batch.size(), getDim(), results);
#if TEST_CUDA
        Executor::testForward();
        cout << "div tested" << endl;
        abort();
#endif
    }
};
#else
class DivExecutor : public Executor {};
#endif

Executor *DivNode::generate() {
    DivExecutor * executor = new DivExecutor();
    return executor;
}

#endif
