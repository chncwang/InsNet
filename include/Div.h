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
};

class DivExecutor : public Executor {};

Executor *DivNode::generate() {
    DivExecutor * executor = new DivExecutor();
    return executor;
}

#endif
