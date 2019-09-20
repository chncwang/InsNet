#ifndef SubOP
#define SubOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class SubNode : public Node {
public:
    SubNode() : Node("sub_node") {}

    void forward(Graph &graph, Node &minuend, Node &subtrahend) {
        if (getDim() != minuend.getDim() || getDim() != subtrahend.getDim()) {
            cerr << boost::format("dim:%1% minuend:%2% subtrahend:%3%") % getDim() %
                minuend.getDim() % subtrahend.getDim() << endl;
            abort();
        }
        minuend_ = &minuend;
        subtrahend_ = &subtrahend;
        vector<Node*> ins = {minuend_, subtrahend_};
        afterForward(graph, ins);
    }

    void compute() override {
        val().vec() = minuend_->getVal().vec() - subtrahend_->getVal().vec();
    }

    void backward() override {
        minuend_->loss().vec() += loss().vec();
        subtrahend_->loss().vec() -= loss().vec();
    }

    PExecutor generate() override;
private:
    Node *minuend_;
    Node *subtrahend_;
};

#if USE_GPU
class SubExecutor : public Executor {
    void forward() override {
        abort();
    }
};
#else
class SubExecutor : public Executor {};
#endif

Executor *SubNode::generate() {
    SubExecutor * executor = new SubExecutor();
    return executor;
}

#endif
