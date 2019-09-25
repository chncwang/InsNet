#ifndef SubOP
#define SubOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class SubNode : public Node {
public:
    SubNode() : Node("sub") {}

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
        cout << "after forward sub degree:" << getDegree() << endl;
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

    friend class SubExecutor;
};

#if USE_GPU
class SubExecutor : public Executor {
    void forward() override {
        vector<const dtype*> minuend, subtrahend;
        vector<dtype*> results;

        for (Node *node : batch) {
            SubNode *sub = static_cast<SubNode*>(node);
            minuend.push_back(sub->minuend_->getVal().value);
            subtrahend.push_back(sub->subtrahend_->getVal().value);
            results.push_back(sub->getVal().value);
            cout << boost::format("minuend:%1% subtrahend:%2% ") % sub->minuend_->getNodeType() %
                sub->subtrahend_->getNodeType() << endl;
            cout << "sub degree:" << sub->getDegree() << endl;
        }

        n3ldg_cuda::SubForward(minuend, subtrahend, batch.size(), getDim(), results);
#if TEST_CUDA
        testForward();
        cout << "sub tested" << endl;
#endif
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
