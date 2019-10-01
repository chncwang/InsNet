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
        }

        n3ldg_cuda::SubForward(minuend, subtrahend, batch.size(), getDim(), results);
#if TEST_CUDA
        testForward();
        cout << "sub forward tested" << endl;
#endif
    }

    void backward() override {
        std::vector<const dtype*> losses;
        std::vector<dtype*> minuend_losses, subtrahend_losses;
        for (Node *n : batch) {
            SubNode *sub = static_cast<SubNode*>(n);
            losses.push_back(sub->loss().value);
            minuend_losses.push_back(sub->minuend_->loss().value);
            subtrahend_losses.push_back(sub->subtrahend_->loss().value);
        }
        int count = batch.size();
        n3ldg_cuda::SubBackward(losses, count, getDim(), minuend_losses, subtrahend_losses);
#if TEST_CUDA
        auto get_inputs = [](Node &node) {
            SubNode &sub = static_cast<SubNode&>(node);
            vector<pair<Node*, string>> inputs = {make_pair(sub.minuend_, "minuend"),
                make_pair(sub.subtrahend_, "subtrahend")};
            return inputs;
        };
        Executor::testBackward(get_inputs);
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
