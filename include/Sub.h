#ifndef SubOP
#define SubOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class SubNode : public AtomicNode, public Poolable<SubNode> {
public:
    SubNode() : AtomicNode("sub") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    string typeSignature() const override {
        return getNodeType();
    }

    void forward(Graph &graph, AtomicNode &minuend, AtomicNode &subtrahend) {
        if (getDim() != minuend.getDim() || getDim() != subtrahend.getDim()) {
            cerr << boost::format("dim:%1% minuend:%2% subtrahend:%3%") % getDim() %
                minuend.getDim() % subtrahend.getDim() << endl;
            abort();
        }
        minuend_ = &minuend;
        subtrahend_ = &subtrahend;
        vector<AtomicNode*> ins = {minuend_, subtrahend_};
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
    AtomicNode *minuend_;
    AtomicNode *subtrahend_;

    friend class SubExecutor;
};

namespace n3ldg_plus {

AtomicNode *sub(Graph &graph, AtomicNode &minuend, AtomicNode &subtrahend) {
    SubNode *result = SubNode::newNode(minuend.getDim());
    result->forward(graph, minuend, subtrahend);
    return result;
}

}

#if USE_GPU
class SubExecutor : public Executor {
    void forward() override {
        vector<dtype*> minuend, subtrahend;
        vector<dtype*> results;

        for (AtomicNode *node : batch) {
            SubNode *sub = static_cast<SubNode*>(node);
            minuend.push_back(sub->minuend_->getVal().value);
            subtrahend.push_back(sub->subtrahend_->getVal().value);
            results.push_back(sub->getVal().value);
            dims_.push_back(node->getDim());
        }

        n3ldg_cuda::SubForward(minuend, subtrahend, batch.size(), dims_, results);
#if TEST_CUDA
        testForward();
        cout << "sub forward tested" << endl;
#endif
    }

    void backward() override {
        std::vector<dtype*> losses;
        std::vector<dtype*> minuend_losses, subtrahend_losses;
        for (AtomicNode *n : batch) {
            SubNode *sub = static_cast<SubNode*>(n);
            losses.push_back(sub->loss().value);
            minuend_losses.push_back(sub->minuend_->loss().value);
            subtrahend_losses.push_back(sub->subtrahend_->loss().value);
        }
#if TEST_CUDA
        auto get_inputs = [](AtomicNode &node) {
            SubNode &sub = static_cast<SubNode&>(node);
            vector<pair<AtomicNode*, string>> inputs = {make_pair(sub.minuend_, "minuend"),
                make_pair(sub.subtrahend_, "subtrahend")};
            return inputs;
        };
        cout << "test before sub backward..." << endl;
        testBeforeBackward(get_inputs);
#endif
        int count = batch.size();
        n3ldg_cuda::SubBackward(losses, count, dims_, minuend_losses, subtrahend_losses);
#if TEST_CUDA
        cout << "test sub backward..." << endl;
        Executor::testBackward(get_inputs);
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

#endif
