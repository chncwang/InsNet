#ifndef DivOP
#define DivOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class FullDivNode : public AtomicNode, public Poolable<FullDivNode>  {
public:
    FullDivNode() : AtomicNode("full_div") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    string typeSignature() const override {
        return getNodeType();
    }

    void forward(Graph &graph, AtomicNode &numerator, AtomicNode &denominator) {
        if (getDim() != numerator.getDim() || getDim() != denominator.getDim()) {
            cerr << boost::format("dim:%1% minuend:%2% subtrahend:%3%") % getDim() %
                numerator.getDim() % denominator.getDim() << endl;
            abort();
        }
        numerator_ = &numerator;
        denominator_ = &denominator;
        vector<AtomicNode*> ins = {numerator_, denominator_};
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
    AtomicNode *numerator_;
    AtomicNode *denominator_;
    friend class FullDivExecutor;
};

#if USE_GPU
class FullDivExecutor : public Executor {
public:
    vector<dtype*> numerators;
    vector<dtype*> denominators;
    vector<int> dims;

    void forward() override {
        vector<dtype*> results;
        for (AtomicNode *node : batch) {
            FullDivNode *div = static_cast<FullDivNode*>(node);
            numerators.push_back(div->numerator_->getVal().value);
            denominators.push_back(div->denominator_->getVal().value);
            results.push_back(div->getVal().value);
            dims.push_back(node->getDim());
        }

        n3ldg_cuda::FullDivForward(numerators, denominators, batch.size(), dims, results);
#if TEST_CUDA
        Executor::testForward();
        cout << "div tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype*> losses;
        vector<dtype*> numerator_losses, denominator_losses;
        for (AtomicNode *node : batch) {
            FullDivNode *div = static_cast<FullDivNode*>(node);
            losses.push_back(node->getLoss().value);
            numerator_losses.push_back(div->numerator_->getLoss().value);
            denominator_losses.push_back(div->denominator_->getLoss().value);
        }

        n3ldg_cuda::FullDivBackward(losses, denominators, numerators, batch.size(), dims,
                numerator_losses, denominator_losses);
#if TEST_CUDA
        auto get_inputs = [](AtomicNode &node) {
            FullDivNode &div = static_cast<FullDivNode&>(node);
            vector<pair<AtomicNode*, string>> results = {make_pair(div.denominator_, "denominator"),
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
    AtomicNode *fullDiv(Graph &graph, AtomicNode &numerator, AtomicNode &denominator) {
        FullDivNode *result = FullDivNode::newNode(numerator.getDim());
        result->forward(graph, numerator, denominator);
        return result;
    }
}

#endif
