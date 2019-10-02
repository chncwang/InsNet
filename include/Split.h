#ifndef N3LDG_PLUS_SPLIT_H
#define N3LDG_PLUS_SPLIT_H

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#if USE_GPU
#include "N3LDG_cuda.h"
#endif
#include <boost/format.hpp>

class SplitNode : public Node {
public:
    SplitNode() : Node("split") {}

    void forward(Graph &graph, Node &input, int offset) {
        if (input.getDim() < offset + getDim()) {
            cerr << boost::format("input dim:%1% offset:%2% this dim:%3%") % input.getDim() %
                offset % getDim() << endl;
            abort();
        }

        offset_ = offset;
        input_ = &input;
        input.addParent(this);

        graph.addNode(this);
    }

    Executor *generate() override;

    void compute () override {
        for (int i = 0; i < getDim(); ++i) {
            val()[i] = input_->val()[i + offset_];
        }
    }

    void backward() override {
        for (int i = 0; i < getDim(); ++i) {
            input_->loss()[i + offset_] += getLoss()[i];
        }
    }

private:
    Node *input_ = nullptr;
    int offset_ = 0;
    friend class SplitExecutor;
};

namespace n3ldg_plus {
Node* split(Graph &graph, int dim, Node &input, int offset) {
    SplitNode *split = new SplitNode;
    split->init(dim);
    split->forward(graph, input, offset);
    return split;
}
}

#if USE_GPU
class SplitExecutor : public Executor {
public:
    void forward() override {
        vector<const dtype*> inputs;
        vector<dtype*> results;
        for (Node *node : batch) {
            SplitNode *split = static_cast<SplitNode*>(node);
            inputs.push_back(split->input_->getVal().value);
            offsets.push_back(split->offset_);
            results.push_back(split->getVal().value);
        }
        n3ldg_cuda::SplitForward(inputs, offsets, batch.size(), getDim(), results);
#if TEST_CUDA
        testForward();
        cout << "split tested" << endl;
#endif
    }

    void backward() override {
        vector<const dtype*> losses;
        vector<dtype *> input_losses;

        for (Node *node : batch) {
            SplitNode *split = static_cast<SplitNode*>(node);
            losses.push_back(split->getLoss().value);
            input_losses.push_back(split->input_->getLoss().value);
        }

        n3ldg_cuda::SplitBackward(losses, offsets, batch.size(), getDim(), input_losses);
#if TEST_CUDA
        auto get_inputs = [](Node &node) {
            SplitNode &split = static_cast<SplitNode&>(node);
            vector<pair<Node *, string>> inputs = {make_pair(split.input_, "input")};
            return inputs;
        };
        Executor::testBackward(get_inputs);
        cout << "split backward tested" << endl;
#endif
    }

private:
        vector<int> offsets;
};
#else
class SplitExecutor : public Executor {};
#endif

Executor *SplitNode::generate() {
    SplitExecutor * executor = new SplitExecutor;
    executor->batch.push_back(this);
    return executor;
}

#endif
