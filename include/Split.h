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

    bool typeEqual(Node *other) override {
        SplitNode *s = static_cast<SplitNode*>(other);
        return Node::typeEqual(other) && offset_ == s->offset_;
    }

    string typeSignature() const override {
        return Node::typeSignature() + "-" + to_string(offset_);
    }

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
};

namespace n3ldg_plus {
Node* split(Graph &graph, int dim, Node &input, int offset) {
    SplitNode *split = new SplitNode;
    split->init(dim);
    split->forward(graph, input, offset);
    return split;
}
}

class SplitExecutor : public Executor {};

Executor *SplitNode::generate() {
    SplitExecutor * executor = new SplitExecutor;
    executor->batch.push_back(this);
    return executor;
}

#endif
