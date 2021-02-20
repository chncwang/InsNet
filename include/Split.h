#ifndef N3LDG_PLUS_SPLIT_H
#define N3LDG_PLUS_SPLIT_H

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#if USE_GPU
#include "N3LDG_cuda.h"
#endif
#include <boost/format.hpp>

class SplitNode : public UniInputNode, public Poolable<SplitNode> {
public:
    SplitNode() : UniInputNode("split") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    string typeSignature() const override {
        return getNodeType();
    }

    void forward(Graph &graph, Node &input, int offset) {
        if (input.getDim() < offset + getDim()) {
            cerr << boost::format("input dim:%1% offset:%2% this dim:%3%") % input.getDim() %
                offset % getDim() << endl;
            abort();
        }

        offset_ = offset;
        UniInputNode::forward(graph, input);
    }

    Executor *generate() override;

    void compute () override {
        for (int i = 0; i < getDim(); ++i) {
            val()[i] = getInput().val()[i + offset_];
        }
    }

    void backward() override {
        for (int i = 0; i < getDim(); ++i) {
            getInput().loss()[i + offset_] += getLoss()[i];
        }
    }
protected:
    virtual bool isDimLegal(const Node &input) const override {
        return offset_ + getDim() <= input.getDim();
    }

private:
    int offset_ = 0;
    friend class SplitExecutor;
    friend class BatchedSplitNode;
};

class BatchedSplitNode : public BatchedNodeImpl<SplitNode> {
public:
    void init(Graph &graph, BatchedNode &input, int dim, int offset) {
        allocateBatch(dim, input.batch().size());
        for (Node *node : batch()) {
            SplitNode *s = dynamic_cast<SplitNode *>(node);
            s->offset_ = offset;
        }
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }

    void init(Graph &graph, BatchedNode &input, int dim, const vector<int> &offsets) {
        allocateBatch(dim, input.batch().size() * offsets.size());
        int i = 0;
        for (int offset : offsets) {
            for (Node *input_node : input.batch()) {
                if (offset + dim > input_node->getDim()) {
                    cerr << boost::format("offset:%1% dim:%2% input dim:%3%\n") % offset % dim %
                        input_node->getDim();
                    abort();
                }
                SplitNode *s = dynamic_cast<SplitNode *>(batch().at(i++));
                s->offset_ = offset;
                s->setInputs({input_node});
            }
        }

        input.addParent(this);
        graph.addNode(this);
    }
};

namespace n3ldg_plus {

Node* split(Graph &graph, Node &input, int dim, int offset) {
    SplitNode *split = SplitNode::newNode(dim);
    split->forward(graph, input, offset);
    return split;
}

BatchedNode* split(Graph &graph, BatchedNode &input, int dim, int offset) {
    BatchedSplitNode *node = new BatchedSplitNode;
    node->init(graph, input, dim, offset);
    return node;
}

BatchedNode *split(Graph &graph, BatchedNode &input, int dim, const vector<int> &offsets) {
    BatchedSplitNode *node = new BatchedSplitNode;
    node->init(graph, input, dim, offsets);
    return node;
}

}

#if USE_GPU
class SplitExecutor : public Executor {
public:
    void forward() override {
        vector<dtype*> inputs;
        vector<dtype*> results;
        for (Node *node : batch) {
            SplitNode *split = static_cast<SplitNode*>(node);
            inputs.push_back(split->getInput().getVal().value);
            offsets.push_back(split->offset_);
            results.push_back(split->getVal().value);
            dims.push_back(split->getDim());
        }
        n3ldg_cuda::SplitForward(inputs, offsets, batch.size(), dims, results);
#if TEST_CUDA
        testForward();
        cout << "split tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype*> losses;
        vector<dtype *> input_losses;

        for (Node *node : batch) {
            SplitNode *split = static_cast<SplitNode*>(node);
            losses.push_back(split->getLoss().value);
            input_losses.push_back(split->getInput().getLoss().value);
        }

        n3ldg_cuda::SplitBackward(losses, offsets, batch.size(), dims, input_losses);
#if TEST_CUDA
        auto get_inputs = [](Node &node) {
            SplitNode &split = static_cast<SplitNode&>(node);
            vector<pair<Node *, string>> inputs = {make_pair(&split.getInput(), "input")};
            return inputs;
        };
        Executor::testBackward(get_inputs);
        cout << "split backward tested" << endl;
#endif
    }

private:
        vector<int> offsets;
        vector<int> dims;
};
#else
class SplitExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }

    int calculateActivations() override {
        return 0;
    }
};
#endif

Executor *SplitNode::generate() {
    SplitExecutor * executor = new SplitExecutor;
    executor->batch.push_back(this);
    return executor;
}

#endif
