#include "n3ldg-plus/operator/split.h"

using std::string;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using std::pair;
using std::make_pair;

namespace n3ldg_plus {

class SplitNode : public UniInputNode, public Poolable<SplitNode> {
public:
    SplitNode() : UniInputNode("split") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    string typeSignature() const override {
        return getNodeType();
    }

    void connect(Node &input, int offset) {
        if (input.size() < offset + size()) {
            cerr << fmt::format("input dim:{} offset:{} this dim:{}\n", input.size(),
                offset, size());
            abort();
        }

        offset_ = offset;
        UniInputNode::connect(input);
    }

    Executor *generate() override;

    void compute () override {
        int row = size() / getColumn();
        int in_row = inputDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            Vec(val().v + i * row, row) = Vec(inputVal().v + i * in_row + offset_, row);
        }
    }

    void backward() override {
        int row = size() / getColumn();
        int in_row = inputDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            Vec(inputGrad().v + i * in_row + offset_, row) += Vec(getGrad().v + i * row, row);
        }
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return offset_ + size() <= input.size();
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    int offset_ = 0;
    friend class SplitExecutor;
    friend class BatchedSplitNode;
};

class BatchedSplitNode : public BatchedNodeImpl<SplitNode> {
public:
    void init(BatchedNode &input, int dim, int offset) {
        allocateBatch(dim, input.batch().size());
        for (Node *node : batch()) {
            SplitNode *s = dynamic_cast<SplitNode *>(node);
            s->offset_ = offset;
        }
        setInputsPerNode({&input});
        afterInit({&input});
    }

    void init(BatchedNode &input, int dim, const vector<int> &offsets) {
        allocateBatch(dim, input.batch().size() * offsets.size());
        int i = 0;
        for (int offset : offsets) {
            for (Node *input_node : input.batch()) {
                if (offset + dim > input_node->size()) {
                    cerr << fmt::format("offset:{} dim:{} input dim:{}\n", offset, dim,
                        input_node->size());
                    abort();
                }
                SplitNode *s = dynamic_cast<SplitNode *>(batch().at(i++));
                s->offset_ = offset;
                s->setInputs({input_node});
            }
        }

        afterInit({&input});
    }

    void init(Node &input, int row, const vector<int> &offsets, int col = 1) {
        allocateBatch(row * col, offsets.size());
        int i = 0;
        for (int offset : offsets) {
            SplitNode *s = dynamic_cast<SplitNode *>(batch().at(i++));
            s->setColumn(col);
            s->offset_ = offset;
            s->setInputs({&input});
        }

        input.addParent(this);
        NodeContainer &graph = input.getNodeContainer();
        graph.addNode(this);
    }
};

Node* split(Node &input, int dim, int offset, int col) {
    SplitNode *split = SplitNode::newNode(dim * col);
    split->setColumn(col);
    split->connect(input, offset);
    return split;
}

BatchedNode* split(BatchedNode &input, int dim, int offset) {
    BatchedSplitNode *node = new BatchedSplitNode;
    node->init(input, dim, offset);
    return node;
}

BatchedNode *split(BatchedNode &input, int dim, const vector<int> &offsets) {
    BatchedSplitNode *node = new BatchedSplitNode;
    node->init(input, dim, offsets);
    return node;
}

BatchedNode *split(Node &input, int row, const vector<int> &offsets, int col) {
    BatchedSplitNode *node = new BatchedSplitNode;
    node->init(input, row, offsets, col);
    return node;
}

#if USE_GPU
class SplitExecutor : public Executor {
public:
    void forward() override {
        int count = batch.size();
        vector<dtype*> inputs;
        vector<dtype*> results;

        inputs.reserve(count);
        results.reserve(count);
        offsets_.reserve(count);
        rows_.reserve(count);
        in_rows_.reserve(count);
        cols_.reserve(count);

        for (Node *node : batch) {
            SplitNode &split = dynamic_cast<SplitNode &>(*node);
            inputs.push_back(split.inputVal().value);
            offsets_.push_back(split.offset_);
            results.push_back(split.getVal().value);
            int col = split.getColumn();
            cols_.push_back(col);
            rows_.push_back(split.size() / col);
            in_rows_.push_back(split.inputDim() / col);
        }
        cuda::SplitForward(inputs, offsets_, count, rows_, in_rows_, cols_, results);
#if TEST_CUDA
        testForward();
        cout << "split tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype*> grads;
        vector<dtype *> input_grads;

        for (Node *node : batch) {
            SplitNode *split = static_cast<SplitNode*>(node);
            grads.push_back(split->getGrad().value);
            input_grads.push_back(split->inputGrad().value);
        }

        cuda::SplitBackward(grads, offsets_, batch.size(), rows_, in_rows_, cols_,
                input_grads);
#if TEST_CUDA
        Executor::testBackward();
        cout << "split backward tested" << endl;
#endif
    }

private:
        vector<int> offsets_;
        vector<int> rows_;
        vector<int> in_rows_;
        vector<int> cols_;
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
    return new SplitExecutor;
}

}
