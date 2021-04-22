#include "n3ldg-plus/operator/concat.h"

using std::vector;
using std::string;
using std::to_string;
using std::cerr;
using std::endl;

namespace n3ldg_plus {

class ConcatNode : public Node, public Poolable<ConcatNode> {
public:
    ConcatNode() : Node("concat") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void clear() override {
        in_rows_.clear();
        ins_.clear();
        Node::clear();
    }

    void setInputs(const vector<Node *> &ins) override {
        int in_size = ins.size();
        int cur_dim = 0;
        in_rows_.reserve(in_size);
        for (int i = 0; i < in_size; ++i) {
            in_rows_.push_back(ins.at(i)->getDim() / getColumn());
            cur_dim += in_rows_.at(i);
        }
        if (cur_dim * getColumn() != getDim()) {
            cerr << "input dim size not match" << cur_dim << "\t" << getDim() << endl;
            abort();
        }
        ins_ = ins;
    }

    void connect(const vector<Node *> &x) {
        if (x.empty()) {
            cerr << "empty inputs for concat" << endl;
            abort();
        }

        setInputs(x);
        afterConnect(x);
    }

    Executor* generate() override;

    string typeSignature() const override {
        string hash_code = Node::getNodeType() + "-" + to_string(in_rows_.size());
        for (int dim : in_rows_) {
            hash_code += "-" + to_string(dim);
        }
        return hash_code;
    }

    void compute() override {
        int in_size = ins_.size();
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            int offset = 0;
            for (int j = 0; j < in_size; ++j) {
                Vec(val().v + i * row + offset, in_rows_.at(j)) =
                    Vec(ins_.at(j)->val().v + i * in_rows_.at(j), in_rows_.at(j));
                offset += in_rows_[j];
            }
        }
    }

    void backward() override {
        int in_size = ins_.size();
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            int offset = 0;
            for (int j = 0; j < in_size; ++j) {
                Vec(ins_[j]->loss().v + i * in_rows_.at(j), in_rows_.at(j)) +=
                    Vec(getLoss().v + i * row + offset, in_rows_.at(j));
                offset += in_rows_[j];
            }
        }
    }

private:
    vector<int> in_rows_;
    vector<Node *> ins_;

    friend class ConcatExecutor;
};

#if USE_GPU
class ConcatExecutor : public Executor {
public:
    void forward() override {
        int count = batch.size();

        vector<dtype*> in_vals, vals;
        in_vals.reserve(inCount() * count);
        vals.reserve(count);
        cols_.reserve(count);
        for (Node *node : batch) {
            ConcatNode *concat = dynamic_cast<ConcatNode*>(node);
            for (Node *in : concat->ins_) {
                in_vals.push_back(in->val().value);
            }
            vals.push_back(node->getVal().value);
            cols_.push_back(concat->getColumn());
        }

        ConcatNode &first = dynamic_cast<ConcatNode &>(*batch.front());
        row_ = first.getDim() / first.getColumn();
        cuda::ConcatForward(in_vals, dynamic_cast<ConcatNode*>(batch.at(0))->in_rows_, vals,
                count, inCount(), row_, cols_);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            cuda::Assert(batch[idx]->val().verify("concat forward"));
        }
        cout << "concat forward tested" << endl;
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype*> in_losses, losses;
        in_losses.reserve(inCount() * count);
        losses.reserve(count);
        for (Node *node : batch) {
            ConcatNode *concat = dynamic_cast<ConcatNode*>(node);
            for (Node *in : concat->ins_) {
                in_losses.push_back(in->loss().value);
            }
            losses.push_back(node->loss().value);
        }

        cuda::ConcatBackward(in_losses, dynamic_cast<ConcatNode*>(batch.at(0))->in_rows_,
                losses, count, inCount(), row_, cols_);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }
        for (int idx = 0; idx < count; idx++) {
            for (int j = 0; j < inCount(); ++j) {
                cuda::Assert(dynamic_cast<ConcatNode *>(batch[idx])->
                        ins_.at(j)->loss().verify("concat backward"));
            }
        }
#endif
    }

private:
    int inCount() {
        return dynamic_cast<ConcatNode *>(batch.front())->ins_.size();
    }

    vector<int> cols_;
    int row_;
};
#else
class ConcatExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }

    int calculateActivations() override {
        return 0;
    }
};
#endif

Executor* ConcatNode::generate() {
    return new ConcatExecutor();
}

class BatchedConcatNode : public BatchedNodeImpl<ConcatNode> {
public:
    void init(const vector<BatchedNode *> &ins) {
        int dim = 0;
        for (BatchedNode *node : ins) {
            dim += node->getDim();
        }
        allocateBatch(dim, ins.front()->batch().size());
        setInputsPerNode(ins);
        afterInit(ins);
    }
};

Node *concat(const vector<Node*> &inputs, int col) {
    int dim = 0;
    for (Node *in : inputs) {
        dim += in->getDim();
    }
    ConcatNode *concat = ConcatNode::newNode(dim);
    concat->setColumn(col);
    concat->connect(inputs);
    return concat;
}

Node *concat(BatchedNode &inputs, int col) {
    int dim = 0;
    for (Node *in : inputs.batch()) {
        dim += in->getDim();
    }
    ConcatNode *concat = ConcatNode::newNode(dim, col == 1);
    concat->setColumn(col);
    concat->setInputs(inputs.batch());
    inputs.addParent(concat);
    NodeContainer &container = inputs.getNodeContainer();
    container.addNode(concat);
    return concat;
}

BatchedNode *concatInBatch(const vector<BatchedNode *> &inputs) {
    BatchedConcatNode *node = new BatchedConcatNode;
    node->init(inputs);
    return node;
}

}
