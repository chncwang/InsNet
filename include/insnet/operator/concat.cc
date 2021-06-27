#include "insnet/operator/concat.h"
#include "insnet/operator/matrix.h"
#include "insnet/util/util.h"

using std::vector;
using std::string;
using std::to_string;
using std::cerr;
using std::cout;
using std::endl;

namespace insnet {

class ConcatNode : public Node, public Poolable<ConcatNode> {
public:
    ConcatNode() : Node("concat") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void clear() override {
        in_rows_.clear();
        Node::clear();
    }

    void setInputs(const vector<Node *> &ins) override {
        int in_size = ins.size();
        int cur_dim = 0;
        in_rows_.reserve(in_size);
        for (int i = 0; i < in_size; ++i) {
            in_rows_.push_back(ins.at(i)->size() / getColumn());
            cur_dim += in_rows_.at(i);
        }
        if (cur_dim * getColumn() != size()) {
            cerr << "input dim size not match" << cur_dim << "\t" << size() << endl;
            abort();
        }

        Node::setInputs(ins);
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
        hash_code += "-" + to_string(getColumn());
        return hash_code;
    }

    void compute() override {
        int in_size = input_vals_.size();
        int row = size() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            int offset = 0;
            for (int j = 0; j < in_size; ++j) {
                Vec(val().v + i * row + offset, in_rows_.at(j)) =
                    Vec(input_vals_.at(j)->v + i * in_rows_.at(j), in_rows_.at(j));
                offset += in_rows_[j];
            }
        }
    }

    void backward() override {
        int in_size = input_vals_.size();
        int row = size() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            int offset = 0;
            for (int j = 0; j < in_size; ++j) {
                Vec(input_grads_.at(j)->v + i * in_rows_.at(j), in_rows_.at(j)) +=
                    Vec(getGrad().v + i * row + offset, in_rows_.at(j));
                offset += in_rows_[j];
            }
        }
    }

protected:
    int forwardOnlyInputValSize() override {
        return inputSize();
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    vector<int> in_rows_;

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
            for (auto &p : concat->input_vals_) {
                in_vals.push_back(p->value);
            }
            vals.push_back(node->getVal().value);
            cols_.push_back(concat->getColumn());
        }

        ConcatNode &first = dynamic_cast<ConcatNode &>(*batch.front());
        row_ = first.size() / first.getColumn();
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
            for (auto &p : concat->input_grads_) {
                in_losses.push_back(p->value);
            }
            losses.push_back(node->grad().value);
        }

        cuda::ConcatBackward(in_losses, dynamic_cast<ConcatNode*>(batch.at(0))->in_rows_,
                losses, count, inCount(), row_, cols_);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }
        for (int idx = 0; idx < count; idx++) {
            for (int j = 0; j < inCount(); ++j) {
                cuda::Assert(dynamic_cast<ConcatNode *>(batch[idx])->input_grads_.at(j)->
                        verify("concat backward"));
            }
        }
#endif
    }

private:
    int inCount() {
        return dynamic_cast<ConcatNode *>(batch.front())->input_vals_.size();
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

class MatrixConcatNode : public Node, public Poolable<MatrixConcatNode> {
public:
    virtual void setNodeDim(int dim) override {
        setDim(dim);
    }

    MatrixConcatNode(): Node("matrix-concat") {}

    void connect(const vector<Node *> &inputs) {
        setInputs(inputs);
        setColumn(inputs.size());
        afterConnect(inputs);
    }

    void setInputs(const vector<Node *> &inputs) override {
        int input_dim = inputs.front()->size();
        for (auto it = inputs.begin() + 1; it != inputs.end(); ++it) {
            if (input_dim != (*it)->size()) {
                cerr << "MatrixConcatNode - forward inconsistent input dims" << endl;
                abort();
            }
        }

        Node::setInputs(inputs);
    }

    void compute() override {
        for (int i = 0; i < inputSize(); ++i) {
            int offset = i * getRow();
            for (int j = 0; j < getRow(); ++j) {
                val().v[offset + j] = input_vals_.at(i)->v[j];
            }
        }
    }

    void backward() override {
        for (int i = 0; i < inputSize(); ++i) {
            int offset = i * getRow();
            for (int j = 0; j < getRow(); ++j) {
                (*input_grads_.at(i))[j] += grad()[offset + j];
            }
        }
    }

    string typeSignature() const override {
        return "MatrixConcatNode-" + to_string(getRow());
    }

    Executor* generate() override;

protected:
    int forwardOnlyInputValSize() override {
        return inputSize();
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    friend class BatchedMatrixConcatNode;
    friend class MatrixConcatExecutor;
};

#if USE_GPU
class MatrixConcatExecutor : public Executor {
public:
    void forward() override {
#if TEST_CUDA
        testForwardInpputs();
        cout << "MatrixConcat forward tested" << endl;
#endif
        in_counts.reserve(batch.size());
        for (Node *node : batch) {
            MatrixConcatNode *concat = dynamic_cast<MatrixConcatNode*>(node);
            in_counts.push_back(concat->getColumn());
        }
        max_in_count = *max_element(in_counts.begin(), in_counts.end());
        vector<dtype *> vals, in_vals;
        vals.reserve(batch.size());
        in_vals.reserve(batch.size());
        int node_i = -1;
        for (Node *node : batch) {
            ++node_i;
            MatrixConcatNode *concat = dynamic_cast<MatrixConcatNode*>(node);
            vals.push_back(concat->getVal().value);
            for (int i = 0; i < max_in_count; ++i) {
                in_vals.push_back(i < in_counts.at(node_i) ? concat->input_vals_.at(i)->value :
                        nullptr);
            }
        }
        cuda::MatrixConcatForward(in_vals, getCount(), getRow(), in_counts, vals);
#if TEST_CUDA
        testForward();
        cout << "MatrixConcat forward tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype *> grads, in_grads;
        grads.reserve(batch.size());
        in_grads.reserve(batch.size());
        int node_i = -1;
        for (Node *node : batch) {
            ++node_i;
            MatrixConcatNode *concat = dynamic_cast<MatrixConcatNode*>(node);
            grads.push_back(concat->getGrad().value);
            for (int i = 0; i < max_in_count; ++i) {
                in_grads.push_back(i < in_counts.at(node_i) ?
                        concat->input_grads_.at(i)->value : nullptr);
            }
        }
        cuda::MatrixConcatBackward(grads, getCount(), getRow(), in_counts, in_grads);
#if TEST_CUDA
        testBackward();
#endif
    }

private:
    vector<int> in_counts;
    int max_in_count;
};
#else
class MatrixConcatExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }

    int calculateActivations() override {
        return 0;
    }
};
#endif

Executor* MatrixConcatNode::generate() {
    return new MatrixConcatExecutor;
}

namespace {

Node *concatToMatrix(const vector<Node *> &inputs) {
    int input_dim = inputs.front()->size();
    MatrixConcatNode *node = MatrixConcatNode::newNode(inputs.size() * input_dim);
    node->connect(inputs);
    return node;
}

bool areNodeSizesEqual(const vector<Node *> &nodes) {
    for (int i = 1; i < nodes.size(); ++i) {
        if (nodes.front()->size() != nodes.at(i)->size()) {
            return false;
        }
    }
    return true;
}

}

Node *cat(const vector<Node*> &inputs, int col) {
    if (col == 1 && areNodeSizesEqual(inputs)) {
        return concatToMatrix(inputs);
    } else {
        int dim = 0;
        for (Node *in : inputs) {
            dim += in->size();
        }
        ConcatNode *concat = ConcatNode::newNode(dim);
        concat->setColumn(col);
        concat->connect(inputs);
        return concat;
    }
}

Node *cat(BatchedNode &inputs, int col) {
    int dim = 0;
    for (Node *in : inputs.batch()) {
        dim += in->size();
    }
    ConcatNode *concat = ConcatNode::newNode(dim);
    concat->setColumn(col);
    concat->setInputs(inputs.batch());
    inputs.addParent(concat);
    NodeContainer &container = inputs.getNodeContainer();
    container.addNode(concat);
    return concat;
}

}
