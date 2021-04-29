#include "n3ldg-plus/operator/matrix.h"
#include "n3ldg-plus/util/util.h"

using std::array;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::make_pair;
using std::pair;

namespace n3ldg_plus {

class MatrixExecutor : public Executor {
public:
    int getRow() const {
        return batch.front()->getRow();
    }

    vector<int> getCols() const {
        vector<int> cols;
        cols.reserve(batch.size());
        for (Node *node : batch) {
            cols.push_back(node->getColumn());
        }
        return cols;
    }
};

class MatrixConcatNode : public Node, public Poolable<MatrixConcatNode> {
public:
    virtual void initNode(int dim) override {
        init(dim);
    }

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
        int input_dim = inputs.front()->getDim();
        for (auto it = inputs.begin() + 1; it != inputs.end(); ++it) {
            if (input_dim != (*it)->getDim()) {
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
class MatrixConcatExecutor : public MatrixExecutor {
public:
    void forward() override {
#if TEST_CUDA
        auto get_inputs = [](Node &node) -> vector<Node *> {
            MatrixConcatNode *m = dynamic_cast<MatrixConcatNode*>(&node);
            return m->getInputs();
        };
        testForwardInpputs(get_inputs);
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
        auto get_inputs = [&](Node &node) {
            vector<pair<Node*, string>> pairs;
            MatrixConcatNode &concat = dynamic_cast<MatrixConcatNode&>(node);
            for (Node *input : concat.getInputs()) {
                pairs.push_back(make_pair(input, input->getNodeType()));
            }
            return pairs;
        };
        testBackward(get_inputs);
#endif
    }

private:
    vector<int> in_counts;
    int max_in_count;
};
#else
class MatrixConcatExecutor : public MatrixExecutor {
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

class MatrixMulMatrixNode : public Node, public Poolable<MatrixMulMatrixNode> {
public:
    MatrixMulMatrixNode() : Node("MatrixMulMatrixNode") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void setInputs(const vector<Node *> &ins) override {
        int a_size = ins.at(0)->getDim();
        if (a_size % k_ != 0) {
            cerr << fmt::format("MatrixMulMatrixNode setInputs a_size:{} k:{}\n", a_size, k_);
            abort();
        }
        Node::setInputs(ins);
    }

    void connect(Node &a, Node &b) {
        setInputs({&a, &b});
        afterConnect({&a, &b});
    }

    void compute() override {
        a_row_ = input_dims_.at(0) / k_;
        b_col_ = input_dims_.at(1) / k_;
        Mat(getVal().v, a_row_, b_col_) = Mat(input_vals_.at(0)->v, a_row_, k_) *
            Mat(input_vals_.at(1)->v, k_, b_col_);
    }

    void backward() override {
        Mat(input_grads_.at(0)->v, a_row_, k_) += Mat(getGrad().v, a_row_, b_col_) *
            Mat(input_vals_.at(1)->v, k_, b_col_).transpose();
        Mat(input_grads_.at(1)->v, k_, b_col_) +=
            Mat(input_vals_.at(0)->v, a_row_, k_).transpose() * Mat(getGrad().v, a_row_, b_col_);
    }

    Executor * generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + to_string(input_dims_.at(0) / k_);
    }

    int k_ = 0;

protected:
    int forwardOnlyInputValSize() override {
        return 0;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    int a_row_;
    int b_col_;
    
    friend class BatchedMatrixMulMatrixNode;
    friend class MatrixMulMatrixExecutor;
};

class BatchedMatrixMulMatrixNode : public BatchedNodeImpl<MatrixMulMatrixNode> {
public:
    void init(BatchedNode &a, BatchedNode &b, int k) {
        int a_row = a.getDim() / k;
        int b_col = b.getDim() / k;
        allocateBatch(a_row * b_col, a.batch().size());
        for (Node *node : batch()) {
            MatrixMulMatrixNode &m = dynamic_cast<MatrixMulMatrixNode &>(*node);
            m.k_ = k;
        }
        setInputsPerNode({&a, &b});
        afterInit({&a, &b});
    }
};

#if USE_GPU
class MatrixMulMatrixExecutor : public Executor {
public:
    void forward() override {
        int count = batch.size();
        vector<dtype *> vals;
        a_vals_.reserve(count);
        b_vals_.reserve(count);
        vals.reserve(count);
        ks_.reserve(count);
        b_cols_.reserve(count);
        for (Node *node : batch) {
            MatrixMulMatrixNode &m = dynamic_cast<MatrixMulMatrixNode &>(*node);
            a_vals_.push_back(m.input_vals_.at(0)->value);
            b_vals_.push_back(m.input_vals_.at(1)->value);
            vals.push_back(m.getVal().value);
            ks_.push_back(m.k_);
            b_cols_.push_back(m.input_dims_.at(1) / m.k_);
        }
        MatrixMulMatrixNode &first = dynamic_cast<MatrixMulMatrixNode &>(*batch.front());
        row_ = first.input_dims_.at(0) / first.k_;
#if TEST_CUDA
        auto get_inputs = [&](Node &node) {
            MatrixMulMatrixNode &t = dynamic_cast<MatrixMulMatrixNode&>(node);
            vector<pair<Node*, string>> pairs = {
                make_pair(t.ins_.at(0), "a"),
                make_pair(t.ins_.at(1), "b")
            };
            return pairs;
        };
        testForwardInpputs(get_inputs);
#endif

        cuda::MatrixMulMatrixForward(a_vals_, b_vals_, count, ks_, b_cols_, row_, vals);
#if TEST_CUDA
        testForward();
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype *> grads, a_grads, b_grads;
        grads.reserve(count);
        a_grads.reserve(count);
        b_grads.reserve(count);

        for (Node *node : batch) {
            MatrixMulMatrixNode &m = dynamic_cast<MatrixMulMatrixNode &>(*node);
            grads.push_back(m.getGrad().value);
            a_grads.push_back(m.input_grads_.at(0)->value);
            b_grads.push_back(m.input_grads_.at(1)->value);
        }

        cuda::MatrixMulMatrixBackward(grads, a_vals_, b_vals_, count, ks_, b_cols_, row_,
                a_grads, b_grads);

#if TEST_CUDA
        auto get_inputs = [&](Node &node) {
            MatrixMulMatrixNode &t = dynamic_cast<MatrixMulMatrixNode&>(node);
            vector<pair<Node*, string>> pairs = {
                make_pair(t.ins_.at(0), "a"),
                make_pair(t.ins_.at(1), "b")
            };
            return pairs;
        };
        testBackward(get_inputs);
#endif
    }

private:
    vector<dtype *> a_vals_, b_vals_;
    vector<int> ks_, b_cols_;
    int row_;
};
#else
class MatrixMulMatrixExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0; // TODO
    }
};
#endif

Executor *MatrixMulMatrixNode::generate() {
    return new MatrixMulMatrixExecutor;
}

class TranMatrixMulMatrixNode : public Node, public Poolable<TranMatrixMulMatrixNode> {
public:
    TranMatrixMulMatrixNode() : Node("TranMatrixMulMatrixNode") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void connect(Node &a, Node &b) {
        vector<Node *> inputs = {&a, &b};
        setInputs(inputs);
        afterConnect(inputs);
    }

    void compute() override {
        a_col_ = input_dims_.at(0) / input_row_;
        b_col_ = input_dims_.at(1) / input_row_;
        Mat(val().v, a_col_, b_col_) = Mat(input_vals_.at(0)->v, input_row_, a_col_).transpose()
            * Mat(input_vals_.at(1)->v, input_row_, b_col_);
        if (use_lower_triangle_mask_) {
            if (a_col_ != b_col_) {
                cerr << fmt::format("a_col_:{} b_col_:{}\n", a_col_, b_col_);
                abort();
            }
            for (int i = 0; i < a_col_; ++i) {
                for (int j = i + 1; j < a_col_; ++j) {
                    val()[i * a_col_ + j] = -INF;
                }
            }
        }
    }

    void backward() override {
        Mat(input_grads_.at(0)->v, input_row_, a_col_) +=
            Mat(input_vals_.at(1)->v, input_row_, b_col_) *
            Mat(getGrad().v, a_col_, b_col_).transpose();
        Mat(input_grads_.at(1)->v, input_row_, b_col_) +=
            Mat(input_vals_.at(0)->v, input_row_, a_col_) * Mat(getGrad().v, a_col_, b_col_);
    }

    Executor * generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + to_string(input_row_) +
            (use_lower_triangle_mask_ ? "-mask" : "-no-mask");
    }

protected:
    int forwardOnlyInputValSize() override {
        return 0;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    int a_col_, b_col_, input_row_;
    bool use_lower_triangle_mask_ = false;
    friend class TranMatrixMulMatrixExecutor;
    friend class BatchedTranMatrixMulMatrixNode;
};

class BatchedTranMatrixMulMatrixNode : public BatchedNodeImpl<TranMatrixMulMatrixNode> {
public:
    void init(BatchedNode &a, BatchedNode &b, int input_row,
            bool use_lower_triangle_mask = false) {
        int a_col = a.getDim() / input_row;
        int b_col = b.getDim() / input_row;
        if (use_lower_triangle_mask && a_col != b_col) {
            cerr << fmt::format("BatchedTranMatrixMulMatrixNode init a_col:{} b_col:{}\n",
                a_col, b_col);
            abort();
        }
        allocateBatch(a_col * b_col, a.batch().size());
        setInputsPerNode({&a, &b});
        for (Node *node : batch()) {
            TranMatrixMulMatrixNode &t = dynamic_cast<TranMatrixMulMatrixNode &>(*node);
            t.use_lower_triangle_mask_ = use_lower_triangle_mask;
            t.input_row_ = input_row;
        }
        afterInit({&a, &b});
    }
};

#if USE_GPU
class TranMatrixMulMatrixExecutor : public Executor {
public:
    void forward() override {
        int count = batch.size();
        vector<dtype *> vals;
        vals.reserve(count);
        a_cols_.reserve(count);
        b_cols_.reserve(count);
        b_cols_.reserve(count);
        a_vals_.reserve(count);
        b_vals_.reserve(count);
        input_row_ = dynamic_cast<TranMatrixMulMatrixNode &>(*batch.front()).input_row_;
        use_lower_triangle_mask_ =
            dynamic_cast<TranMatrixMulMatrixNode &>(*batch.front()).use_lower_triangle_mask_;
        for (Node *node : batch) {
            TranMatrixMulMatrixNode &t = dynamic_cast<TranMatrixMulMatrixNode &>(*node);
            a_vals_.push_back(t.input_vals_.at(0)->value);
            b_vals_.push_back(t.input_vals_.at(1)->value);
            vals.push_back(t.getVal().value);
            a_cols_.push_back(t.input_dims_.at(0) / input_row_);
            b_cols_.push_back(t.input_dims_.at(1) / input_row_);
        }

        cuda::TranMatrixMulMatrixForward(a_vals_, b_vals_, count, a_cols_, b_cols_, input_row_,
                use_lower_triangle_mask_, vals);

#if TEST_CUDA
        testForward();
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype *> a_grads, b_grads, grads;
        a_grads.reserve(count);
        b_grads.reserve(count);
        for (Node *node : batch) {
            TranMatrixMulMatrixNode &t = dynamic_cast<TranMatrixMulMatrixNode &>(*node);
            a_grads.push_back(t.input_grads_.at(0)->value);
            b_grads.push_back(t.input_grads_.at(1)->value);
            grads.push_back(t.getGrad().value);
        }
        cuda::TranMatrixMulMatrixBackward(grads, a_vals_, b_vals_, count, a_cols_, b_cols_,
                input_row_, a_grads, b_grads);

#if TEST_CUDA
        auto get_inputs = [&](Node &node) {
            vector<pair<Node*, string>> pairs;
            TranMatrixMulMatrixNode &t = dynamic_cast<TranMatrixMulMatrixNode &>(node);
            pairs.push_back(make_pair(t.ins_.at(0), "a"));
            pairs.push_back(make_pair(t.ins_.at(1), "b"));
            return pairs;
        };
        testBackward(get_inputs);
#endif
    }

private:
    vector<dtype *> a_vals_, b_vals_;
    vector<int> a_cols_, b_cols_;
    int input_row_;
    bool use_lower_triangle_mask_;
};
#else
class TranMatrixMulMatrixExecutor : public Executor {
public:
    int calculateFLOPs() override {
        abort();
    }
};
#endif

Executor* TranMatrixMulMatrixNode::generate() {
    return new TranMatrixMulMatrixExecutor;
}

Node *concatToMatrix(const vector<Node *> &inputs) {
    int input_dim = inputs.front()->getDim();
    MatrixConcatNode *node = MatrixConcatNode::newNode(inputs.size() * input_dim);
    node->connect(inputs);
    return node;
}

BatchedNode *tranMatrixMulMatrix(BatchedNode &a, BatchedNode &b, int input_row,
        bool use_lower_triangle_mask) {
    BatchedTranMatrixMulMatrixNode *node = new BatchedTranMatrixMulMatrixNode;
    node->init(a, b, input_row, use_lower_triangle_mask);
    return node;
}

Node *matrixMulMatrix(Node &a, Node &b, int k) {
    int a_row = a.getDim() / k;
    int b_col = b.getDim() / k;
    MatrixMulMatrixNode *result = MatrixMulMatrixNode::newNode(a_row * b_col);
    result->setColumn(b_col);
    result->k_ = k;
    result->connect(a, b);
    return result;
}

BatchedNode *matrixMulMatrix(BatchedNode &a, BatchedNode &b, int k) {
    BatchedMatrixMulMatrixNode *node = new BatchedMatrixMulMatrixNode;
    node->init(a, b, k);
    return node;
}

}
