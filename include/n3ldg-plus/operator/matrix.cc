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

class MatrixMulMatrixNode : public Node, public Poolable<MatrixMulMatrixNode> {
public:
    MatrixMulMatrixNode() : Node("MatrixMulMatrixNode") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void setInputs(const vector<Node *> &ins) override {
        int a_size = ins.at(0)->size();
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
        int a_row = a.size() / k;
        int b_col = b.size() / k;
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
        testForwardInpputs();
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
#if TEST_CUDA
        cout << "testing matmul backward input" << endl;
        testBeforeBackward();
#endif
        cuda::MatrixMulMatrixBackward(grads, a_vals_, b_vals_, count, ks_, b_cols_, row_,
                a_grads, b_grads);

#if TEST_CUDA
        cout << "testing matmul backward" << endl;
        int i = 0;
        for (Node *node: batch) {
            MatrixMulMatrixNode &m = dynamic_cast<MatrixMulMatrixNode&>(*node);
            int a_dim = m.input_vals_.at(0)->dim;
            int b_dim = m.input_vals_.at(1)->dim;
            int k = m.k_;
            cout << fmt::format("i:{} a_row:{} b_col:{} k:{}", i, a_dim / k, b_dim / k, k) << endl;
            ++i;
        }
        testBackward();
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
        if (use_lower_triangular_mask_) {
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
            (use_lower_triangular_mask_ ? "-mask" : "-no-mask");
    }

    int a_col_, b_col_, input_row_;
    bool use_lower_triangular_mask_ = false;

protected:
    int forwardOnlyInputValSize() override {
        return 0;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    friend class TranMatrixMulMatrixExecutor;
};

class BatchedTranMatrixMulMatrixNode : public BatchedNodeImpl<TranMatrixMulMatrixNode> {
public:
    void init(BatchedNode &a, BatchedNode &b, int input_row,
            bool use_lower_triangular_mask = false) {
        int a_col = a.size() / input_row;
        int b_col = b.size() / input_row;
        if (use_lower_triangular_mask && a_col != b_col) {
            cerr << fmt::format("BatchedTranMatrixMulMatrixNode init a_col:{} b_col:{}\n",
                a_col, b_col);
            abort();
        }
        allocateBatch(a_col * b_col, a.batch().size());
        setInputsPerNode({&a, &b});
        for (Node *node : batch()) {
            TranMatrixMulMatrixNode &t = dynamic_cast<TranMatrixMulMatrixNode &>(*node);
            t.use_lower_triangular_mask_ = use_lower_triangular_mask;
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
        use_lower_triangular_mask_ =
            dynamic_cast<TranMatrixMulMatrixNode &>(*batch.front()).use_lower_triangular_mask_;
        for (Node *node : batch) {
            TranMatrixMulMatrixNode &t = dynamic_cast<TranMatrixMulMatrixNode &>(*node);
            a_vals_.push_back(t.input_vals_.at(0)->value);
            b_vals_.push_back(t.input_vals_.at(1)->value);
            vals.push_back(t.getVal().value);
            a_cols_.push_back(t.input_dims_.at(0) / input_row_);
            b_cols_.push_back(t.input_dims_.at(1) / input_row_);
        }

        cuda::TranMatrixMulMatrixForward(a_vals_, b_vals_, count, a_cols_, b_cols_, input_row_,
                use_lower_triangular_mask_, vals);

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
        testBackward();
#endif
    }

private:
    vector<dtype *> a_vals_, b_vals_;
    vector<int> a_cols_, b_cols_;
    int input_row_;
    bool use_lower_triangular_mask_;
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

BatchedNode *tranMatrixMulMatrix(BatchedNode &a, BatchedNode &b, int input_row,
        bool use_lower_triangular_mask) {
    BatchedTranMatrixMulMatrixNode *node = new BatchedTranMatrixMulMatrixNode;
    node->init(a, b, input_row, use_lower_triangular_mask);
    return node;
}

Node *matmul(Node &a, Node &b, int b_row, bool transpose_a, bool use_lower_triangular_mask) {
    if (transpose_a) {
        int a_col = a.size() / b_row;
        if (a_col * b_row != a.size()) {
            cerr << fmt::format("a size:{} b_row:{}", a.size(), b_row) << endl;
            abort();
        }
        int b_col = b.size() / b_row;
        if (b_col * b_row != b.size()) {
            cerr << fmt::format("b size:{} b_row:{}", b.size(), b_row) << endl;
            abort();
        }
        if (use_lower_triangular_mask && a_col != b_col) {
            cerr << fmt::format("matmul init a_col:{} b_col:{}\n", a_col, b_col);
            abort();
        }
        TranMatrixMulMatrixNode *result = TranMatrixMulMatrixNode::newNode(a_col * b_col);
        result->use_lower_triangular_mask_ = use_lower_triangular_mask;
        result->input_row_ = b_row;
        result->connect(a, b);
        return result;
    } else {
        if (use_lower_triangular_mask) {
            cerr << fmt::format("matmul transpose_a:{} use_lower_triangular_mask:{}", transpose_a,
                    use_lower_triangular_mask) << endl;
            abort();
        }
        int a_row = a.size() / b_row;
        int b_col = b.size() / b_row;
        MatrixMulMatrixNode *result = MatrixMulMatrixNode::newNode(a_row * b_col);
        result->setColumn(b_col);
        result->k_ = b_row;
        result->connect(a, b);
        return result;
    }
}

BatchedNode *matrixMulMatrix(BatchedNode &a, BatchedNode &b, int k) {
    BatchedMatrixMulMatrixNode *node = new BatchedMatrixMulMatrixNode;
    node->init(a, b, k);
    return node;
}

}
