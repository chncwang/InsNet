#include "n3ldg-plus/operator/matrix.h"

using std::array;
using std::vector;
using std::cerr;
using std::endl;
using std::string;
using std::to_string;

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

    [[deprecated]]
    void connect(NodeAbs &topo_input, const vector<Node *> &inputs) {
        setInputs(inputs);
        topo_input.addParent(this);
        setColumn(inputs.size());
        topo_input.getNodeContainer().addNode(this);
    }

    void setInputs(const vector<Node *> &inputs) override {
        int input_dim = inputs.front()->getDim();
        for (auto it = inputs.begin() + 1; it != inputs.end(); ++it) {
            if (input_dim != (*it)->getDim()) {
                cerr << "MatrixConcatNode - forward inconsistent input dims" << endl;
                abort();
            }
        }

        in_nodes = inputs;
    }

    void compute() override {
        for (int i = 0; i < in_nodes.size(); ++i) {
            int offset = i * getRow();
            for (int j = 0; j < getRow(); ++j) {
                val().v[offset + j] = in_nodes.at(i)->getVal().v[j];
            }
        }
    }

    void backward() override {
        for (int i = 0; i < in_nodes.size(); ++i) {
            int offset = i * getRow();
            for (int j = 0; j < getRow(); ++j) {
                in_nodes.at(i)->loss()[j] += loss()[offset + j];
            }
        }
    }

    string typeSignature() const override {
        return "MatrixConcatNode-" + to_string(getRow());
    }

    Executor* generate() override;

    const vector<Node *> &getInputs() const {
        return in_nodes;
    }

private:
    vector<Node *> in_nodes;
    friend class BatchedMatrixConcatNode;
};

class BatchedMatrixConcatNode : public BatchedNodeImpl<MatrixConcatNode> {
public:
    void init(BatchedNode &input, int group) {
        if (input.batch().size() % group != 0) {
            cerr << fmt::format("input batch size:{} group:{}\n", input.batch().size(),
                    group);
        }
        int input_count = input.batch().size() / group;
        allocateBatch(input.getDim() * input_count, group);

        int group_i = 0;
        for (Node *node : batch()) {
            MatrixConcatNode *m = dynamic_cast<MatrixConcatNode *>(node);
            vector<Node *> in_nodes;
            in_nodes.reserve(input_count);
            for (int i = 0; i < input_count; ++i) {
                in_nodes.push_back(input.batch().at(group_i * input_count + i));
            }
            ++group_i;
            m->in_nodes = move(in_nodes);
            m->setColumn(input_count);
        }

        afterInit({&input});
    }
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
                in_vals.push_back(i < in_counts.at(node_i) ?
                        concat->getInputs().at(i)->getVal().value : nullptr);
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
            grads.push_back(concat->getLoss().value);
            for (int i = 0; i < max_in_count; ++i) {
                in_grads.push_back(i < in_counts.at(node_i) ?
                        concat->getInputs().at(i)->getLoss().value : nullptr);
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

class MatrixAndVectorMultiExecutor;

class MatrixAndVectorMultiNode : public Node, public Poolable<MatrixAndVectorMultiNode> {
public:
    MatrixAndVectorMultiNode() : Node("MatrixAndVectorMulti") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void setInputs(const vector<Node *> &ins) override {
        matrix_ = ins.at(0);
        vector_ = ins.at(1);
    }

    void connect(Node &matrix, Node &vec) {
        setInputs({&matrix, &vec});
        afterConnect({&matrix, &vec});
    }

    void compute() override {
        int col = vector_->getDim();
        int row = getDim();
        val().mat() = Mat(matrix_->getVal().v, row, col) * vector_->getVal().mat();
    }

    void backward() override {
        int col = vector_->getDim();
        int row = getDim();
        Mat(matrix_->loss().v, row, col) += getLoss().mat() *
            vector_->getVal().mat().transpose();
        vector_->loss().mat() += Mat(matrix_->getVal().v, row, col).transpose() * getLoss().mat();
    }

    Executor * generate() override;

    string typeSignature() const override {
        return Node::typeSignature();
    }

private:
    Node *vector_ = nullptr;
    Node *matrix_ = nullptr;
    friend class MatrixAndVectorMultiExecutor;
    friend class BatchedMatrixAndVectorMultiNode;
};

class BatchedMatrixAndVectorMultiNode: public BatchedNodeImpl<MatrixAndVectorMultiNode> {
public:
    void init(Node &matrix, BatchedNode &vec, int *dim = nullptr) {
        if (dim == nullptr) {
            int dim = matrix.getDim() / vec.getDim();
            allocateBatch(dim, vec.batch().size());
        } else {
            allocateBatch(*dim, vec.batch().size());
        }
        int i = 0;
        for (Node *node : batch()) {
            MatrixAndVectorMultiNode *m = dynamic_cast<MatrixAndVectorMultiNode *>(node);
            m->setInputs({&matrix, vec.batch().at(+i++)});
        }

        matrix.addParent(this);
        vec.addParent(this);
        matrix.getNodeContainer().addNode(this);
    }

    void init(BatchedNode &matrix, BatchedNode &vec, int *dim = nullptr) {
        int group = matrix.batch().size();
        if (vec.batch().size() % group != 0) {
            cerr << fmt::format("BatchedTranMatrixMulVectorNode init vec size:{} group:{}\n",
                vec.batch().size(), group);
            abort();
        }
        if (dim == nullptr) {
            int dim = matrix.getDim() / vec.getDim();
            allocateBatch(dim, vec.batch().size());
        } else {
            allocateBatch(*dim, vec.batch().size());
        }

        int node_i = 0;
        for (Node *matrix_node : matrix.batch()) {
            int vec_count = vec.batch().size() / group;
            for (int i = 0; i < vec_count; ++i) {
                Node *node = batch().at(node_i);
                node->setInputs({matrix_node, vec.batch().at(node_i++)});
            }
        }

        afterInit({&matrix, &vec});
    }
};

#if USE_GPU
class MatrixAndVectorMultiExecutor : public Executor {
public:
    void forward() override {
#if TEST_CUDA
        auto get_inputs = [&](Node &node)->vector<Node *> {
            MatrixAndVectorMultiNode &multi = dynamic_cast<MatrixAndVectorMultiNode&>(node);
            vector<Node *> inputs = {multi.matrix_, multi.vector_};
            return inputs;
        };
        testForwardInpputs(get_inputs);
        for (Node *node : batch) {
            MatrixAndVectorMultiNode *multi = dynamic_cast<MatrixAndVectorMultiNode *>(node);
            multi->matrix_->val().copyFromHostToDevice();
            multi->vector_->val().copyFromHostToDevice();
        }
#endif
        auto vals = getVals();
        matrix_vals_.reserve(batch.size());
        vector_vals_.reserve(batch.size());
        cols_.reserve(batch.size());
        for (Node *node : batch) {
            MatrixAndVectorMultiNode *multi = dynamic_cast<MatrixAndVectorMultiNode *>(node);
            matrix_vals_.push_back(multi->matrix_->getVal().value);
            vector_vals_.push_back(multi->vector_->getVal().value);
            cols_.push_back(multi->vector_->getDim());
        }
        MatrixAndVectorMultiNode *x = dynamic_cast<MatrixAndVectorMultiNode *>(batch.front());
        row_ = x->getDim();
        cuda::MatrixAndVectorMultiForward(matrix_vals_, vector_vals_, batch.size(), row_,
                cols_, vals);
#if TEST_CUDA
        testForward();
        cout << "MatrixAndVectorMultiExecutor forward tested" << endl;
#endif
    }

    void backward() override {
        auto grads = getGrads();

        vector<dtype *> matrix_grads, vector_grads;
        matrix_grads.reserve(batch.size());
        vector_grads.reserve(batch.size());
        for (Node *node : batch) {
            MatrixAndVectorMultiNode *multi = dynamic_cast<MatrixAndVectorMultiNode *>(node);
            matrix_grads.push_back(multi->matrix_->getLoss().value);
            vector_grads.push_back(multi->vector_->getLoss().value);
        }
        cuda::MatrixAndVectorMultiBackward(grads, matrix_vals_, vector_vals_, batch.size(),
                row_, cols_, matrix_grads, vector_grads);
#if TEST_CUDA
        auto get_inputs = [&](Node &node) {
            MatrixAndVectorMultiNode &multi = dynamic_cast<MatrixAndVectorMultiNode&>(node);
            vector<pair<Node *, string>> inputs = {make_pair(multi.matrix_, "matrix"),
            make_pair(multi.vector_, "vector")};
            return inputs;
        };
        testBackward(get_inputs);
        cout << "MatrixAndVectorMultiExecutor backward tested" << endl;
#endif
    }

private:
    int row_;
    vector<int> cols_;
    vector<dtype *> matrix_vals_, vector_vals_;
};
#else
class MatrixAndVectorMultiExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0; // TODO
    }
};
#endif

Executor * MatrixAndVectorMultiNode::generate() {
    return new MatrixAndVectorMultiExecutor;
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
        ins_.at(0) = ins.at(0);
        ins_.at(1) = ins.at(1);
    }

    void connect(Node &a, Node &b) {
        setInputs({&a, &b});
        afterConnect({&a, &b});
    }

    void compute() override {
        a_row_ = ins_.at(0)->getDim() / k_;
        b_col_ = ins_.at(1)->getDim() / k_;
        Mat(getVal().v, a_row_, b_col_) = Mat(ins_.at(0)->getVal().v, a_row_, k_) *
            Mat(ins_.at(1)->getVal().v, k_, b_col_);
    }

    void backward() override {
        Mat(ins_.at(0)->loss().v, a_row_, k_) += Mat(getLoss().v, a_row_, b_col_) *
            Mat(ins_.at(1)->getVal().v, k_, b_col_).transpose();
        Mat(ins_.at(1)->loss().v, k_, b_col_) +=
            Mat(ins_.at(0)->getVal().v, a_row_, k_).transpose() * Mat(getLoss().v, a_row_, b_col_);
    }

    Executor * generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + to_string(ins_.at(0)->getDim() / k_);
    }

    int k_ = 0;

private:
    array<Node *, 2> ins_;
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
            a_vals_.push_back(m.ins_.at(0)->getVal().value);
            b_vals_.push_back(m.ins_.at(1)->getVal().value);
            vals.push_back(m.getVal().value);
            ks_.push_back(m.k_);
            b_cols_.push_back(m.ins_.at(1)->getDim() / m.k_);
        }
        MatrixMulMatrixNode &first = dynamic_cast<MatrixMulMatrixNode &>(*batch.front());
        row_ = first.ins_.at(0)->getDim() / first.k_;
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
            grads.push_back(m.getLoss().value);
            a_grads.push_back(m.ins_.at(0)->getLoss().value);
            b_grads.push_back(m.ins_.at(1)->getLoss().value);
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

class TranMatrixMulVectorNode : public Node, public Poolable<TranMatrixMulVectorNode> {
public:
    TranMatrixMulVectorNode() : Node("TranMatrixMulVectorNode") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void setInputs(const vector<Node *> &ins) override {
        matrix_ = ins.at(0);
        vector_ = ins.at(1);
        int max_col = matrix_->getDim() / vector_->getDim();
        if (getDim() > max_col) {
            cerr << fmt::format("TranMatrixMulVectorNode setInputs dim:{} max_col:{}\n",
                getDim(), max_col);
            abort();
        }
    }

    void connect(Node &matrix, Node &vec) {
        setInputs({&matrix, &vec});
        afterConnect({&matrix, &vec});
    }

    void compute() override {
        val().mat() = Mat(matrix_->getVal().v, vector_->getDim(), getDim()).transpose() *
            vector_->getVal().mat();
    }

    void backward() override {
        Mat(matrix_->loss().v, vector_->getDim(), getDim()) += vector_->getVal().mat() *
            getLoss().mat().transpose();
        vector_->loss().mat() += Mat(matrix_->val().v, vector_->getDim(), getDim()) *
            getLoss().mat();
    }

    Executor * generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + to_string(vector_->getDim());
    }

private:
    Node *matrix_ = nullptr;
    Node *vector_ = nullptr;
    friend class BatchedTranMatrixMulVectorNode;
    friend class TranMatrixMulVectorExecutor;
};

class BatchedTranMatrixMulVectorNode : public BatchedNodeImpl<TranMatrixMulVectorNode> {
public:
    void init(Node &matrix, BatchedNode &vec, const vector<int> *dims = nullptr) {
        if (dims == nullptr) {
            int dim = matrix.getDim() / vec.getDim();
            allocateBatch(dim, vec.batch().size());
        } else {
            allocateBatch(*dims);
        }
        int i = 0;
        for (Node *node : batch()) {
            node->setInputs({&matrix, vec.batch().at(i++)});
        }
        matrix.addParent(this);
        vec.addParent(this);
        matrix.getNodeContainer().addNode(this);
    }

    void init(BatchedNode &matrix, BatchedNode &vec, const vector<int> *dims = nullptr) {
        int group = matrix.batch().size();
        if (vec.batch().size() % group != 0) {
            cerr << fmt::format("BatchedTranMatrixMulVectorNode init vec size:{} group:{}\n",
                vec.batch().size(), group);
            abort();
        }

        if (dims == nullptr) {
            int dim = matrix.getDim() / vec.getDim();
            allocateBatch(dim, vec.batch().size());
        } else {
            vector<int> overall_dims;
            overall_dims.reserve(group * dims->size());
            for (int i = 0; i < group; ++i) {
                for (int dim : *dims) {
                    overall_dims.push_back(dim);
                }
            }
            allocateBatch(overall_dims);
        }

        int node_i = 0;
        for (Node *matrix_node : matrix.batch()) {
            int vec_count = vec.batch().size() / group;
            for (int i = 0; i < vec_count; ++i) {
                Node *node = batch().at(node_i);
                node->setInputs({matrix_node, vec.batch().at(node_i++)});
            }
        }

        afterInit({&matrix, &vec});
    }
};

#if USE_GPU
class TranMatrixMulVectorExecutor : public Executor {
public:
    void forward() override {
        cols_.reserve(batch.size());
        matrices_.reserve(batch.size());
        vectors_.reserve(batch.size());
        vals_.reserve(batch.size());
        for (Node *node : batch) {
            TranMatrixMulVectorNode *t = dynamic_cast<TranMatrixMulVectorNode *>(node);
            cols_.push_back(t->getDim());
            matrices_.push_back(t->matrix_->getVal().value);
            vectors_.push_back(t->vector_->getVal().value);
            vals_.push_back(t->getVal().value);
        }
        cuda::TranMatrixMulVectorForward(matrices_, vectors_, batch.size(), cols_, row(),
                vals_);
#if TEST_CUDA
        testForward();
#endif
    }

    void backward() override {
#if TEST_CUDA
        auto get_inputs = [&](Node &node) {
            TranMatrixMulVectorNode &t = dynamic_cast<TranMatrixMulVectorNode&>(node);
            vector<pair<Node*, string>> pairs = {
                make_pair(t.matrix_, "matrix"),
                make_pair(t.vector_, "vector")
            };
            return pairs;
        };
#endif
        vector<dtype *> matrix_grads, vector_grads, grads;
        matrix_grads.reserve(batch.size());
        vector_grads.reserve(batch.size());
        grads.reserve(batch.size());
        for (Node *node : batch) {
            TranMatrixMulVectorNode *t = dynamic_cast<TranMatrixMulVectorNode *>(node);
            grads.push_back(t->getLoss().value);
            matrix_grads.push_back(t->matrix_->getLoss().value);
            vector_grads.push_back(t->vector_->getLoss().value);
        }
        cuda::TranMatrixMulVectorBackward(grads, matrices_, vectors_, batch.size(),
                cols_, row(), matrix_grads, vector_grads);
#if TEST_CUDA
        testBackward(get_inputs);
#endif
    }

private:
    int row() const {
        TranMatrixMulVectorNode *t = dynamic_cast<TranMatrixMulVectorNode *>(batch.front());
        return t->vector_->getDim();
    }

    vector<int> cols_;
    vector<dtype *> matrices_, vectors_, vals_;
};
#else
class TranMatrixMulVectorExecutor : public Executor {
    int calculateFLOPs() override {
        abort();
    }
};
#endif

Executor *TranMatrixMulVectorNode::generate() {
    return new TranMatrixMulVectorExecutor;
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

    void setInputs(const vector<Node *> &ins) override {
        ins_ = {ins.at(0), ins.at(1)};
    }

    void connect(Node &a, Node &b) {
        vector<Node *> inputs = {&a, &b};
        setInputs(inputs);
        afterConnect(inputs);
    }

    void compute() override {
        a_col_ = ins_.at(0)->getDim() / input_row_;
        b_col_ = ins_.at(1)->getDim() / input_row_;
        Mat(val().v, a_col_, b_col_) = Mat(ins_.at(0)->getVal().v, input_row_, a_col_).transpose()
            * Mat(ins_.at(1)->getVal().v, input_row_, b_col_);
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
        Mat(ins_.at(0)->loss().v, input_row_, a_col_) +=
            Mat(ins_.at(1)->getVal().v, input_row_, b_col_) *
            Mat(getLoss().v, a_col_, b_col_).transpose();
        Mat(ins_.at(1)->loss().v, input_row_, b_col_) +=
            Mat(ins_.at(0)->getVal().v, input_row_, a_col_) *
            Mat(getLoss().v, a_col_, b_col_);
    }

    Executor * generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + to_string(input_row_) +
            (use_lower_triangle_mask_ ? "-mask" : "-no-mask");
    }

private:
    array<Node *, 2> ins_;
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
            a_vals_.push_back(t.ins_.at(0)->getVal().value);
            b_vals_.push_back(t.ins_.at(1)->getVal().value);
            vals.push_back(t.getVal().value);
            a_cols_.push_back(t.ins_.at(0)->getDim() / input_row_);
            b_cols_.push_back(t.ins_.at(1)->getDim() / input_row_);
        }

        cuda::TranMatrixMulMatrixForward(a_vals_, b_vals_, count, a_cols_, b_cols_,
                input_row_, use_lower_triangle_mask_, vals);

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
            a_grads.push_back(t.ins_.at(0)->getLoss().value);
            b_grads.push_back(t.ins_.at(1)->getLoss().value);
            grads.push_back(t.getLoss().value);
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

Node *concatToMatrix(BatchedNode &input) {
    const auto &inputs = input.batch();
    int input_dim = inputs.front()->getDim();
    MatrixConcatNode *node = MatrixConcatNode::newNode(inputs.size() * input_dim);
    node->connect(input, inputs);
    return node;
}

BatchedNode *concatToMatrix(BatchedNode &input, int group) {
    BatchedMatrixConcatNode *node = new BatchedMatrixConcatNode;
    node->init(input, group);
    return node;
}

Node *matrixAndVectorMulti(Node &matrix, Node &vec) {
    int dim = matrix.getDim() / vec.getDim();
    if (matrix.getDim() % vec.getDim() != 0) {
        cerr << fmt::format("vec dim:%1% matrix dim:%2%", vec.getDim(), matrix.getDim()) <<
            endl;
        abort();
    }
    MatrixAndVectorMultiNode *node = MatrixAndVectorMultiNode::newNode(dim);
    node->connect(matrix, vec);
    return node;
}

BatchedNode *matrixAndVectorMulti(Node &matrix, BatchedNode &vec, int *dim) {
    if (matrix.getDim() % vec.getDim() != 0) {
        cerr << fmt::format("vec dim:{} matrix dim:{}\n", vec.getDim(), matrix.getDim());
        abort();
    }

    BatchedMatrixAndVectorMultiNode *node = new BatchedMatrixAndVectorMultiNode;
    node->init(matrix, vec, dim);
    return node;
}

BatchedNode *matrixAndVectorMulti(BatchedNode &matrix, BatchedNode &vec, int *dim) {
    BatchedMatrixAndVectorMultiNode *node = new BatchedMatrixAndVectorMultiNode;
    node->init(matrix, vec, dim);
    return node;
}

Node *tranMatrixMulVector(Node &matrix, Node &vec, int dim) {
    TranMatrixMulVectorNode *node = TranMatrixMulVectorNode::newNode(dim);
    node->connect(matrix, vec);
    return node;
}

Node *tranMatrixMulVector(Node &matrix, Node &vec) {
    int dim = matrix.getDim() / vec.getDim();
    return tranMatrixMulVector(matrix, vec, dim);
}

BatchedNode *tranMatrixMulVector(Node &matrix, BatchedNode &vec,
        const vector<int> *dims) {
    BatchedTranMatrixMulVectorNode *node = new BatchedTranMatrixMulVectorNode;
    node->init(matrix, vec, dims);
    return node;
}

BatchedNode *tranMatrixMulVector(BatchedNode &matrix, BatchedNode &vec,
        const vector<int> *dims) {
    BatchedTranMatrixMulVectorNode *node = new BatchedTranMatrixMulVectorNode;
    node->init(matrix, vec, dims);
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
