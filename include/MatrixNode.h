#ifndef N3LDG_PLUS_MATRX_NODE_H
#define N3LDG_PLUS_MATRX_NODE_H

#include "Node.h"
#include "Graph.h"

class MatrixNode : public Node {
public:
    explicit MatrixNode(const string &node_type) : Node(node_type + "-matrix") {}
};

class MatrixExecutor : public Executor {
public:
    int getRow() const {
        return static_cast<MatrixNode *>(batch.front())->getRow();
    }

    vector<int> getCols() const {
        vector<int> cols;
        cols.reserve(batch.size());
        for (Node *node : batch) {
            MatrixNode *matrix = static_cast<MatrixNode *>(node);
            cols.push_back(matrix->getColumn());
        }
        return cols;
    }
};

class MatrixConcatNode : public MatrixNode, public Poolable<MatrixConcatNode> {
public:
    virtual void initNode(int dim) override {
        init(dim);
    }

    virtual void setNodeDim(int dim) override {
        setDim(dim);
    }

    MatrixConcatNode(): MatrixNode("concat") {}

    void forward(Graph &graph, const vector<Node *> &inputs) {
        int input_dim = inputs.front()->getDim();
        for (auto it = inputs.begin() + 1; it != inputs.end(); ++it) {
            if (input_dim != (*it)->getDim()) {
                cerr << "MatrixConcatNode - forward inconsistent input dims" << endl;
                abort();
            }
        }

        in_nodes = inputs;
        for (Node *in : inputs) {
            in->addParent(this);
        }
        setColumn(inputs.size());
        graph.addNode(this);
    }

    void forward(Graph &graph, NodeAbs &topo_input, const vector<Node *> &inputs) {
        int input_dim = inputs.front()->getDim();
        for (auto it = inputs.begin() + 1; it != inputs.end(); ++it) {
            if (input_dim != (*it)->getDim()) {
                cerr << "MatrixConcatNode - forward inconsistent input dims" << endl;
                abort();
            }
        }

        in_nodes = inputs;
        topo_input.addParent(this);
        setColumn(inputs.size());
        graph.addNode(this);
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
    void init(Graph &graph, BatchedNode &input, const vector<vector<Node *>> &cols) {
        vector<int> dims;
        for (const auto &v : cols) {
            dims.push_back(v.front()->getDim() * v.size());
        }
        allocateBatch(dims);

        for (int i = 0; i < batch().size(); ++i) {
            MatrixConcatNode *node = dynamic_cast<MatrixConcatNode *>(batch().at(i));
            node->in_nodes = cols.at(i);
            node->setColumn(cols.at(i).size());
        }

        input.addParent(this);
        graph.addNode(this);
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
        for (Node *node : batch) {
            MatrixConcatNode *concat = static_cast<MatrixConcatNode*>(node);
            in_counts.push_back(concat->getColumn());
        }
        max_in_count = *max_element(in_counts.begin(), in_counts.end());
        vector<dtype *> vals, in_vals;
        int node_i = -1;
        for (Node *node : batch) {
            ++node_i;
            MatrixConcatNode *concat = static_cast<MatrixConcatNode*>(node);
            vals.push_back(concat->getVal().value);
            for (int i = 0; i < max_in_count; ++i) {
                in_vals.push_back(i < in_counts.at(node_i) ?
                        concat->getInputs().at(i)->getVal().value : nullptr);
            }
        }
        n3ldg_cuda::MatrixConcatForward(in_vals, getCount(), getRow(), in_counts, vals);
#if TEST_CUDA
        testForward();
        cout << "MatrixConcat forward tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype *> grads, in_grads;
        int node_i = -1;
        for (Node *node : batch) {
            ++node_i;
            MatrixConcatNode *concat = static_cast<MatrixConcatNode*>(node);
            grads.push_back(concat->getLoss().value);
            for (int i = 0; i < max_in_count; ++i) {
                in_grads.push_back(i < in_counts.at(node_i) ?
                        concat->getInputs().at(i)->getLoss().value : nullptr);
            }
        }
        n3ldg_cuda::MatrixConcatBackward(grads, getCount(), getRow(), in_counts, in_grads);
#if TEST_CUDA
        auto get_inputs = [&](Node &node) {
            vector<pair<Node*, string>> pairs;
            MatrixConcatNode &concat = static_cast<MatrixConcatNode&>(node);
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

class MatrixAndVectorPointwiseMultiExecutor;

class MatrixAndVectorPointwiseMultiNode : public MatrixNode,
    public Poolable<MatrixAndVectorPointwiseMultiNode> {
public:
    MatrixAndVectorPointwiseMultiNode() : MatrixNode("MatrixAndVectorPointwiseMulti") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void forward(Graph &graph, Node &matrix, Node &vec) {
        if (getDim() % vec.getDim() != 0) {
            cerr << boost::format("MatrixConcatNode forward - dim:%1% vec dim:%2%") % getDim() %
                vec.getDim() << endl;
            abort();
        }

        setInputs({&matrix, &vec});
        afterForward(graph, {&matrix, &vec});
    }

    void setInputs(const vector<Node *> &ins) override {
        matrix_ = ins.at(0);
        vector_ = ins.at(1);
    }

    void compute() override {
        int in_dim = vector_->getDim();
        int col = getDim() / in_dim;
        for (int i = 0; i < col; ++i) {
            int matrix_i = i * in_dim;
            Vec(val().v + matrix_i, in_dim) = vector_->getVal().vec() *
                Vec(matrix_->getVal().v + matrix_i, in_dim);
        }
    }

    void backward() override {
        int in_dim = vector_->getDim();
        int col = getDim() / in_dim;
        for (int i = 0; i < col; ++i) {
            int matrix_i = i * in_dim;
            Vec(matrix_->loss().v + matrix_i, in_dim) += Vec(getLoss().v + matrix_i, in_dim) *
                vector_->getVal().vec();
            vector_->loss().vec() += Vec(getLoss().v + matrix_i, in_dim) *
                Vec(matrix_->getVal().v + matrix_i, in_dim);
        }
    }


    string typeSignature() const override {
        return getNodeType() + to_string(vector_->getDim());
    }

    Executor *generate() override;

private:
    Node *matrix_ = nullptr;
    Node *vector_ = nullptr;
    friend class MatrixAndVectorPointwiseMultiExecutor;
    friend class BatchedMatrixAndVectorPointwiseMultiNode;
};

class BatchedMatrixAndVectorPointwiseMultiNode :
    public BatchedNodeImpl<MatrixAndVectorPointwiseMultiNode> {
public:
    void init(Graph &graph, Node &matrix, BatchedNode &vec) {
        allocateBatch(matrix.getDim(), vec.batch().size());
        int i = 0;
        for (Node *node : batch()) {
            MatrixAndVectorPointwiseMultiNode *m =
                dynamic_cast<MatrixAndVectorPointwiseMultiNode *>(node);
            m->matrix_ = &matrix;
            m->vector_ = vec.batch().at(i++);
        }
        matrix.addParent(this);
        vec.addParent(this);
        graph.addNode(this);
    }

    void init(Graph &graph, BatchedNode &matrices, BatchedNode &vec) {
        vector<int> dims = matrices.getDims();
        allocateBatch(dims);
        setInputsPerNode({&matrices, &vec});
        afterInit(graph, {&matrices, &vec});
    }
};

#if USE_GPU
class MatrixAndVectorPointwiseMultiExecutor : public MatrixExecutor {
public:
    void forward() override {
        vector<dtype *> vals = getVals();
        for (Node *node : batch) {
            MatrixAndVectorPointwiseMultiNode *multi =
                static_cast<MatrixAndVectorPointwiseMultiNode *>(node);
#if TEST_CUDA
            multi->matrix_->val().copyFromHostToDevice();
            multi->vector_->val().copyFromHostToDevice();
#endif
            matrix_vals.push_back(multi->matrix_->getVal().value);
            vector_vals.push_back(multi->vector_->getVal().value);
            cols.push_back(multi->matrix_->getDim() / multi->vector_->getDim());
        }
        MatrixAndVectorPointwiseMultiNode *x = dynamic_cast<MatrixAndVectorPointwiseMultiNode *>(
                batch.front());
        int row = x->vector_->getDim();
        n3ldg_cuda::MatrixAndVectorPointwiseMultiForward(matrix_vals, vector_vals, batch.size(),
                row, cols, vals);
#if TEST_CUDA
        testForward();
        cout << "MatrixAndVectorPointwiseMultiExecutor forward tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype *> grads = getGrads();
        vector<dtype *> matrix_grads, vector_grads;
        for (Node *node : batch) {
            MatrixAndVectorPointwiseMultiNode *multi =
                static_cast<MatrixAndVectorPointwiseMultiNode *>(node);
            matrix_grads.push_back(multi->matrix_->getLoss().value);
            vector_grads.push_back(multi->vector_->getLoss().value);
        }
        MatrixAndVectorPointwiseMultiNode *x = dynamic_cast<MatrixAndVectorPointwiseMultiNode *>(
                batch.front());
        int row = x->vector_->getDim();
        n3ldg_cuda::MatrixAndVectorPointwiseMultiBackward(grads, matrix_vals, vector_vals,
                    batch.size(), row, cols, matrix_grads, vector_grads);
#if TEST_CUDA
        auto get_inputs = [](Node &node)->vector<pair<Node *, string>> {
            MatrixAndVectorPointwiseMultiNode &multi =
                static_cast<MatrixAndVectorPointwiseMultiNode&>(node);
            return {make_pair(multi.matrix_, "matrix"), make_pair(multi.vector_, "vector")};
        };
        for (Node *node : batch) {
            MatrixAndVectorPointwiseMultiNode *multi =
                static_cast<MatrixAndVectorPointwiseMultiNode *>(node);
            cout << boost::format("matrix:%1% vec:%2%") % multi->matrix_ % multi->vector_ << endl;
            cout << boost::format("matrix loss:%1% vec loss:%2%") % multi->matrix_->getLoss().value
                % multi->vector_->getLoss().value << endl;
        }
        testBackward(get_inputs);
        cout << "MatrixAndVectorPointwiseMultiExecutor backward tested" << endl;
#endif
    }
private:
    vector<dtype *> matrix_vals, vector_vals;
    vector<int> cols;
};
#else
class MatrixAndVectorPointwiseMultiExecutor : public MatrixExecutor {
public:
    int calculateFLOPs() override {
        cerr << "MatrixAndVectorPointwiseMultiExecutor - calculateFLOPs" << endl;
        abort();
        return 0;
    }

    int calculateActivations() override {
        cerr << "MatrixAndVectorPointwiseMultiExecutor - calculateActivations" << endl;
        abort();
        return 0;
    }
};
#endif

Executor *MatrixAndVectorPointwiseMultiNode::generate() {
    return new MatrixAndVectorPointwiseMultiExecutor;
}

class MatrixColSumNode : public UniInputNode, public Poolable<MatrixColSumNode> {
public:
    MatrixColSumNode() : UniInputNode("MatrixColSum") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void compute() override {
        Node &input = *getInput();
        int input_row  = input.getDim() / getDim();
        for (int i = 0; i < getDim(); ++i) {
            dtype sum = 0;
            for (int j = 0; j < input_row; ++j) {
                sum += input.getVal()[i * input_row + j];
            }
            val()[i] = sum;
        }
    }

    void backward() override {
        Node &input = *static_cast<MatrixNode *>(getInput());
        int input_row  = input.getDim() / getDim();
        for (int i = 0; i < getDim(); ++i) {
            for (int j = 0; j < input_row; ++j) {
                input.loss()[i * input_row + j] += getLoss()[i];
            }
        }
    }

    virtual string typeSignature() const override {
        int input_row = getInput()->getDim() / getDim();
        return "MatrixColSum-" + to_string(input_row);
    }

    Executor *generate() override;

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return true;
    }
};

class BatchedMatrixColSumNode : public BatchedNodeImpl<MatrixColSumNode> {
public:
    void init(Graph &graph, BatchedNode &input, int dim) {
        allocateBatch(dim, input.batch().size());
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }

    void init(Graph &graph, BatchedNode &input, const vector<int> &dims) {
        allocateBatch(dims);
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
};

#if USE_GPU
class MatrixColSumExecutor : public UniInputExecutor {
public:
    void forward() override {
        vector<dtype *> in_vals;
        MatrixColSumNode *first = dynamic_cast<MatrixColSumNode *>(batch.front());
        row = first->getInput()->getDim() / first->getDim();
        for (Node *node : batch) {
            MatrixColSumNode &sum = *static_cast<MatrixColSumNode *>(node);
            in_vals.push_back(sum.getInput()->getVal().value);
            cols.push_back(sum.getDim());
        }
        auto vals = getVals();
        n3ldg_cuda::MatrixColSumForward(in_vals, batch.size(), cols, row, vals);
#if TEST_CUDA
        testForward();
        cout << "MatrixColSumForward tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype *> grads = getGrads();
        vector<dtype *> in_grads;
        for (Node *node : batch) {
            MatrixColSumNode &sum = *static_cast<MatrixColSumNode *>(node);
            in_grads.push_back(sum.getInput()->getLoss().value);
            cols.push_back(sum.getDim());
        }
        n3ldg_cuda::MatrixColSumBackward(grads, batch.size(), cols, row, in_grads);
#if TEST_CUDA
        testBackward();
        cout << "MatrixColSumBackward tested" << endl;
#endif
    }

private:
    vector<int> cols;
    int row;
};
#else
class MatrixColSumExecutor : public Executor {
    int calculateFLOPs() override {
        return 0; // TODO
    }
};
#endif

Executor *MatrixColSumNode::generate() {
    return new MatrixColSumExecutor;
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

    void forward(Graph &graph, Node &matrix, Node &vec) {
        setInputs({&matrix, &vec});
        afterForward(graph, {&matrix, &vec});
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
    void init(Graph &graph, Node &matrix, BatchedNode &vec, int *dim = nullptr) {
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
        graph.addNode(this);
    }
};

#if USE_GPU
class MatrixAndVectorMultiExecutor : public Executor {
public:
    void forward() override {
#if TEST_CUDA
        auto get_inputs = [&](Node &node)->vector<Node *> {
            MatrixAndVectorMultiNode &multi = static_cast<MatrixAndVectorMultiNode&>(node);
            vector<Node *> inputs = {multi.matrix_, multi.vector_};
            return inputs;
        };
        testForwardInpputs(get_inputs);
        for (Node *node : batch) {
            MatrixAndVectorMultiNode *multi = static_cast<MatrixAndVectorMultiNode *>(node);
            multi->matrix_->val().copyFromHostToDevice();
            multi->vector_->val().copyFromHostToDevice();
        }
#endif
        auto vals = getVals();
        for (Node *node : batch) {
            MatrixAndVectorMultiNode *multi = static_cast<MatrixAndVectorMultiNode *>(node);
            matrix_vals_.push_back(multi->matrix_->getVal().value);
            vector_vals_.push_back(multi->vector_->getVal().value);
            cols_.push_back(multi->vector_->getDim());
        }
        MatrixAndVectorMultiNode *x = static_cast<MatrixAndVectorMultiNode *>(batch.front());
        row_ = x->getDim();
        n3ldg_cuda::MatrixAndVectorMultiForward(matrix_vals_, vector_vals_, batch.size(), row_,
                cols_, vals);
#if TEST_CUDA
        testForward();
        cout << "MatrixAndVectorMultiExecutor forward tested" << endl;
#endif
    }

    void backward() override {
        auto grads = getGrads();

        vector<dtype *> matrix_grads, vector_grads;
        for (Node *node : batch) {
            MatrixAndVectorMultiNode *multi = static_cast<MatrixAndVectorMultiNode *>(node);
            matrix_grads.push_back(multi->matrix_->getLoss().value);
            vector_grads.push_back(multi->vector_->getLoss().value);
        }
        n3ldg_cuda::MatrixAndVectorMultiBackward(grads, matrix_vals_, vector_vals_, batch.size(),
                row_, cols_, matrix_grads, vector_grads);
#if TEST_CUDA
        auto get_inputs = [&](Node &node) {
            MatrixAndVectorMultiNode &multi = static_cast<MatrixAndVectorMultiNode&>(node);
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

class MatrixTransposeNode : public UniInputNode, public Poolable<MatrixTransposeNode> {
public:
    MatrixTransposeNode() : UniInputNode("MatrixTranspose") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void compute() override {
        Node &input = *getInput();
        int input_row = getColumn();
        int input_col = getRow();
        for (int i = 0; i < input_row; ++i) {
            for (int j = 0; j < input_col; ++j) {
                val()[input_col * i + j] = input.getVal()[input_row * j + i];
            }
        }
    }

    void backward() override {
        Node &input = *getInput();
        int input_row = getColumn();
        int input_col = getRow();
        for (int i = 0; i < input_row; ++i) {
            for (int j = 0; j < input_col; ++j) {
                input.loss()[input_row * j + i] += getLoss()[input_col * i + j];
            }
        }
    }

    virtual string typeSignature() const override {
        return "MatrixTranspose-" + to_string(getColumn());
    }

    Executor *generate() override;

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }
};

class BatchedMatrixTransposeNode : public BatchedNodeImpl<MatrixTransposeNode> {
public:
    void init(Graph &graph, BatchedNode &input, int input_row) {
        allocateBatch(input.getDims());
        for (Node *node : batch()) {
            MatrixTransposeNode *m = dynamic_cast<MatrixTransposeNode *>(node);
            m->setColumn(input_row);
        }
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
};

#if USE_GPU
class MatrixTransposeExecutor : public UniInputExecutor {
public:
    void forward() override {
        vector<dtype *> matrices, vals;
        for (Node *node : batch) {
            MatrixTransposeNode *m = dynamic_cast<MatrixTransposeNode *>(node);
            matrices.push_back(m->getInput()->getVal().value);
            vals.push_back(m->getVal().value);
            input_cols_.push_back(m->getRow());
        }
        n3ldg_cuda::MatrixTransposeForward(matrices, batch.size(), batch.front()->getColumn(),
                input_cols_, vals);
#if TEST_CUDA
        testForward();
        cout << "MatrixTransposeForward forward tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype *> grads, matrix_grads;
        for (Node *node : batch) {
            MatrixTransposeNode *m = dynamic_cast<MatrixTransposeNode *>(node);
            matrix_grads.push_back(m->getInput()->getLoss().value);
            grads.push_back(m->getLoss().value);
        }
        n3ldg_cuda::MatrixTransposeBackward(grads, batch.size(), batch.front()->getColumn(),
                input_cols_, matrix_grads);
#if TEST_CUDA
        testBackward();
        cout << "MatrixTransposeBackward tested" << endl;
#endif
    }

private:
    vector<int> input_cols_;
};
#else
class MatrixTransposeExecutor : public Executor {
    int calculateFLOPs() override {
        return 0; // TODO
    }
};
#endif

Executor *MatrixTransposeNode::generate() {
    return new MatrixTransposeExecutor;
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
            cerr << boost::format("TranMatrixMulVectorNode setInputs dim:%1% max_col:%2%")
                % getDim() % max_col << endl;
            abort();
        }
    }

    void forward(Graph &graph, Node &matrix, Node &vec) {
        setInputs({&matrix, &vec});
        afterForward(graph, {&matrix, &vec});
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
    void init(Graph &graph, Node &matrix, BatchedNode &vec, const vector<int> *dims = nullptr) {
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
        graph.addNode(this);
    }
};

#if USE_GPU
class TranMatrixMulVectorExecutor : public Executor {
public:
    void forward() override {
        for (Node *node : batch) {
            TranMatrixMulVectorNode *t = dynamic_cast<TranMatrixMulVectorNode *>(node);
            cols_.push_back(t->getDim());
            matrices_.push_back(t->matrix_->getVal().value);
            vectors_.push_back(t->vector_->getVal().value);
            vals_.push_back(t->getVal().value);
        }
        n3ldg_cuda::TranMatrixMulVectorForward(matrices_, vectors_, batch.size(), cols_, row(),
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
        for (Node *node : batch) {
            TranMatrixMulVectorNode *t = dynamic_cast<TranMatrixMulVectorNode *>(node);
            grads.push_back(t->getLoss().value);
            matrix_grads.push_back(t->matrix_->getLoss().value);
            vector_grads.push_back(t->vector_->getLoss().value);
        }
        n3ldg_cuda::TranMatrixMulVectorBackward(grads, matrices_, vectors_, batch.size(),
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

namespace n3ldg_plus {

Node *concatToMatrix(Graph &graph, const vector<Node *> &inputs) {
    int input_dim = inputs.front()->getDim();
    MatrixConcatNode *node = MatrixConcatNode::newNode(inputs.size() * input_dim);
    node->forward(graph, inputs);
    return node;
}

Node *concatToMatrix(Graph &graph, NodeAbs &topo_input, const vector<Node *> &inputs) {
    int input_dim = inputs.front()->getDim();
    MatrixConcatNode *node = MatrixConcatNode::newNode(inputs.size() * input_dim);
    node->forward(graph, topo_input, inputs);
    return node;
}

BatchedNode *concatToMatrix(Graph &graph, BatchedNode &input, const vector<vector<Node *>> &cols) {
    BatchedMatrixConcatNode *concat = new BatchedMatrixConcatNode;
    concat->init(graph, input, cols);
    return concat;
}

Node *matrixPointwiseMultiply(Graph &graph, Node &matrix, Node &vec) {
    MatrixAndVectorPointwiseMultiNode *node = MatrixAndVectorPointwiseMultiNode::newNode(
            matrix.getDim());
    node->forward(graph, matrix, vec);
    return node;
}

BatchedNode *matrixPointwiseMultiply(Graph &graph, Node &matrix, BatchedNode &vec) {
    BatchedMatrixAndVectorPointwiseMultiNode *node = new BatchedMatrixAndVectorPointwiseMultiNode;
    node->init(graph, matrix, vec);
    return node;
}

BatchedNode *matrixPointwiseMultiply(Graph &graph, BatchedNode &matrices, BatchedNode &vec) {
    BatchedMatrixAndVectorPointwiseMultiNode *node = new BatchedMatrixAndVectorPointwiseMultiNode;
    node->init(graph, matrices, vec);
    return node;
}

Node *matrixColSum(Graph &graph, Node &input, int input_col) {
    if (input.getDim() % input_col != 0) {
        cerr << boost::format("input dim:%1% input col:%2%") % input.getDim() % input_col << endl;
        abort();
    }
    MatrixColSumNode *node = MatrixColSumNode::newNode(input_col);
    node->forward(graph, input);
    return node;
}

BatchedNode *matrixColSum(Graph &graph, BatchedNode &input, int input_col) {
    BatchedMatrixColSumNode *node = new BatchedMatrixColSumNode;
    node->init(graph, input, input_col);
    return node;
}

BatchedNode *matrixColSum(Graph &graph, BatchedNode &input, const vector<int> &cols) {
    BatchedMatrixColSumNode *node = new BatchedMatrixColSumNode;
    node->init(graph, input, cols);
    return node;
}

Node *matrixAndVectorMulti(Graph &graph, Node &matrix, Node &vec) {
    int dim = matrix.getDim() / vec.getDim();
    if (matrix.getDim() % vec.getDim() != 0) {
        cerr << boost::format("vec dim:%1% matrix dim:%2%") % vec.getDim() % matrix.getDim() <<
            endl;
        abort();
    }
    MatrixAndVectorMultiNode *node = MatrixAndVectorMultiNode::newNode(dim);
    node->forward(graph, matrix, vec);
    return node;
}

//BatchedNode *matrixAndVectorMulti(Graph &graph, BatchedNode &matrices, BatchedNode &vec,
//        int head_count) {
//    int i = 0;
//    for (Node *matrix : matrices.batch()) {
//        if (matrix->getDim() % head_count != 0) {
//            cerr << boost::format("matrix dim:%1% head_count:%2%") % matrix->getDim() %
//                head_count << endl;
//            abort();
//        }
//        if (vec.batch().at(i)->getDim() % head_count != 0) {
//            cerr << boost::format("vec dim:%1% head_count:%2%") % vec.batch().at(i)->getDim() %
//                head_count << endl;
//            abort();
//        }
//        if (matrix->getDim() % vec.batch().at(i)->getDim() != 0) {
//            cerr << boost::format("vec dim:%1% matrix dim:%2%") % vec.batch().at(i)->getDim() %
//                matrix->getDim() << endl;
//            abort();
//        }
//        ++i;
//    }

//    BatchedMatrixAndVectorMultiNode *node = new BatchedMatrixAndVectorMultiNode;
//    node->init(graph, matrices, vec);
//    return node;
//}

BatchedNode *matrixAndVectorMulti(Graph &graph, Node &matrix, BatchedNode &vec,
        int *dim = nullptr) {
    if (matrix.getDim() % vec.getDim() != 0) {
        cerr << boost::format("vec dim:%1% matrix dim:%2%") % vec.getDim() % matrix.getDim() <<
            endl;
        abort();
    }

    BatchedMatrixAndVectorMultiNode *node = new BatchedMatrixAndVectorMultiNode;
    node->init(graph, matrix, vec, dim);
    return node;
}

Node *transposeMatrix(Graph &graph, Node &matrix, int input_row) {
    if (matrix.getDim() % input_row != 0) {
        cerr << boost::format("matrix dim:%1% input_row:%2%") % matrix.getDim() % input_row <<
            endl;
        abort();
    }
    MatrixTransposeNode *node = MatrixTransposeNode::newNode(matrix.getDim());
    node->setColumn(input_row);
    node->forward(graph, matrix);
    return node;
}

BatchedNode *transposeMatrix(Graph &graph, BatchedNode &input, int input_row) {
    auto dims = input.getDims();
    for (int dim : dims) {
        if (dim % input_row != 0) {
            cerr << boost::format("matrix dim:%1% input_row:%2%") % input.getDim() % input_row <<
                endl;
            abort();
        }
    }
    BatchedMatrixTransposeNode *node = new BatchedMatrixTransposeNode;
    node->init(graph, input, input_row);
    return node;
}

Node *tranMatrixMulVector(Graph &graph, Node &matrix, Node &vec, int dim) {
    TranMatrixMulVectorNode *node = TranMatrixMulVectorNode::newNode(dim);
    node->forward(graph, matrix, vec);
    return node;
}

Node *tranMatrixMulVector(Graph &graph, Node &matrix, Node &vec) {
    int dim = matrix.getDim() / vec.getDim();
    return tranMatrixMulVector(graph, matrix, vec, dim);
}

BatchedNode *tranMatrixMulVector(Graph &graph, Node &matrix, BatchedNode &vec,
        const vector<int> *dims = nullptr) {
    BatchedTranMatrixMulVectorNode *node = new BatchedTranMatrixMulVectorNode;
    node->init(graph, matrix, vec, dims);
    return node;
}

}
#endif
