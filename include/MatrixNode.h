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
        setInputs(inputs);
        for (Node *in : inputs) {
            in->addParent(this);
        }
        setColumn(inputs.size());
        graph.addNode(this);
    }

    void forward(Graph &graph, NodeAbs &topo_input, const vector<Node *> &inputs) {
        setInputs(inputs);
        topo_input.addParent(this);
        setColumn(inputs.size());
        graph.addNode(this);
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
    void init(Graph &graph, BatchedNode &input, int group) {
        if (input.batch().size() % group != 0) {
            cerr << boost::format("input batch size:%1% group:%2%") % input.batch().size() % group
                << endl;
        }
        int input_count = input.batch().size() / group;
        allocateBatch(input.getDim() * input_count, group);

        int group_i = 0;
        for (Node *node : batch()) {
            MatrixConcatNode *m = dynamic_cast<MatrixConcatNode *>(node);
            vector<Node *> in_nodes;
            for (int i = 0; i < input_count; ++i) {
                in_nodes.push_back(input.batch().at(group_i * input_count + i));
            }
            ++group_i;
            m->in_nodes = move(in_nodes);
            m->setColumn(input_count);
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

    void init(Graph &graph, BatchedNode &matrix, BatchedNode &vec, int *dim = nullptr) {
        int group = matrix.batch().size();
        if (vec.batch().size() % group != 0) {
            cerr << boost::format("BatchedTranMatrixMulVectorNode init vec size:%1% group:%2%") %
                vec.batch().size() % group << endl;
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

    void init(Graph &graph, BatchedNode &matrix, BatchedNode &vec,
            const vector<int> *dims = nullptr) {
        int group = matrix.batch().size();
        if (vec.batch().size() % group != 0) {
            cerr << boost::format("BatchedTranMatrixMulVectorNode init vec size:%1% group:%2%") %
                vec.batch().size() % group << endl;
            abort();
        }

        if (dims == nullptr) {
            int dim = matrix.getDim() / vec.getDim();
            allocateBatch(dim, vec.batch().size());
        } else {
            vector<int> overall_dims;
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

Node *concatToMatrix(Graph &graph, NodeAbs &topo_input) {
    const auto &inputs = topo_input.batch();
    int input_dim = inputs.front()->getDim();
    MatrixConcatNode *node = MatrixConcatNode::newNode(inputs.size() * input_dim);
    node->forward(graph, topo_input, inputs);
    return node;
}

BatchedNode *concatToMatrix(Graph &graph, BatchedNode &input, int group) {
    BatchedMatrixConcatNode *node = new BatchedMatrixConcatNode;
    node->init(graph, input, group);
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

BatchedNode *matrixAndVectorMulti(Graph &graph, BatchedNode &matrix, BatchedNode &vec,
        int *dim = nullptr) {
    BatchedMatrixAndVectorMultiNode *node = new BatchedMatrixAndVectorMultiNode;
    node->init(graph, matrix, vec, dim);
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

BatchedNode *tranMatrixMulVector(Graph &graph, BatchedNode &matrix, BatchedNode &vec,
        const vector<int> *dims = nullptr) {
    BatchedTranMatrixMulVectorNode *node = new BatchedTranMatrixMulVectorNode;
    node->init(graph, matrix, vec, dims);
    return node;
}

}
#endif
