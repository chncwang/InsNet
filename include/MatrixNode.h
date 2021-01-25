#ifndef N3LDG_PLUS_MATRX_NODE_H
#define N3LDG_PLUS_MATRX_NODE_H

#include "Node.h"
#include "Graph.h"

class MatrixNode : public AtomicNode {
public:
    explicit MatrixNode(const string &node_type) : AtomicNode(node_type + "-matrix") {}
};

class MatrixExecutor : public Executor {
public:
    int getRow() const {
        return static_cast<MatrixNode *>(batch.front())->getRow();
    }

    vector<int> getCols() const {
        vector<int> cols;
        cols.reserve(batch.size());
        for (AtomicNode *node : batch) {
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

    void forward(Graph &graph, const vector<AtomicNode *> &inputs) {
        int input_dim = inputs.front()->getDim();
        for (auto it = inputs.begin() + 1; it != inputs.end(); ++it) {
            if (input_dim != (*it)->getDim()) {
                cerr << "MatrixConcatNode - forward inconsistent input dims" << endl;
                abort();
            }
        }

        in_nodes = inputs;
        for (AtomicNode *in : inputs) {
            in->addParent(this);
        }
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

    const vector<AtomicNode *> &getInputs() const {
        return in_nodes;
    }

private:
    vector<AtomicNode *> in_nodes;
};

#if USE_GPU
class MatrixConcatExecutor : public MatrixExecutor {
public:
    void forward() override {
        for (AtomicNode *node : batch) {
            MatrixConcatNode *concat = static_cast<MatrixConcatNode*>(node);
            in_counts.push_back(concat->getColumn());
        }
        max_in_count = *max_element(in_counts.begin(), in_counts.end());
        vector<dtype *> vals, in_vals;
        int node_i = -1;
        for (AtomicNode *node : batch) {
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
        for (AtomicNode *node : batch) {
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
        auto get_inputs = [&](AtomicNode &node) {
            vector<pair<AtomicNode*, string>> pairs;
            MatrixConcatNode &concat = static_cast<MatrixConcatNode&>(node);
            for (AtomicNode *input : concat.getInputs()) {
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

    void forward(Graph &graph, AtomicNode &matrix, AtomicNode &vec) {
        if (getDim() % vec.getDim() != 0) {
            cerr << boost::format("MatrixConcatNode forward - dim:%1% vec dim:%2%") % getDim() %
                vec.getDim() << endl;
            abort();
        }
        matrix.addParent(this);
        vec.addParent(this);
        matrix_ = &matrix;
        vector_ = &vec;
        graph.addNode(this);
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
        return "MatrixAndVectorPointwiseMulti-" + to_string(vector_->getDim());
    }

    Executor *generate() override;

private:
    AtomicNode *matrix_ = nullptr;
    AtomicNode *vector_ = nullptr;
    friend class MatrixAndVectorPointwiseMultiExecutor;
};

#if USE_GPU
class MatrixAndVectorPointwiseMultiExecutor : public MatrixExecutor {
public:
    void forward() override {
        vector<dtype *> vals = getVals();
        for (AtomicNode *node : batch) {
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
        for (AtomicNode *node : batch) {
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
        auto get_inputs = [](AtomicNode &node)->vector<pair<AtomicNode *, string>> {
            MatrixAndVectorPointwiseMultiNode &multi =
                static_cast<MatrixAndVectorPointwiseMultiNode&>(node);
            return {make_pair(multi.matrix_, "matrix"), make_pair(multi.vector_, "vector")};
        };
        for (AtomicNode *node : batch) {
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
        AtomicNode &input = *getInput();
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
        AtomicNode &input = *static_cast<MatrixNode *>(getInput());
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
    virtual bool isDimLegal(const AtomicNode &input) const override {
        return true;
    }
};

#if USE_GPU
class MatrixColSumExecutor : public UniInputExecutor {
public:
    void forward() override {
        vector<dtype *> in_vals;
        MatrixColSumNode *first = dynamic_cast<MatrixColSumNode *>(batch.front());
        row = first->getInput()->getDim() / first->getDim();
        for (AtomicNode *node : batch) {
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
        for (AtomicNode *node : batch) {
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

class MatrixAndVectorMultiNode : public AtomicNode, public Poolable<MatrixAndVectorMultiNode> {
public:
    MatrixAndVectorMultiNode() : AtomicNode("MatrixAndVectorMulti") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void forward(Graph &graph, AtomicNode &matrix, AtomicNode &vec, int head_count) {
        head_count_ = head_count;
        matrix.addParent(this);
        matrix_ = &matrix;
        vec.addParent(this);
        vector_ = &vec;
        graph.addNode(this);
    }

    void compute() override {
        int col = vector_->getDim() / head_count_;
        int row = getDim();
        int head_dim = getDim() / head_count_;
        dtype *matrix_head = new dtype[head_dim * col];
        for (int i = 0; i < head_count_; ++i) {
            for (int j = 0; j < col; ++j) {
                for (int k = 0; k < head_dim; ++k) {
                    matrix_head[head_dim * j + k] =
                        matrix_->getVal()[row * j + k + i * head_dim];
                }
            }
            Mat(val().v + head_dim * i, head_dim, 1) = Mat(matrix_head, head_dim, col) *
                Mat(vector_->getVal().v + col * i, col, 1);
        }
        delete [] matrix_head;
    }

    void backward() override {
        int col = vector_->getDim() / head_count_;
        int row = getDim();
        int head_dim = getDim() / head_count_;
        dtype *matrix_head = new dtype[head_dim * col];
        for (int i = 0; i < head_count_; ++i) {
            for (int j = 0; j < col; ++j) {
                for (int k = 0; k < head_dim; ++k) {
                    matrix_head[head_dim * j + k] =
                        matrix_->getLoss()[row * j + k + i * head_dim];
                }
            }

            Mat(matrix_head, head_dim, col) = Mat(getLoss().v + head_dim * i, head_dim, 1) *
                Mat(vector_->getVal().v + col * i, col, 1).transpose();

            for (int j = 0; j < col; ++j) {
                for (int k = 0; k < head_dim; ++k) {
                    matrix_->loss()[row * j + k + i * head_dim] += matrix_head[head_dim * j + k];
                }
            }

            for (int j = 0; j < col; ++j) {
                for (int k = 0; k < head_dim; ++k) {
                    matrix_head[head_dim * j + k] =
                        matrix_->getVal()[row * j + k + i * head_dim];
                }
            }

            Mat(vector_->loss().v + i * col, col, 1) +=
                Mat(matrix_head, head_dim, col).transpose() *
                Mat(getLoss().v + i * head_dim, head_dim, 1);
        }
        delete [] matrix_head;
    }

    Executor * generate() override;

    string typeSignature() const override {
        return AtomicNode::typeSignature() + to_string(head_count_);
    }

private:
    AtomicNode *vector_ = nullptr;
    AtomicNode *matrix_ = nullptr;
    int head_count_ = 1;
    friend class MatrixAndVectorMultiExecutor;
};

#if USE_GPU
class MatrixAndVectorMultiExecutor : public Executor {
public:
    void forward() override {
#if TEST_CUDA
        auto get_inputs = [&](AtomicNode &node)->vector<AtomicNode *> {
            MatrixAndVectorMultiNode &multi = static_cast<MatrixAndVectorMultiNode&>(node);
            vector<AtomicNode *> inputs = {multi.matrix_, multi.vector_};
            return inputs;
        };
        testForwardInpputs(get_inputs);
        for (AtomicNode *node : batch) {
            MatrixAndVectorMultiNode *multi = static_cast<MatrixAndVectorMultiNode *>(node);
            multi->matrix_->val().copyFromHostToDevice();
            multi->vector_->val().copyFromHostToDevice();
        }
#endif
        auto vals = getVals();
        int head_count = dynamic_cast<MatrixAndVectorMultiNode*>(batch.front())->head_count_;
        for (AtomicNode *node : batch) {
            MatrixAndVectorMultiNode *multi = static_cast<MatrixAndVectorMultiNode *>(node);
            matrix_vals_.push_back(multi->matrix_->getVal().value);
            vector_vals_.push_back(multi->vector_->getVal().value);
            cols_.push_back(multi->vector_->getDim() / head_count);
        }
        MatrixAndVectorMultiNode *x = static_cast<MatrixAndVectorMultiNode *>(batch.front());
        row_ = x->getDim() / head_count;
        n3ldg_cuda::MatrixAndVectorMultiForward(matrix_vals_, vector_vals_, batch.size(),
                head_count, row_, cols_, vals);
#if TEST_CUDA
        testForward();
        cout << "MatrixAndVectorMultiExecutor forward tested" << endl;
#endif
    }

    void backward() override {
        auto grads = getGrads();

        vector<dtype *> matrix_grads, vector_grads;
        for (AtomicNode *node : batch) {
            MatrixAndVectorMultiNode *multi = static_cast<MatrixAndVectorMultiNode *>(node);
            matrix_grads.push_back(multi->matrix_->getLoss().value);
            vector_grads.push_back(multi->vector_->getLoss().value);
        }
        int head_count = dynamic_cast<MatrixAndVectorMultiNode*>(batch.front())->head_count_;
        n3ldg_cuda::MatrixAndVectorMultiBackward(grads, matrix_vals_, vector_vals_, batch.size(),
                head_count, row_, cols_, matrix_grads, vector_grads);
#if TEST_CUDA
        auto get_inputs = [&](AtomicNode &node) {
            MatrixAndVectorMultiNode &multi = static_cast<MatrixAndVectorMultiNode&>(node);
            vector<pair<AtomicNode *, string>> inputs = {make_pair(multi.matrix_, "matrix"),
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
        AtomicNode &input = *getInput();
        int input_row = getColumn();
        int input_col = getRow();
        for (int i = 0; i < input_row; ++i) {
            for (int j = 0; j < input_col; ++j) {
                val()[input_col * i + j] = input.getVal()[input_row * j + i];
            }
        }
    }

    void backward() override {
        AtomicNode &input = *getInput();
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
    virtual bool isDimLegal(const AtomicNode &input) const override {
        return input.getDim() == getDim();
    }
};

#if USE_GPU
class MatrixTransposeExecutor : public UniInputExecutor {
public:
    void forward() override {
        vector<dtype *> matrices, vals;
        for (AtomicNode *node : batch) {
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
        for (AtomicNode *node : batch) {
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

namespace n3ldg_plus {

MatrixNode *concatToMatrix(Graph &graph, const vector<AtomicNode *> &inputs) {
    int input_dim = inputs.front()->getDim();
    MatrixConcatNode *node = MatrixConcatNode::newNode(inputs.size() * input_dim);
    node->forward(graph, inputs);
    return node;
}

MatrixNode *matrixPointwiseMultiply(Graph &graph, AtomicNode &matrix, AtomicNode &vec) {
    MatrixAndVectorPointwiseMultiNode *node = MatrixAndVectorPointwiseMultiNode::newNode(
            matrix.getDim());
    node->forward(graph, matrix, vec);
    return node;
}

AtomicNode *matrixColSum(Graph &graph, AtomicNode &input, int input_col) {
    if (input.getDim() % input_col != 0) {
        cerr << boost::format("input dim:%1% input col:%2%") % input.getDim() % input_col << endl;
        abort();
    }
    MatrixColSumNode *node = MatrixColSumNode::newNode(input_col);
    node->forward(graph, input);
    return node;
}

AtomicNode *matrixAndVectorMulti(Graph &graph, AtomicNode &matrix, AtomicNode &vec, int head_count) {
    int dim = matrix.getDim() * head_count / vec.getDim();
    if (matrix.getDim() % head_count != 0) {
        cerr << boost::format("matrix dim:%1% head_count:%2%") % matrix.getDim() % head_count <<
            endl;
        abort();
    }
    if (vec.getDim() % head_count != 0) {
        cerr << boost::format("vec dim:%1% head_count:%2%") % vec.getDim() % head_count <<
            endl;
        abort();
    }
    if (matrix.getDim() % vec.getDim() != 0) {
        cerr << boost::format("vec dim:%1% matrix dim:%2%") % vec.getDim() % matrix.getDim() <<
            endl;
        abort();
    }
    MatrixAndVectorMultiNode *node = MatrixAndVectorMultiNode::newNode(dim);
    node->forward(graph, matrix, vec, head_count);
    return node;
}

AtomicNode *transposeMatrix(Graph &graph, AtomicNode &matrix, int input_row) {
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

}
#endif
