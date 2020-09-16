#ifndef N3LDG_PLUS_MATRX_NODE_H
#define N3LDG_PLUS_MATRX_NODE_H

#include "Node.h"
#include "Graph.h"

class MatrixNode : public Node {
public:
    explicit MatrixNode(const string &node_type) : Node(node_type + "-matrix") {}

    int getColumn() const {
        return column_;
    }

    int getRow() const {
        return getDim() / column_;
    }

protected:
    void setColumn(int column) {
        if (getDim() % column != 0) {
            cerr << boost::format("MatrixNode setColumn - dim:%1% column:%2%") % getDim() % column
                << endl;
            abort();
        }
        column_ = column;
    }

private:
    int column_;
};

class MatrixExecutor : public Executor {
public:
    int getRow() const {
        return static_cast<MatrixNode *>(batch.front())->getRow();
    }
};

class MatrixConcatNode : public MatrixNode, public Poolable<MatrixConcatNode> {
public:
    virtual int getKey() const override {
        return getDim();
    }

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

    bool typeEqual(Node *other) override {
        MatrixConcatNode *concat = static_cast<MatrixConcatNode*>(other);
        return concat->getNodeType() == getNodeType() && concat->getRow() == getRow();
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
};

#if USE_GPU
class MatrixConcatExecutor : public MatrixExecutor {
public:
    void forward() override {
        for (Node *node : batch) {
            MatrixConcatNode *concat = static_cast<MatrixConcatNode*>(node);
//            cout << "in count:" << concat->getColumn() << endl;
            in_counts.push_back(concat->getColumn());
        }
        max_in_count = *max_element(in_counts.begin(), in_counts.end());
//        cout << "max_in_count:%d" << max_in_count << endl;
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

namespace n3ldg_plus {

MatrixNode *concatToMatrix(Graph &graph, const vector<Node *> &inputs) {
    int input_dim = inputs.front()->getDim();
    MatrixConcatNode *node = MatrixConcatNode::newNode(inputs.size() * input_dim);
    node->forward(graph, inputs);
    return node;
}

}
#endif
