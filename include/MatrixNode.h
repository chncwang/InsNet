#include "Node.h"
#include "Graph.h"

template<typename T>
class MatrixNode : public Node, Poolable<T> {
public:
    explicit MatrixNode(const string &node_type) : Node(node_type + "-matrix") {}

    int getColumn() const {
        return column_;
    }

    int getRow() const {
        return getDim() / column_;
    }

    virtual int getKey() const override {
        return getDim();
    }

    virtual void initNode(int dim) override {
        init(dim);
    }

    virtual void setNodeDim(int dim) override {
        setDim(dim);
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

template<typename T>
class MatrixExecutor : public Executor {
    int getColumn() const {
        return static_cast<MatrixNode<T> *>(batch.front())->getColumn();
    }
};

class MatrixConcatNode : public MatrixNode<MatrixConcatNode> {
public:
    MatrixConcatNode(): MatrixNode("concat") {}

    void forward(Graph &graph, vector<Node *> &inputs) {
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
        graph.addNode(this);
    }

    void compute() override {
        for (int i = 0; i < in_nodes.size(); ++i) {
            int offset = i * getInputDim();
            for (int j = 0; j < getInputDim(); ++j) {
                val().v[offset + j] = in_nodes.at(i)->getVal().v[j];
            }
        }
    }

    void backward() override {
        for (int i = 0; i < in_nodes.size(); ++i) {
            int offset = i * getInputDim();
            for (int j = 0; j < getInputDim(); ++j) {
                in_nodes.at(i)->loss()[j] += loss()[offset + j];
            }
        }
    }

    int getInputDim() const {
        return in_nodes.front()->getDim();
    }

    bool typeEqual(Node *other) override {
        MatrixConcatNode *concat = static_cast<MatrixConcatNode*>(other);
        return Node::typeEqual(other) && concat->getInputDim() == getInputDim();
    }

    string typeSignature() const override {
        return Node::typeSignature() + "-" + to_string(getInputDim());
    }

    Executor* generate() override;
private:
    vector<Node *> in_nodes;
};

#if USE_GPU
class MatrixConcatExecutor : public MatrixExecutor<MatrixConcatNode> {
public:
    void forward() override {
        int count = batch.size();
        vector<int> in_counts;
        for (Node *node : batch) {
            MatrixConcatNode *concat = static_cast<MatrixConcatNode*>(node);
            in_counts.push_back(concat->getInputDim());
        }
        int max_in_count = *max(in_counts.begin(), in_counts.end());
        vector<dtype *> vals, in_vals;
        int node_i = -1;
        for (Node *node : batch) {
            ++node_i;
            MatrixConcatNode *concat = static_cast<MatrixConcatNode*>(node);
            vals.push_back(concat->getVal().value);
            for (int i = 0; i < max_in_count; ++i) {
                in_vals.push_back(i < max_in_count ? in_vals.at(i) : nullptr);
            }
        }
    }

    void backward() override {

    }
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
