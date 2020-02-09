#ifndef N3LDG_PLUS_PARAM_OP_H
#define N3LDG_PLUS_PARAM_OP_H

#include "Param.h"
#include "Graph.h"

class ParamRowExecutor;
class ParamRowNode : public Node {
public:
    ParamRowNode(Param &param, int row_index): Node("param-row"), param_(&param),
    row_index_(row_index) {}

    Param* getParam() const {
        return param_;
    }

    void forward(Graph &graph) {
        if (getDim() > param_->inDim()) {
            cerr << boost::format("ParamRowNode - forward - node dim is %1%, but param dim is %2%")
                % getDim() % param_->inDim() << endl;
            abort();
        }
        graph.addNode(this);
    }

    void compute() override {
        for (int i = 0; i < getDim(); ++i) {
            val().v[i] = param_->val.mat()(0, i);
        }
    }

    void backward() override {
        // TODO
    }

    Executor* generate() override;

    bool typeEqual(Node* other) override {
        ParamRowNode *paramrow = static_cast<ParamRowNode*>(other);
        return Node::typeEqual(other) && param_ == paramrow->param_ &&
            row_index_ == paramrow->row_index_;
    }

    string typeSignature() const override {
        return Node::typeSignature() + "-" + addressToString(param_) + "-" +
            std::to_string(row_index_);
    }

private:
    Param *param_;
    int row_index_;

    friend class ParamRowExecutor;
};

namespace n3ldg_plus {

Node *paramRow(Graph &graph, Param &param, int row_index, int dim) {
    ParamRowNode *node = new ParamRowNode(param, row_index);
    node->init(dim);
    node->forward(graph);
    return node;
}

}

#if USE_GPU
class ParamRowExecutor : public Executor {
public:
    void forward() override {
        PageLockedVector<dtype*> vals;
        transform(batch.begin(), batch.end(), back_inserter(vals), gpu_get_node_val);
        ParamRowNode *node = static_cast<ParamRowNode*>(batch.front());
        n3ldg_cuda::ParamRowForward(node->getParam()->val.value, node->row_index_,
                node->getParam()->outDim(), batch.size(), getDim(), vals,
                n3ldg_cuda::StreamManager::ins().stream(VAL_STREAM));
#if TEST_CUDA
        Executor::testForward();
#endif
    }
};
#else
class ParamRowExecutor : public Executor {};
#endif

Executor* ParamRowNode::generate() {
    return new ParamRowExecutor;
}

#endif
