#include "n3ldg-plus/operator/broadcast.h"

using std::string;
using std::to_string;

namespace n3ldg_plus {

class BroadcastNode : public UniInputNode, public Poolable<BroadcastNode> {
public:
    BroadcastNode() : UniInputNode("broadcast") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    Executor *generate() override;

    void compute() override {
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            Vec(val().v + row * i, row) = getInput().getVal().vec();
        }
    }

    void backward() override {
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            getInput().loss().vec() += Vec(loss().v + row * i, row);
        }
    }

    string typeSignature() const override {
        return Node::getNodeType() + to_string(getDim() / getColumn());
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return getColumn() * input.getDim() == getDim();
    }
};

class BroadcastExecutor : public UniInputExecutor {
public:
    int calculateFLOPs() override {
        return 0; // TODO
    }
};

Executor *BroadcastNode::generate() {
    return new BroadcastExecutor;
}

Node *broadcast(Node &input, int count) {
    BroadcastNode *result = BroadcastNode::newNode(input.getDim() * count);
    result->setColumn(count);
    result->connect(input);
    return result;
}

}
