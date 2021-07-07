#include "insnet/operator/param.h"
#include "fmt/core.h"

using namespace std;

namespace insnet {

class ParamNode : public Node, public Poolable<ParamNode> {
public:
    ParamNode() : Node("paramNode") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void connect(Graph &graph, BaseParam &param) {
        if (size() != param.val().size) {
            cerr << fmt::format("node size:{} param size:{}", size(), param.val().size) << endl;
            abort();
        }
        param_ = &param;
        graph.addNode(this);
    }

    string typeSignature() const override {
        return Node::getNodeType() + "-" + addressToString(param_);
    }

    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }

    Executor* generate() override;

protected:
    int forwardOnlyInputValSize() override {
        return 0;
    }

    bool isValForwardOnly() const override {
        return true;
    }


private:
    BaseParam *param_;
    friend class ParamExecutor;
};

Node* param(Graph &graph, BaseParam &param) {
    ParamNode *node = ParamNode::newNode(param.val().size);
    node->connect(graph, param);
    return node;
}

#if USE_GPU
class ParamExecutor : public Executor {
public:
};
#else
class ParamExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }

    void forward () override {
        if (batch.size() > 1) {
            cerr << "The param op should only be used once in a computation graph. - batch size:"
                << batch.size() << endl;
            abort();
        }
        ParamNode &node = dynamic_cast<ParamNode &>(*batch.front());
        node.val().vec() = node.param_->val().vec();
    }

    void backward() override {
        ParamNode &node = dynamic_cast<ParamNode &>(*batch.front());
        node.param_->grad().vec() += node.grad().vec();
    }
};
#endif

Executor *ParamNode::generate() {
    return new ParamExecutor;
}

}
