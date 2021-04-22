#include "n3ldg-plus/operator/add.h"

using std::vector;
using std::cerr;
using std::endl;
using std::string;
using std::to_string;

namespace n3ldg_plus {

class PAddNode : public Node, public Poolable<PAddNode> {
public:
    PAddNode() : Node("point-add") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void setInputs(const vector<Node *> &ins) override {
        this->ins_ = ins;
    }

    void connect(const vector<Node *> &x) {
        if (x.empty()) {
            cerr << "empty inputs for add" << endl;
            abort();
        }

        for (int i = 0; i < x.size(); i++) {
            if (x.at(i)->getDim() != getDim()) {
                cerr << "dim does not match" << endl;
                abort();
            }
        }
        setInputs(x);
        afterConnect(x);
    }

    void compute() override {
        int nSize = ins_.size();
        val().zero();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < getDim(); idx++) {
                val()[idx] += ins_.at(i)->val()[idx];
            }
        }
    }

    void backward() override {
        int nSize = ins_.size();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < getDim(); idx++) {
                ins_.at(i)->loss()[idx] += loss()[idx];
            }
        }
    }

    Executor *generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + "-" + to_string(ins_.size());
    }

private:
    vector<Node *> ins_;

    friend class BatchedPAddNode;
    friend class PAddExecutor;
};

class BatchedPAddNode : public BatchedNodeImpl<PAddNode> {
public:
    void init(const vector<BatchedNode *> &inputs) {
        allocateBatch(inputs.front()->getDim(), inputs.front()->batch().size());

        for (BatchedNode *in : inputs) {
            if (in->getDim() != getDim()) {
                cerr << "dim does not match" << endl;
                abort();
            }
        }

        setInputsPerNode(inputs);
        afterInit( inputs);
    }
};

#if USE_GPU
class PAddExecutor : public Executor {
public:
    void forward() override {
        int count = batch.size();

        for (int i = 0; i < inCount(); ++i) {
            vector<dtype*> ins;
            ins.reserve(count);
            for (Node * n : batch) {
                PAddNode *padd = dynamic_cast<PAddNode*>(n);
                ins.push_back(padd->ins_.at(i)->val().value);
#if TEST_CUDA
                cuda::Assert(padd->ins_.at(i)->val().verify("PAdd forward input"));
#endif
            }
        }
        vector<dtype*> in_vals, outs;
        in_vals.reserve(count * inCount());
        outs.reserve(count);
        dims_.reserve(count);
        for (Node * n : batch) {
            PAddNode &padd = dynamic_cast<PAddNode&>(*n);
            outs.push_back(padd.val().value);
            dims_.push_back(padd.getDim());
            for (Node *in : padd.ins_) {
                in_vals.push_back(in->getVal().value);
            }
        }
        max_dim_ = *max_element(dims_.begin(), dims_.end());
        cuda::PAddForward(in_vals, count, dims_, max_dim_, inCount(), outs, dim_arr_);
#if TEST_CUDA
        testForward();
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype *> out_grads, in_grads;
        out_grads.reserve(count);
        in_grads.reserve(count * inCount());
        for (Node *n : batch) {
            PAddNode &padd = dynamic_cast<PAddNode&>(*n);
            out_grads.push_back(padd.getLoss().value);
            for (Node *in : padd.ins_) {
                in_grads.push_back(in->getLoss().value);
            }
        }
        cuda::PAddBackward(out_grads, count, max_dim_, inCount(), in_grads, dim_arr_);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        for (Node *n : batch) {
            PAddNode *add = dynamic_cast<PAddNode*>(n);
            for (Node *in : add->ins_) {
                cuda::Assert(in->loss().verify("PAddExecutor backward"));
            }
        }
        cout << "PAddExecutor backward tested" << endl;
#endif
    }

private:
    int inCount() {
        return dynamic_cast<PAddNode &>(*batch.front()).ins_.size();
    }

    vector<int> dims_;
    int max_dim_;
    cuda::IntArray dim_arr_;
};
#else
class PAddExecutor : public Executor {
public:
    int calculateFLOPs() override {
        int sum = 0;
        for (Node *node : batch) {
            PAddNode *add = dynamic_cast<PAddNode*>(node);
            sum += add->getDim() * add->ins_.size();
        }
        return sum;
    }
};
#endif

Executor *PAddNode::generate() {
    return new PAddExecutor();
}

Node *add(const vector<Node*> &inputs) {
    int dim = inputs.front()->getDim();
    PAddNode *result = PAddNode::newNode(dim);
    result->connect(inputs);
    return result;
}

BatchedNode *addInBatch(const vector<BatchedNode *> &inputs) {
    BatchedPAddNode *node = new BatchedPAddNode;
    node->init(inputs);
    return node;
}

}
