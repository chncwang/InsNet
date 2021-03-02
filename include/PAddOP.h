#ifndef PAddOP
#define PAddOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

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

    void forward(Graph &cg, const vector<Node *>& x) {
        if (x.empty()) {
            std::cerr << "empty inputs for add" << std::endl;
            abort();
        }

        for (int i = 0; i < x.size(); i++) {
            if (x.at(i)->getDim() != getDim()) {
                std::cerr << "dim does not match" << std::endl;
                abort();
            }
        }
        setInputs(x);
        afterForward(cg, x);
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

    PExecutor generate() override;

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
    void init(Graph &graph, const vector<BatchedNode *> &inputs) {
        allocateBatch(inputs.front()->getDim(), inputs.front()->batch().size());

        for (BatchedNode *in : inputs) {
            if (in->getDim() != getDim()) {
                std::cerr << "dim does not match" << std::endl;
                abort();
            }
        }

        setInputsPerNode(inputs);
        afterInit(graph, inputs);
    }
};

namespace n3ldg_plus {
    Node *add(Graph &graph, const vector<Node*> &inputs) {
        int dim = inputs.front()->getDim();
        PAddNode *result = PAddNode::newNode(dim);
        result->forward(graph, inputs);
        return result;
    }

    BatchedNode *addInBatch(Graph &graph, const vector<BatchedNode *> &inputs) {
        BatchedPAddNode *node = new BatchedPAddNode;
        node->init(graph, inputs);
        return node;
    }
}

#if USE_GPU
class PAddExecutor : public Executor {
public:
    void forward() override {
        int count = batch.size();

        for (int i = 0; i < inCount(); ++i) {
            std::vector<dtype*> ins;
            ins.reserve(count);
            for (Node * n : batch) {
                PAddNode *padd = dynamic_cast<PAddNode*>(n);
                ins.push_back(padd->ins_.at(i)->val().value);
#if TEST_CUDA
                n3ldg_cuda::Assert(padd->ins_.at(i)->val().verify("PAdd forward input"));
#endif
            }
        }
        std::vector<dtype*> in_vals, outs;
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
        n3ldg_cuda::PAddForward(in_vals, count, dims_, max_dim_, inCount(), outs, dim_arr_);
#if TEST_CUDA
        testForward();
#endif
    }

    void backward() override {
        int count = batch.size();
        std::vector<dtype *> out_grads, in_grads;
        out_grads.reserve(count);
        in_grads.reserve(count * inCount());
        for (Node *n : batch) {
            PAddNode &padd = dynamic_cast<PAddNode&>(*n);
            out_grads.push_back(padd.getLoss().value);
            for (Node *in : padd.ins_) {
                in_grads.push_back(in->getLoss().value);
            }
        }
        n3ldg_cuda::PAddBackward(out_grads, count, max_dim_, inCount(), in_grads, dim_arr_);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        for (Node *n : batch) {
            PAddNode *add = dynamic_cast<PAddNode*>(n);
            for (Node *in : add->ins_) {
                n3ldg_cuda::Assert(in->loss().verify("PAddExecutor backward"));
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
    n3ldg_cuda::IntArray dim_arr_;
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

PExecutor PAddNode::generate() {
    return new PAddExecutor();
}

#endif
