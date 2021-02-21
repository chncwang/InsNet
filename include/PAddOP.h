#ifndef PAddOP
#define PAddOP

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class PAddNode : public Node, public Poolable<PAddNode> {
public:
    vector<Node *> ins;

    PAddNode() : Node("point-add") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void setInputs(const vector<Node *> &ins) override {
        this->ins = ins;
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
        int nSize = ins.size();
        val().zero();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < getDim(); idx++) {
                val()[idx] += ins.at(i)->val()[idx];
            }
        }
    }


    void backward() override {
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < getDim(); idx++) {
                ins.at(i)->loss()[idx] += loss()[idx];
            }
        }
    }


public:
    PExecutor generate() override;

    string typeSignature() const override {
        return Node::typeSignature() + "-" + to_string(ins.size());
    }
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

class PAddExecutor : public Executor {
public:
    int in_count;
    int dim;
    Tensor1D x, y;
    int sumDim;

#if !USE_GPU
    int calculateFLOPs() override {
        int sum = 0;
        for (Node *node : batch) {
            PAddNode *add = static_cast<PAddNode*>(node);
            sum += add->getDim() * add->ins.size();
        }
        return sum;
    }
#endif

#if USE_GPU
    void  forward() {
        int count = batch.size();

        std::vector<std::vector<dtype*>> in_vals;
        in_vals.reserve(in_count);
        for (int i = 0; i < in_count; ++i) {
            std::vector<dtype*> ins;
            ins.reserve(count);
            for (Node * n : batch) {
                PAddNode *padd = static_cast<PAddNode*>(n);
                ins.push_back(padd->ins.at(i)->val().value);
#if TEST_CUDA
                cout << "input type:" << padd->ins.at(i)->typeSignature() << endl;
                n3ldg_cuda::Assert(padd->ins.at(i)->val().verify("PAdd forward input"));
#endif
            }
            in_vals.push_back(ins);
        }
        std::vector<dtype *> outs;
        outs.reserve(count);
        for (Node * n : batch) {
            PAddNode *padd = static_cast<PAddNode*>(n);
            outs.push_back(padd->val().value);
        }
        n3ldg_cuda::PAddForward(in_vals, count, dim, in_count, outs);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
        }
        for (Node *n : batch) {
            n3ldg_cuda::Assert(n->val().verify("PAdd forward"));
        }
#endif
    }
#else
    void  forward() override {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
        }
    }
#endif

#if USE_GPU
    void backward() {
        int count = batch.size();
        std::vector<std::vector<dtype*>> in_losses;
        in_losses.reserve(in_count);
        for (int i = 0; i < in_count; ++i) {
            std::vector<dtype*> ins;
            ins.reserve(count);
            for (Node * n : batch) {
                PAddNode *padd = static_cast<PAddNode*>(n);
                ins.push_back(padd->ins.at(i)->loss().value);
            }
            in_losses.push_back(ins);
        }
        std::vector<dtype *> out_losses;
        out_losses.reserve(count);
        for (Node * n : batch) {
            PAddNode *padd = static_cast<PAddNode*>(n);
            out_losses.push_back(padd->loss().value);
        }
        n3ldg_cuda::PAddBackward(out_losses, count, dim, in_count, in_losses);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        for (Node *n : batch) {
            PAddNode *add = static_cast<PAddNode*>(n);
            for (Node *in : add->ins) {
                n3ldg_cuda::Assert(in->loss().verify("PAddExecutor backward"));
            }
        }
        cout << "PAddExecutor backward tested" << endl;
#endif
    }
#else
    void backward() override {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }
    }
#endif
};

PExecutor PAddNode::generate() {
    PAddExecutor* exec = new PAddExecutor();
    exec->in_count = ins.size();
    exec->dim = getDim();
    return exec;
}

#endif
