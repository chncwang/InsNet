#ifndef CONCAT
#define CONCAT

/*
*  Concat.h:
*  concatenatation operation.
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/


#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#if USE_GPU
#include "N3LDG_cuda.h"
#endif
#include "profiler.h"

class ConcatNode : public Node, public Poolable<ConcatNode> {
public:
    vector<int> inDims;
    vector<PNode> ins;

    ConcatNode() : Node("concat") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void clear() override {
        inDims.clear();
        ins.clear();
        Node::clear();
    }

    void forward(Graph &cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cerr << "empty inputs for concat" << std::endl;
            abort();
        }

        for (int i = 0; i < x.size(); i++) {
            ins.push_back(x[i]);
        }

        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }
        int curDim = 0;
        for (int i = 0; i < nSize; ++i) {
            inDims.push_back(ins.at(i)->getDim());
            curDim += inDims[i];
        }
        if (curDim != getDim()) {
            std::cerr << "input dim size not match" << curDim << "\t" << getDim() << std::endl;
            abort();
        }
        cg.addNode(this);
    }

    PExecutor generate() override;

    string typeSignature() const override {
        string hash_code = Node::typeSignature() + "-" + to_string(inDims.size());
        for (int dim : inDims) {
            hash_code += "-" + to_string(dim);
        }
        return hash_code;
    }

    void compute() override {
        int nSize = ins.size();
        int offset = 0;
        for (int i = 0; i < nSize; ++i) {
            memcpy(val().v + offset, ins.at(i)->val().v,
                    inDims.at(i) * sizeof(dtype));
            offset += inDims[i];
        }
    }

    void backward() override {
        int nSize = ins.size();
        int offset = 0;
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < inDims[i]; idx++) {
                ins[i]->loss()[idx] += loss()[offset + idx];
            }
            offset += inDims[i];
        }
    }
};

#if USE_GPU
class ConcatExecutor : public Executor {
public:
    int outDim;
    int inCount;

    void forward() override {
        int count = batch.size();

        std::vector<dtype*> in_vals, vals;
        in_vals.reserve(inCount * count);
        vals.reserve(count);
        for (Node *node : batch) {
            ConcatNode *concat = static_cast<ConcatNode*>(node);
            for (Node *in : concat->ins) {
                in_vals.push_back(in->val().value);
            }
            vals.push_back(node->getVal().value);
        }

        n3ldg_cuda::ConcatForward(in_vals, static_cast<ConcatNode*>(batch.at(0))->inDims, vals,
                count, inCount, outDim);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch[idx]->val().verify("concat forward"));
        }
        cout << "concat forward tested" << endl;
#endif
    }

    void backward() override {
        int count = batch.size();
        std::vector<dtype*> in_losses, losses;
        in_losses.reserve(inCount * count);
        losses.reserve(count);
        for (Node *node : batch) {
            ConcatNode *concat = static_cast<ConcatNode*>(node);
            for (Node *in : concat->ins) {
                in_losses.push_back(in->loss().value);
            }
            losses.push_back(node->loss().value);
        }

        n3ldg_cuda::ConcatBackward(in_losses, static_cast<ConcatNode*>(batch.at(0))->inDims,
                losses, count, inCount, outDim);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }
        for (int idx = 0; idx < count; idx++) {
            for (int j = 0; j < inCount; ++j) {
                n3ldg_cuda::Assert(static_cast<ConcatNode *>(batch[idx])->
                        ins[j]->loss().verify("concat backward"));
            }
        }
#endif
    }
};
#else
class ConcatExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }

    int calculateActivations() override {
        return 0;
    }
};
#endif

PExecutor ConcatNode::generate() {
    ConcatExecutor* exec = new ConcatExecutor();
    exec->batch.push_back(this);
#if USE_GPU
    exec->inCount = this->ins.size();
    exec->outDim = 0;
    for (int d : inDims) {
        exec->outDim += d;
    }
#endif
    return exec;
}

class ScalarConcatNode : public Node, public Poolable<ScalarConcatNode> {
public:
    ScalarConcatNode() : Node("scalar_concat") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void forward(Graph &graph, const vector<Node *> &ins) {
        if (ins.size() != getDim()) {
            cerr << "ScalarConcatNode forward - ins size error" << endl;
            cerr << boost::format("ins size:%1% dim:%2%") % ins.size() % getDim() << endl;
            abort();
        }

        ins_ = ins;
        for (Node *n : ins) {
            if (n->getDim() != 1) {
                cerr << "ScalarConcatNode forward - non scalar found" << endl;
                abort();
            }
            n->addParent(this);
        }
        graph.addNode(this);
    }

    Executor *generate() override;

    void compute() override {
        int i = 0;
        for (Node *in : ins_) {
            val()[i++] = in->getVal()[0];
        }
    }

    void backward() override {
        int i = 0;
        for (Node *in : ins_) {
            in->loss()[0] += getLoss()[i++];
        }
    }

    const vector<Node *> ins() const {
        return ins_;
    }

    string typeSignature() const override {
        return getNodeType();
    }

private:
    vector<Node *> ins_;
};

#if USE_GPU
class ScalarConcatExecutor : public Executor {
public:
    void forward() override {
        vector<dtype *> in_vals, vals;
        for (Node *node : batch) {
            dims_.push_back(node->getDim());
        }
        max_dim_ = *max_element(dims_.begin(), dims_.end());
        for (Node *node : batch) {
            ScalarConcatNode *concat = static_cast<ScalarConcatNode *>(node);
            for (Node *in : concat->ins()) {
                in_vals.push_back(in->getVal().value);
            }
            for (int i = 0; i < max_dim_ - concat->ins().size(); ++i) {
                in_vals.push_back(nullptr);
            }
            vals.push_back(node->getVal().value);
        }
        n3ldg_cuda::ScalarConcatForward(in_vals, batch.size(), dims_, max_dim_, vals);
#if TEST_CUDA
        Executor::testForward();
#endif
    }

    void backward() override {
        vector<dtype *> losses, in_losses;
        for (Node *node : batch) {
            ScalarConcatNode *concat = static_cast<ScalarConcatNode *>(node);
            for (Node *in : concat->ins()) {
                in_losses.push_back(in->getLoss().value);
            }
            for (int i = 0; i < max_dim_ - concat->ins().size(); ++i) {
                in_losses.push_back(nullptr);
            }
            losses.push_back(node->getLoss().value);
        }
#if TEST_CUDA
        auto get_inputs = [](Node &node) {
            ScalarConcatNode &concat = static_cast<ScalarConcatNode&>(node);
            vector<pair<Node *, string>> results;
            for (Node *n : concat.ins()) {
                results.push_back(make_pair(n, "input"));
            }
            return results;
        };
//        cout << "test before scalar concat" << endl;
        Executor::testBeforeBackward(get_inputs);
#endif
//        cout << "gpu loss:";
//        batch.front()->getLoss().print();
//        for (int dim : dims_) {
//            cout << "dim:" << dim;
//        }
        n3ldg_cuda::ScalarConcatBackward(losses, batch.size(), dims_, max_dim_, in_losses);
#if TEST_CUDA
//        cout << "batch:" << this->batch.size() << endl;
//        cout << "loss:" << batch.front()->getLoss().toString() << endl;
        Executor::testBackward(get_inputs);
//        cout << "scalar concat tested" << endl;
#endif
    }

private:
    vector<int> dims_;
    int max_dim_;
};
#else
class ScalarConcatExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }
};
#endif

Executor *ScalarConcatNode::generate() {
    return new ScalarConcatExecutor;
}

namespace n3ldg_plus {

Node *concat(Graph &graph, vector<Node*> inputs) {
    int dim = 0;
    for (Node *in : inputs) {
        dim += in->getDim();
    }
    ConcatNode *concat = ConcatNode::newNode(dim);
    concat->forward(graph, inputs);
    return concat;
}

Node *scalarConcat(Graph &graph, vector<Node *> inputs) {
    ScalarConcatNode *concat = ScalarConcatNode::newNode(inputs.size());
    concat->forward(graph, inputs);
    return concat;
}

}

#endif
