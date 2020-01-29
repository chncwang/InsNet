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

class ConcatNode : public Node {
public:
    vector<int> inDims;
    vector<PNode> ins;

    ConcatNode() : Node("concat") {}

    void forward(Graph &cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cerr << "empty inputs for concat" << std::endl;
            abort();
        }

        ins.clear();
        for (int i = 0; i < x.size(); i++) {
            ins.push_back(x[i]);
        }

        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }
        inDims.clear();
        int curDim = 0;
        for (int i = 0; i < nSize; ++i) {
            inDims.push_back(ins[i]->val().dim);
            curDim += inDims[i];
        }
        if (curDim != getDim()) {
            std::cerr << "input dim size not match" << curDim << "\t" << getDim() << std::endl;
            abort();
        }
        cg.addNode(this);
    }

    PExecutor generate() override;

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) override {
        if (!Node::typeEqual(other)) {
            return false;
        }
        ConcatNode *o = static_cast<ConcatNode*>(other);
        if (inDims.size() != o->inDims.size()) {
            return false;
        }
        for (int i = 0; i < inDims.size(); ++i) {
            if (inDims.at(i) != o->inDims.at(i)) {
                return false;
            }
        }
        return true;
    }

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

    void  forward() {
        int count = batch.size();

        std::vector<dtype*> in_vals, vals;
        in_vals.reserve(inCount * count);
        vals.reserve(count);
        for (Node *node : batch) {
            ConcatNode *concat = static_cast<ConcatNode*>(node);
            for (Node *in : concat->ins) {
                in_vals.push_back(in->val().value);
            }
            vals.push_back(node->val().value);
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

    void backward() {
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

class ScalarConcatNode : public Node {
public:
    ScalarConcatNode() : Node("scalar_concat") {}

    void forward(Graph &graph, const vector<Node *> &ins) {
        if (ins.size() != getDim()) {
            cerr << "ScalarConcatNode forward - ins size error" << endl;
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

    bool typeEqual(Node *other) override {
        return getNodeType() == other->getNodeType();
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
                in_losses.push_back(in->getVal().value);
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
        cout << "test before scalar concat" << endl;
        Executor::testBeforeBackward(get_inputs);
#endif
        n3ldg_cuda::ScalarConcatBackward(losses, batch.size(), dims_, max_dim_, in_losses);
#if TEST_CUDA
        Executor::testBackward(get_inputs);
#endif
    }

private:
    vector<int> dims_;
    int max_dim_;
};
#else
class ScalarConcatExecutor : public Executor {};
#endif

Executor *ScalarConcatNode::generate() {
    return new ScalarConcatExecutor;
}

namespace n3ldg_plus {

Node *concat(Graph &graph, const vector<Node*> inputs) {
    int dim = 0;
    for (Node *in : inputs) {
        dim += in->getDim();
    }
    ConcatNode *concat = new ConcatNode;
    concat->init(dim);
    concat->forward(graph, inputs);
    return concat;
}

Node *scalarConcat(Graph &graph, const vector<Node *> inputs) {
    ScalarConcatNode *concat = new ScalarConcatNode;
    concat->init(inputs.size());
    concat->forward(graph, inputs);
    return concat;
}

}

#endif
