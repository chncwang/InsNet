#ifndef ATTENTION_HELP
#define ATTENTION_HELP

/*
*  AttentionHelp.h:
*  attention softmax help nodes
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include <memory>

class AttentionSoftMaxNode : public Node {
public:
    vector<dtype> masks, mask_losses;
    vector<dtype> unnormed_masks;
    dtype sum;
    vector<Node *> unnormeds;
    vector<Node *> ins;

    AttentionSoftMaxNode() : Node("AttentionSoftmax") {}

    void init(int ndim) {
        Node::init(ndim);
    }

    void forward(Graph &cg, vector<Node *>& x, vector<Node *>& a) {
        if (x.empty()) {
            std::cerr << "empty inputs for attention help node" << std::endl;
            abort();
        }
        if (x.size() != a.size()) {
            std::cerr <<
                "the number of input nodes does not equal the number of attention factors." <<
                std::endl;
            abort();
        }
        int nSize = x.size();
        for (int i = 0; i < nSize; i++) {
            if (x.at(i)->val().dim != getDim() || a.at(i)->val().dim != 1) {
                std::cerr << "input matrixes are not matched" << std::endl;
                abort();
            }
            ins.push_back(x.at(i));
            unnormeds.push_back(a.at(i));
        }

        for (int i = 0; i < nSize; i++) {
            ins.at(i)->addParent(this);
            unnormeds.at(i)->addParent(this);
        }

        cg.addNode(this);
    }

    PExecutor generate();

    bool typeEqual(Node * other) {
        return Node::typeEqual(other);
    }

    void compute() {
        int nSize = ins.size();
        unnormed_masks.resize(nSize);
        masks.resize(nSize);

        sum = 0;
        for (int i = 0; i < nSize; ++i) {
            unnormed_masks.at(i) = fexp(unnormeds.at(i)->val()[0]);
            sum += unnormed_masks.at(i);
        }

        for (int i = 0; i < nSize; ++i) {
            masks.at(i) = unnormed_masks.at(i) / sum;
        }

        val().zero();
        for (int i = 0; i < nSize; ++i) {
            val().vec() += masks.at(i) * ins.at(i)->val().vec();
        }
    }

    void backward() {
        int nSize = ins.size();
        mask_losses.resize(nSize);
        for (int i = 0; i < nSize; i++) {
            ins.at(i)->loss().vec() += loss().vec() * masks.at(i);
            mask_losses.at(i) = 0;
            for (int idx = 0; idx < getDim(); idx++) {
                mask_losses.at(i) += loss()[idx] * ins.at(i)->val()[idx];
            }
        }

        for (int i = 0; i < nSize; i++) {
            for (int j = 0; j < nSize; j++) {
                unnormeds.at(i)->loss()[0] -= masks.at(i) * masks.at(j) * mask_losses.at(j);
                if (i == j) {
                    unnormeds.at(i)->loss()[0] += masks.at(i) * mask_losses.at(i);
                }
            }
        }
    }
};


#if USE_GPU
class AttentionSoftMaxExecutor : public Executor {
public:
    int dim;
    std::vector<int> in_counts;
    int max_in_count;
    std::vector<std::shared_ptr<Tensor2D>> masks;
    std::vector<dtype*> raw_masks;
    std::vector<dtype*> ins;

    void forward() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent("attention forward");
        int count = batch.size();
        in_counts.reserve(count);
        masks.reserve(count);
        for (Node *n : batch) {
            AttentionSoftMaxNode *attention =
                static_cast<AttentionSoftMaxNode*>(n);
            in_counts.push_back(attention->ins.size());
        }
        max_in_count = *std::max_element(in_counts.begin(), in_counts.end());
        for (Node *n : batch) {
            AttentionSoftMaxNode *attention =
                static_cast<AttentionSoftMaxNode*>(n);
            in_counts.push_back(attention->ins.size());
            auto p = std::make_shared<Tensor2D>();
            p->init(dim, max_in_count);
            masks.push_back(p);
        }
        std::vector<dtype*> unnormeds, vals;
        ins.reserve(count * max_in_count);
        unnormeds.reserve(count * max_in_count);
        vals.reserve(count);
        for (Node *n : batch) {
            AttentionSoftMaxNode *att = static_cast<AttentionSoftMaxNode*>(n);
            vals.push_back(att->val().value);
            for (int i = 0; i < att->ins.size(); ++i) {
                ins.push_back(att->ins.at(i)->val().value);
                unnormeds.push_back(att->unnormeds.at(i)->val().value);
            }
            for (int i = 0; i < max_in_count - att->ins.size(); ++i) {
                ins.push_back(NULL);
                unnormeds.push_back(NULL);
            }
        }

        raw_masks.reserve(count);
        for (auto &p : masks) {
            raw_masks.push_back(p->value);
        }
        n3ldg_cuda::ScalarAttentionForward(ins, unnormeds, in_counts, count,
                dim, raw_masks, vals);
#if TEST_CUDA
        for (Node *n : batch) {
            n->compute();
            n3ldg_cuda::Assert(n->val().verify("AttentionSoftMaxExecutor forward"));
        }
#endif
        profiler.EndCudaEvent();
    }

    void backward() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent("attention backward");
        int count = batch.size();
        std::vector<dtype*> losses, in_losses, unnormed_losses;
        losses.reserve(count);
        in_losses.reserve(count * max_in_count);
        unnormed_losses.reserve(count * max_in_count);

        for (Node *n : batch) {
            losses.push_back(n->loss().value);
            AttentionSoftMaxNode *att = static_cast<AttentionSoftMaxNode*>(n);
            for (int i = 0; i < att->ins.size(); ++i) {
                in_losses.push_back(att->ins.at(i)->loss().value);
                unnormed_losses.push_back(att->unnormeds.at(i)->loss().value);
            }
            for (int i = 0; i < max_in_count - att->ins.size(); ++i) {
                in_losses.push_back(NULL);
                unnormed_losses.push_back(NULL);
            }
        }

        n3ldg_cuda::ScalarAttentionBackward(losses, ins, raw_masks, in_counts,
                count, dim, in_losses, unnormed_losses);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        for (Node *n : batch) {
            AttentionSoftMaxNode *att = static_cast<AttentionSoftMaxNode*>(n);
            for (Node *in : att->ins) {
                n3ldg_cuda::Assert(in->loss().verify("AttentionSoftMaxExecutor backward ins"));
            }

            for (Node *un : att->unnormeds) {
                n3ldg_cuda::Assert(un->loss().verify(
                            "AttentionSoftMaxExecutor backward unnormeds"));
            }
        }
#endif
        profiler.EndCudaEvent();
    }
};
#else
class AttentionSoftMaxExecutor : public Executor {};
#endif

PExecutor AttentionSoftMaxNode::generate() {
    AttentionSoftMaxExecutor* exec = new AttentionSoftMaxExecutor();
    exec->batch.push_back(this);
#if USE_GPU
    exec->dim = getDim();
#endif
    return exec;
}


class AttentionSoftMaxVNode : public Node {
public:
    vector<Tensor1D> masks, mask_losses;
    vector<Tensor1D> unnormed_masks;
    Tensor1D sum;
    vector<Node *> unnormeds;
    vector<Node *> ins;

    AttentionSoftMaxVNode() : Node("AttentionSoftmaxV") {}

    void init(int ndim) {
        Node::init(ndim);
//        int count = masks.size();
//        for (int idx = 0; idx < count; idx++) {
//            masks[idx].init(ndim);
//            mask_losses[idx].init(ndim);
//            unnormed_masks[idx].init(ndim);
//        }
        sum.init(ndim);
#if !USE_GPU
        sum.zero();
#endif
    }

    void forward(Graph &cg, const vector<Node *>& x, const vector<Node *>& a) {
        if (x.size() == 0) {
            std::cerr << "empty inputs for attention help node" << std::endl;
            abort();
        }
        if (x.size() != a.size()) {
            std::cerr << "the number of input nodes does not equal the number of attention factors." << std::endl;
            abort();
        }

        int nSize = x.size();
        for (int i = 0; i < nSize; i++) {
            if (x.at(i)->val().dim != getDim() || a.at(i)->val().dim != getDim()) {
                std::cerr << "input matrixes are not matched" << std::endl;
                abort();
            }
            ins.push_back(x.at(i));
            unnormeds.push_back(a.at(i));
        }

        for (int i = 0; i < nSize; i++) {
            ins.at(i)->addParent(this);
            unnormeds.at(i)->addParent(this);
        }

        cg.addNode(this);
    }

    PExecutor generate();

    bool typeEqual(Node * other) {
        return Node::typeEqual(other);
    }

    void compute() {
        int nSize = ins.size();
        unnormed_masks.resize(nSize);
        masks.resize(nSize);
        for (int i = 0; i < nSize; ++i) {
            unnormed_masks.at(i).init(getDim());
            masks.at(i).init(getDim());
        }
        sum.zero();
        for (int i = 0; i < nSize; ++i) {
            unnormed_masks.at(i).vec() = unnormeds.at(i)->val().vec().unaryExpr(ptr_fun(fexp));
            sum.vec() += unnormed_masks.at(i).vec();
        }

        for (int i = 0; i < nSize; ++i) {
            masks.at(i).vec() = unnormed_masks.at(i).vec() / sum.vec();
        }

        val().zero();
        for (int i = 0; i < nSize; ++i) {
            val().vec() += masks.at(i).vec() * ins.at(i)->val().vec();
        }
    }

    void backward() {
        int nSize = ins.size();
        mask_losses.resize(nSize);
        for (int i = 0; i < nSize; ++i) {
            mask_losses.at(i).init(getDim());
        }

        for (int i = 0; i < nSize; i++) {
            ins.at(i)->loss().vec() += loss().vec() * masks.at(i).vec();
            mask_losses.at(i).vec() = loss().vec() * ins.at(i)->val().vec();
        }

        for (int idx = 0; idx < getDim(); idx++) {
            for (int i = 0; i < nSize; i++) {
                for (int j = 0; j < nSize; j++) {
                    unnormeds.at(i)->loss()[idx] -= masks.at(i)[idx] * masks.at(j)[idx] * mask_losses[j][idx];
                    if (i == j) {
                        unnormeds.at(i)->loss()[idx] += masks[i][idx] * mask_losses[i][idx];
                    }
                }
            }
        }
    }
};

#if USE_GPU
class AttentionSoftMaxVExecutor : public Executor {
public:
    int dim;
    std::vector<int> in_counts;
    int max_in_count;
    std::vector<std::shared_ptr<Tensor2D>> masks;
    std::vector<dtype*> raw_masks;
    std::vector<dtype*> ins;

    void forward() {
        int count = batch.size();
        in_counts.reserve(count);
        for (Node *n : batch) {
            AttentionSoftMaxVNode *attention =
                static_cast<AttentionSoftMaxVNode*>(n);
            in_counts.push_back(attention->ins.size());
        }
        max_in_count = *std::max_element(in_counts.begin(), in_counts.end());
        masks.reserve(count);
        for (Node *n : batch) {
            AttentionSoftMaxVNode *attention =
                static_cast<AttentionSoftMaxVNode*>(n);
            in_counts.push_back(attention->ins.size());
            auto p = std::make_shared<Tensor2D>();
            p->init(dim, max_in_count);
            masks.push_back(p);
        }
        std::vector<dtype*> unnormeds, vals;
        ins.reserve(count * max_in_count);
        unnormeds.reserve(count * max_in_count);
        vals.reserve(count);
        for (Node *n : batch) {
            AttentionSoftMaxVNode *att =
                static_cast<AttentionSoftMaxVNode*>(n);
            vals.push_back(att->val().value);
            for (int i = 0; i < att->ins.size(); ++i) {
#if TEST_CUDA
                n3ldg_cuda::Assert(att->ins.at(i)->val().verify(
                            "AttentionSoftMaxVExecutor forward initial val"));
#endif
                ins.push_back(att->ins.at(i)->val().value);
#if TEST_CUDA
                n3ldg_cuda::Assert(att->unnormeds.at(i)->val().verify(
                            "AttentionSoftMaxVExecutor forward  initial unnormeds"));
#endif
                unnormeds.push_back(att->unnormeds.at(i)->val().value);
            }
            for (int i = 0; i < max_in_count - att->ins.size(); ++i) {
                ins.push_back(NULL);
                unnormeds.push_back(NULL);
            }
        }

        raw_masks.reserve(count);
        for (auto &p : masks) {
            raw_masks.push_back(p->value);
        }
        n3ldg_cuda::VectorAttentionForward(ins, unnormeds, in_counts, count,
                dim, raw_masks, vals);
#if TEST_CUDA
        for (Node *n : batch) {
            n->compute();
            n3ldg_cuda::Assert(n->val().verify("AttentionSoftMaxVExecutor forward"));
        }
#endif
    }

    void backward() {
        int count = batch.size();
        std::vector<dtype*> losses, in_losses, unnormed_losses;
        losses.reserve(count);
        in_losses.reserve(count * max_in_count);
        unnormed_losses.reserve(count * max_in_count);

        for (Node *n : batch) {
            losses.push_back(n->loss().value);
            AttentionSoftMaxVNode *att = static_cast<AttentionSoftMaxVNode*>(n);
            for (int i = 0; i < att->ins.size(); ++i) {
                in_losses.push_back(att->ins.at(i)->loss().value);
                unnormed_losses.push_back(att->unnormeds.at(i)->loss().value);
            }
            for (int i = 0; i < max_in_count - att->ins.size(); ++i) {
                in_losses.push_back(NULL);
                unnormed_losses.push_back(NULL);
            }
        }

        n3ldg_cuda::VectorAttentionBackward(losses, ins, raw_masks, in_counts,
                count, dim, in_losses, unnormed_losses);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        for (Node *n : batch) {
            AttentionSoftMaxVNode *att = static_cast<AttentionSoftMaxVNode*>(n);
            for (Node *in : att->ins) {
                n3ldg_cuda::Assert(in->loss().verify("AttentionSoftMaxExecutor backward ins"));
            }

            for (Node *un : att->unnormeds) {
                n3ldg_cuda::Assert(un->loss().verify(
                            "AttentionSoftMaxExecutor backward unnormeds"));
            }
        }
#endif
    }
};
#else
class AttentionSoftMaxVExecutor : public Executor {
  public:
    void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
        }
    }

    void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }
    }
};
#endif

PExecutor AttentionSoftMaxVNode::generate() {
    AttentionSoftMaxVExecutor* exec = new AttentionSoftMaxVExecutor();
    exec->batch.push_back(this);
#if USE_GPU
    exec->dim = getDim();
#endif
    return exec;
}

#endif
