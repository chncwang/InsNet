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
    vector<PNode> unnormeds;
    vector<PNode> ins;

    AttentionSoftMaxNode() : Node() {
        ins.clear();
        unnormeds.clear();
        node_type = "AttentionSoftmax";
    }

    ~AttentionSoftMaxNode() {
        masks.clear();
        mask_losses.clear();
        unnormed_masks.clear();
        ins.clear();
        unnormeds.clear();
    }

    void setParam(int maxsize) {
        masks.resize(maxsize);
        mask_losses.resize(maxsize);
        unnormed_masks.resize(maxsize);
    }


    void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
    }

  public:
    void forward(Graph *cg, const vector<PNode>& x, const vector<PNode>& a) {
        if (x.size() == 0) {
            std::cout << "empty inputs for attention help node" << std::endl;
            return;
        }
        if (x.size() != a.size()) {
            std::cout << "the number of input nodes does not equal the number of attention factors." << std::endl;
            return;
        }
        int nSize = x.size();
        ins.clear();
        unnormeds.clear();
        for (int i = 0; i < nSize; i++) {
            if (x[i]->val.dim != dim || a[i]->val.dim != 1) {
                std::cout << "input matrixes are not matched" << std::endl;
                abort();
            }
            ins.push_back(x[i]);
            unnormeds.push_back(a[i]);
        }

        degree = 0;
        for (int i = 0; i < nSize; i++) {
            ins[i]->addParent(this);
            unnormeds[i]->addParent(this);
        }

        cg->addNode(this);
    }


  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

  public:

    void compute() {
        int nSize = ins.size();

        sum = 0;
        for (int i = 0; i < nSize; ++i) {
            unnormed_masks[i] = fexp(unnormeds[i]->val[0]);
            sum += unnormed_masks[i];
        }

        for (int i = 0; i < nSize; ++i) {
            masks[i] = unnormed_masks[i] / sum;
        }

        val.zero();
        for (int i = 0; i < nSize; ++i) {
            val.vec() += masks[i] * ins[i]->val.vec();
        }
    }

    void backward() {
        int nSize = ins.size();
        for (int i = 0; i < nSize; i++) {
            ins[i]->loss.vec() += loss.vec() * masks[i];
            mask_losses[i] = 0;
            for (int idx = 0; idx < dim; idx++) {
                mask_losses[i] += loss[idx] * ins[i]->val[idx];
            }
        }

        for (int i = 0; i < nSize; i++) {
            for (int j = 0; j < nSize; j++) {
                unnormeds[i]->loss[0] -= masks[i] * masks[j] * mask_losses[j];
                if (i == j) {
                    unnormeds[i]->loss[0] += masks[i] * mask_losses[i];
                }
            }
        }


    }

};


#if USE_GPU
class AttentionSoftMaxExecute : public Execute {
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
            vals.push_back(att->val.value);
            for (int i = 0; i < att->ins.size(); ++i) {
                ins.push_back(att->ins.at(i)->val.value);
                unnormeds.push_back(att->unnormeds.at(i)->val.value);
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
            AttentionSoftMaxNode *att = static_cast<AttentionSoftMaxNode*>(n);
            n3ldg_cuda::Assert(n->val.verify(
                        "AttentionSoftMaxExecute forward"));
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
            losses.push_back(n->loss.value);
            AttentionSoftMaxNode *att = static_cast<AttentionSoftMaxNode*>(n);
            for (int i = 0; i < att->ins.size(); ++i) {
                in_losses.push_back(att->ins.at(i)->loss.value);
                unnormed_losses.push_back(att->unnormeds.at(i)->loss.value);
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
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }

        for (Node *n : batch) {
            AttentionSoftMaxNode *att = static_cast<AttentionSoftMaxNode*>(n);
            for (Node *in : att->ins) {
                n3ldg_cuda::Assert(in->loss.verify(
                            "AttentionSoftMaxExecute backward ins"));
            }

            for (Node *un : att->unnormeds) {
                n3ldg_cuda::Assert(un->loss.verify(
                            "AttentionSoftMaxExecute backward unnormeds"));
            }
        }
#endif
    }
};
#else
class AttentionSoftMaxExecute : public Execute {
  public:
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};
#endif

PExecute AttentionSoftMaxNode::generate(bool bTrain, dtype cur_drop_factor) {
    AttentionSoftMaxExecute* exec = new AttentionSoftMaxExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
#if USE_GPU
    exec->dim = dim;
#endif
    return exec;
}


class AttentionSoftMaxVNode : public Node {
  public:
    vector<Tensor1D> masks, mask_losses;
    vector<Tensor1D> unnormed_masks;
    Tensor1D sum;
    vector<PNode> unnormeds;
    vector<PNode> ins;

  public:
    AttentionSoftMaxVNode() : Node() {
        ins.clear();
        unnormeds.clear();
        node_type = "AttentionSoftmaxV";
    }

    ~AttentionSoftMaxVNode() {
        masks.clear();
        mask_losses.clear();
        unnormed_masks.clear();
        ins.clear();
        unnormeds.clear();
    }

    void setParam(int maxsize) {
        masks.resize(maxsize);
        mask_losses.resize(maxsize);
        unnormed_masks.resize(maxsize);
    }


    void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
        int count = masks.size();
        for (int idx = 0; idx < count; idx++) {
            masks[idx].init(ndim);
            mask_losses[idx].init(ndim);
            unnormed_masks[idx].init(ndim);
        }
        sum.init(ndim);
#if !USE_GPU
        sum.zero();
#endif
    }

  public:
    void forward(Graph *cg, const vector<PNode>& x, const vector<PNode>& a) {
        if (x.size() == 0) {
            std::cout << "empty inputs for attention help node" << std::endl;
            return;
        }
        if (x.size() != a.size()) {
            std::cout << "the number of input nodes does not equal the number of attention factors." << std::endl;
            return;
        }
        int nSize = x.size();
        ins.clear();
        unnormeds.clear();
        for (int i = 0; i < nSize; i++) {
            if (x[i]->val.dim != dim || a[i]->val.dim != dim) {
                std::cout << "input matrixes are not matched" << std::endl;
                abort();
            }
            ins.push_back(x[i]);
            unnormeds.push_back(a[i]);
        }

        degree = 0;
        for (int i = 0; i < nSize; i++) {
            ins[i]->addParent(this);
            unnormeds[i]->addParent(this);
        }

        cg->addNode(this);
    }


  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

  public:

    void compute() {
        int nSize = ins.size();

        sum.zero();
        for (int i = 0; i < nSize; ++i) {
            unnormed_masks[i].vec() = unnormeds[i]->val.vec().unaryExpr(ptr_fun(fexp));
            sum.vec() += unnormed_masks[i].vec();
        }

        for (int i = 0; i < nSize; ++i) {
            masks[i].vec() = unnormed_masks[i].vec() / sum.vec();
        }

        val.zero();
        for (int i = 0; i < nSize; ++i) {
            val.vec() += masks[i].vec() * ins[i]->val.vec();
        }
    }

    void backward() {
        int nSize = ins.size();
        for (int i = 0; i < nSize; i++) {
            ins[i]->loss.vec() += loss.vec() * masks[i].vec();
            mask_losses[i].vec() = loss.vec() * ins[i]->val.vec();
        }

        for (int idx = 0; idx < dim; idx++) {
            for (int i = 0; i < nSize; i++) {
                for (int j = 0; j < nSize; j++) {
                    unnormeds[i]->loss[idx] -= masks[i][idx] * masks[j][idx] * mask_losses[j][idx];
                    if (i == j) {
                        unnormeds[i]->loss[idx] += masks[i][idx] * mask_losses[i][idx];
                    }
                }
            }
        }


    }

};

#if USE_GPU
class AttentionSoftMaxVExecute : public Execute {
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
            vals.push_back(att->val.value);
            for (int i = 0; i < att->ins.size(); ++i) {
#if TEST_CUDA
                n3ldg_cuda::Assert(att->ins.at(i)->val.verify("AttentionSoftMaxVExecute forward initial val"));
#endif
                ins.push_back(att->ins.at(i)->val.value);
#if TEST_CUDA
                n3ldg_cuda::Assert(att->unnormeds.at(i)->val.verify("AttentionSoftMaxVExecute forward  initial unnormeds"));
#endif
                unnormeds.push_back(att->unnormeds.at(i)->val.value);
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
            AttentionSoftMaxVNode *att =
                static_cast<AttentionSoftMaxVNode*>(n);
            n3ldg_cuda::Assert(n->val.verify(
                        "AttentionSoftMaxVExecute forward"));
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
            losses.push_back(n->loss.value);
            AttentionSoftMaxVNode *att = static_cast<AttentionSoftMaxVNode*>(n);
            for (int i = 0; i < att->ins.size(); ++i) {
                in_losses.push_back(att->ins.at(i)->loss.value);
                unnormed_losses.push_back(att->unnormeds.at(i)->loss.value);
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
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }

        for (Node *n : batch) {
            AttentionSoftMaxVNode *att = static_cast<AttentionSoftMaxVNode*>(n);
            for (Node *in : att->ins) {
                n3ldg_cuda::Assert(in->loss.verify(
                            "AttentionSoftMaxExecute backward ins"));
            }

            for (Node *un : att->unnormeds) {
                n3ldg_cuda::Assert(un->loss.verify(
                            "AttentionSoftMaxExecute backward unnormeds"));
            }
        }
#endif
    }
};
#else
class AttentionSoftMaxVExecute : public Execute {
  public:
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};
#endif

PExecute AttentionSoftMaxVNode::generate(bool bTrain, dtype cur_drop_factor) {
    AttentionSoftMaxVExecute* exec = new AttentionSoftMaxVExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
#if USE_GPU
    exec->dim = dim;
#endif
    return exec;
}

#endif
