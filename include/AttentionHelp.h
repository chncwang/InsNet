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


class AttentionSoftMaxNode : public Node {
  public:
    vector<dtype> masks, mask_losses;
    vector<dtype> unnormed_masks;
    dtype sum;
    vector<PNode> unnormeds;
    vector<PNode> ins;

  public:
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

    inline void clearValue() {
        Node::clearValue();
        ins.clear();
        unnormeds.clear();
        sum = 0;
    }

    inline void setParam(int maxsize) {
        masks.resize(maxsize);
        mask_losses.resize(maxsize);
        unnormed_masks.resize(maxsize);
    }


    inline void init(int ndim, dtype dropout) {
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
                clearValue();
                return;
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
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

  public:

    inline void compute() {
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


class AttentionSoftMaxExecute : public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            AttentionSoftMaxNode* ptr = (AttentionSoftMaxNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            AttentionSoftMaxNode* ptr = (AttentionSoftMaxNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute AttentionSoftMaxNode::generate(bool bTrain) {
    AttentionSoftMaxExecute* exec = new AttentionSoftMaxExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
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

    inline void clearValue() {
        Node::clearValue();
        ins.clear();
        unnormeds.clear();
        sum.zero();
    }

    inline void setParam(int maxsize) {
        masks.resize(maxsize);
        mask_losses.resize(maxsize);
        unnormed_masks.resize(maxsize);
    }


    inline void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
        int count = masks.size();
        for (int idx = 0; idx < count; idx++) {
            masks[idx].init(ndim);
            mask_losses[idx].init(ndim);
            unnormed_masks[idx].init(ndim);
        }
        sum.init(ndim);
        sum.zero();
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
                clearValue();
                return;
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
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

  public:

    inline void compute() {
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


class AttentionSoftMaxVExecute : public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            AttentionSoftMaxVNode* ptr = (AttentionSoftMaxVNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            AttentionSoftMaxVNode* ptr = (AttentionSoftMaxVNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute AttentionSoftMaxVNode::generate(bool bTrain) {
    AttentionSoftMaxVExecute* exec = new AttentionSoftMaxVExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}
//#endif

#endif
