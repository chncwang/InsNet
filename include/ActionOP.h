/*
 * SparseOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef ACTIONOP_H_
#define ACTIONOP_H_

#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "SparseParam.h"

// for sparse features
class ActionParams {
  public:
    SparseParam W;
    PAlphabet elems;
    int nVSize;
    int nDim;

  public:
    ActionParams() {
        nVSize = 0;
        nDim = 0;
        elems = NULL;
    }

    void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W);
    }

    void initialWeights(int nOSize) {
        if (nVSize == 0) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        nDim = nOSize;
        W.initial(nOSize, nVSize);
        W.val.random(0.01);
    }


    //random initialization
    void initial(PAlphabet alpha, int nOSize) {
        elems = alpha;
        nVSize =elems->size();
        initialWeights(nOSize);
    }

    int getFeatureId(const string& strFeat) {
        int idx = elems->from_string(strFeat);
        return idx;
    }

};

//only implemented sparse linear node.
//non-linear transformations are not support,
class ActionNode : public Node {
  public:
    ActionParams* param;
    int actid;
    PNode in;

    ActionNode() : Node() {
        actid = -1;
        param = NULL;
        node_type = "actionnode";
    }

    void setParam(ActionParams* paramInit) {
        param = paramInit;
    }

    //can not be dropped since the output is a scalar
    void init(int ndim, dtype dropout) {
        dim = 1;
        Node::init(dim, -1);
    }

    //notice the output
    void forward(Graph *cg, const string& ac, PNode x) {
        actid = param->getFeatureId(ac);
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

    void compute() {
        if (param->nDim != in->dim) {
            std::cout << "warning: action dim not equal ." << std::endl;
        }
        val[0] = 0;
        if (actid >= 0) {
            for (int idx = 0; idx < in->dim; idx++) {
                val[0] += in->val[idx] * param->W.val[actid][idx];
            }
        } else {
            std::cout << "unknown action" << std::endl;
        }
    }

    //no output losses
    void backward() {
        if (actid >= 0) {
            for (int idx = 0; idx < in->dim; idx++) {
                in->loss[idx] += loss[0] * param->W.val[actid][idx];
                param->W.grad[actid][idx] += loss[0] * in->val[idx];
            }
            param->W.indexers[actid] = true;
        }
    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        ActionNode* conv_other = (ActionNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

    size_t typeHashCode() const override {
        return Node::typeHashCode() ^ ::typeHashCode(param);
    }
};


class ActionExecute :public Execute {
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


PExecute ActionNode::generate(bool bTrain, dtype cur_drop_factor) {
    ActionExecute* exec = new ActionExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
}

#endif /* ACTIONOP_H_ */
