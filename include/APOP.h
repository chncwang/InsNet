/*
 * APOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef APOP_H_
#define APOP_H_

#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "APParam.h"

// for sparse features
struct APParams {
  public:
    APParam W;
    PAlphabet elems;
    int nVSize;
    int nDim;

  public:
    APParams() {
        nVSize = 0;
        nDim = 0;
        elems = NULL;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W);
    }

    inline void initialWeights(int nOSize) {
        if (nVSize == 0) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        nDim = nOSize;
        W.initial(nOSize, nVSize);
    }

    //random initialization
    inline void initial(PAlphabet alpha, int nOSize, int base = 1) {
        assert(base >= 1);
        elems = alpha;
        nVSize = base * elems->size();
        if (base > 1) {
            std::cout << "nVSize: " << nVSize << ", Alpha Size = " << elems->size()  << ", Require more Alpha."<< std::endl;
            elems->set_fixed_flag(false);
        }
        initialWeights(nOSize);
    }

    inline int getFeatureId(const string& strFeat) {
        int idx = elems->from_string(strFeat);
        if(!elems->m_b_fixed && elems->m_size >= nVSize) {
            std::cout << "AP Alphabet stopped collecting features" << std::endl;
            elems->set_fixed_flag(true);
        }
        return idx;
    }

};

//only implemented sparse linear node.
//non-linear transformations are not support,
class APNode : public Node {
  public:
    APParams* param;
    vector<int> ins;
    bool bTrain;

  public:
    APNode() : Node() {
        ins.clear();
        param = NULL;
        node_type = "apnode";
    }

    inline void setParam(APParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        ins.clear();
        bTrain = false;
    }

  public:
    //notice the output
    void forward(Graph *cg, const vector<string>& x) {
        int featId;
        int featSize = x.size();
        for (int idx = 0; idx < featSize; idx++) {
            featId = param->getFeatureId(x[idx]);
            if (featId >= 0) {
                ins.push_back(featId);
            }
        }
        degree = 0;
        cg->addNode(this);
        bTrain = cg->train;
    }
  public:
    inline void compute() {
        param->W.value(ins, val, bTrain);
    }

    //no output losses
    void backward() {
        //assert(param != NULL);
        param->W.loss(ins, loss);
    }

  public:
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        APNode* conv_other = (APNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};

class APExecute :public Execute {
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};


inline PExecute APNode::generate(bool bTrain, dtype cur_drop_factor) {
    APExecute* exec = new APExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
}

#endif /* APOP_H_ */
