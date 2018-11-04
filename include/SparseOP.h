/*
 * SparseOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef SPARSEOP_H_
#define SPARSEOP_H_

#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "SparseParam.h"

// for sparse features
class SparseParams {
  public:
    SparseParam W;
    PAlphabet elems;
    int nVSize;
    int nDim;

  public:
    SparseParams() {
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
    }


    //random initialization
    void initial(PAlphabet alpha, int nOSize, int base = 1) {
        assert(base >= 1);
        elems = alpha;
        nVSize = base * elems->size();
        if (base > 1) {
            std::cout << "nVSize: " << nVSize << ", Alpha Size = " << elems->size()  << ", Require more Alpha."<< std::endl;
            elems->set_fixed_flag(false);
        }
        initialWeights(nOSize);
    }

    int getFeatureId(const string& strFeat) {
        int idx = elems->from_string(strFeat);
        if(!elems->m_b_fixed && elems->m_size >= nVSize) {
            std::cout << "Sparse Alphabet stopped collecting features" << std::endl;
            elems->set_fixed_flag(true);
        }
        return idx;
    }

};

//only implemented sparse linear node.
//non-linear transformations are not support,
class SparseNode : public Node {
  public:
    SparseParams* param;
    vector<int> ins;

    SparseNode() : Node() {
        ins.clear();
        param = NULL;
        node_type = "sparsenode";
    }

    void setParam(SparseParams* paramInit) {
        param = paramInit;
    }

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
    }

    void compute() {
        param->W.value(ins, val);
    }

    //no output losses
    void backward() {
        //assert(param != NULL);
        param->W.loss(ins, loss);
    }

    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        SparseNode* conv_other = (SparseNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};


class SparseExecute :public Execute {
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


PExecute SparseNode::generate(bool bTrain, dtype cur_drop_factor) {
    SparseExecute* exec = new SparseExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
}

#endif /* SPARSEOP_H_ */
