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
    inline void initial(PAlphabet alpha, int nOSize) {
        elems = alpha;
        nVSize = elems->size();
        initialWeights(nOSize);
    }

    inline int getFeatureId(const string& strFeat) {
        return elems->from_string(strFeat);
    }

};

//only implemented sparse linear node.
//non-linear transformations are not support,
struct APNode : Node {
  public:
    APParams* param;
    vector<int> tx;

  public:
    APNode() : Node() {
        tx.clear();
        param = NULL;
    }

    inline void setParam(APParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        tx.clear();
    }

  public:
    //notice the output
    void forward(Graph *cg, const vector<string>& x) {
        int featId;
        int featSize = x.size();
        for (int idx = 0; idx < featSize; idx++) {
            featId = param->getFeatureId(x[idx]);
            if (featId >= 0) {
                tx.push_back(featId);
            }
        }
        param->W.value(featId, val, cg->train);
        cg->addNode(this);
    }

    //no output losses
    void backward() {
        //assert(param != NULL);
        param->W.loss(tx, loss);
    }

};

#endif /* APOP_H_ */
