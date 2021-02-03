#ifndef SPARSEOP_H_
#define SPARSEOP_H_

#include "MyLib.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "SparseParam.h"
#include "ModelUpdate.h"

// for sparse features
class SparseParams {
  public:
    SparseParam W;
    Alphabet elems;
    int nVSize;
    int nDim;

  public:
    SparseParams() {
        nVSize = 0;
        nDim = 0;
    }

    void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W);
    }

    void initWeights(int nOSize) {
        if (nVSize == 0) {
            std::cerr << "nVSize is 0" << std::endl;
            abort();
        }
        nDim = nOSize;
        W.init(nOSize, nVSize);
    }


    void init(const Alphabet &alpha, int nOSize) {
        elems = alpha;
        nVSize = elems.size();
        initWeights(nOSize);
    }

    int getFeatureId(const string& strFeat) {
        int idx = elems.from_string(strFeat);
        return idx;
    }

};

#endif /* SPARSEOP_H_ */
