/*
 * TransferOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mason
 */

#ifndef TransferOP_H_
#define TransferOP_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class TransferParams {
  public:
    vector<Param> W;
    PAlphabet elems;
    int nVSize;
    int nInSize;
    int nOutSize;

  public:
    TransferParams() {
        nVSize = 0;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        for(int idx = 0; idx < nVSize; idx++) {
            ada.addParam(&(W[idx]));
        }
    }

    inline void initial(PAlphabet alpha, int nOSize, int nISize) {
        elems = alpha;
        nVSize = elems->size();
        nInSize = nISize;
        nOutSize = nOSize;
        W.resize(nVSize);
        for(int idx = 0; idx < nVSize; idx++) {
            W[idx].initial(nOSize, nISize);
        }
    }

    inline int getElemId(const string& strFeat) {
        return elems->from_string(strFeat);
    }

    // will add it
    inline void save(std::ofstream &os) const {

    }

    // will add it
    inline void load(std::ifstream &is) {

    }

};



class TransferNode : public Node {
  public:
    PNode in;
    int xid;
    TransferParams* param;

  public:
    TransferNode() : Node() {
        in = NULL;
        xid = -1;
        param = NULL;

    }


    inline void setParam(TransferParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
        xid = -1;
    }

  public:
    void forward(Graph *cg, PNode x, const string& strNorm) {
        in = x;
        xid = param->getElemId(strNorm);
        if (xid < 0) {
            std::cout << "TransferNode warning: could find the label: " << strNorm << std::endl;
        }
        degree = 0;
        in->addParent(this);
    }

  public:
    void compute() {
        if (xid >= 0) {
            val.mat() = param->W[xid].val.mat() * in->val.mat();
        }
    }

    void backward() {
        if(xid >= 0) {
            param->W[xid].grad.mat() += loss.mat() * in->val.tmat();
            in->loss.mat() += param->W[xid].val.mat().transpose() * loss.mat();
        }
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        TransferNode* conv_other = (TransferNode*)other;
        if (param != conv_other->param) {
            return false;
        }
        if (xid != conv_other->xid) {
            return false;
        }

        return true;
    }

};

#if USE_GPU
class TransferExecute :public Execute {
  public:
    Tensor2D x, y;
    int inDim, outDim;
    int xid;
    TransferParams* param;
    bool bTrain;

  public:
    inline void  forward() {
        int count = batch.size();
        x.init(inDim, count);
        y.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            TransferNode* ptr = (TransferNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                x[idx][idy] = ptr->in->val[idy];
            }
        }

        if (xid >= 0) {
            y.mat() = param->W[xid].val.mat() * x.mat();
        }

        for (int idx = 0; idx < count; idx++) {
            TransferNode* ptr = (TransferNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        Tensor2D lx, lty, ly;
        lx.init(inDim, count);
        lty.init(outDim, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            TransferNode* ptr = (TransferNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }
        }

        if (xid >= 0) {
            param->W[xid].grad.mat() += ly.mat() * x.mat().transpose();
            lx.mat() += param->W[xid].val.mat().transpose() * ly.mat();
        }

        for (int idx = 0; idx < count; idx++) {
            TransferNode* ptr = (TransferNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss[idy] += lx[idx][idy];
            }
        }
    }
};

inline PExecute TransferNode::generate(bool bTrain) {
    TransferExecute* exec = new TransferExecute();
    exec->batch.push_back(this);
    exec->inDim = param->nInSize;
    exec->outDim = param->nOutSize;
    exec->param = param;
    exec->xid = xid;
    exec->bTrain = bTrain;
    return exec;
}
#elif USE_BASE
class TransferExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            TransferNode* ptr = (TransferNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            TransferNode* ptr = (TransferNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute TransferNode::generate(bool bTrain) {
    TransferExecute* exec = new TransferExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->dim = dim;
    return exec;
};

#else
class TransferExecute :public Execute {
  public:
    Tensor2D x, y;
    int inDim, outDim;
    int xid;
    TransferParams* param;
    bool bTrain;

  public:
    inline void  forward() {
        int count = batch.size();
        x.init(inDim, count);
        y.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            TransferNode* ptr = (TransferNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                x[idx][idy] = ptr->in->val[idy];
            }
        }

        if (xid >= 0) {
            y.mat() = param->W[xid].val.mat() * x.mat();
        }

        for (int idx = 0; idx < count; idx++) {
            TransferNode* ptr = (TransferNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        if (xid < 0) return;
        int count = batch.size();
        Tensor2D lx, ly;
        lx.init(inDim, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            TransferNode* ptr = (TransferNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }
        }

        if (xid >= 0) {
            param->W[xid].grad.mat() += ly.mat() * x.mat().transpose();
            lx.mat() += param->W[xid].val.mat().transpose() * ly.mat();
        }

        for (int idx = 0; idx < count; idx++) {
            TransferNode* ptr = (TransferNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss[idy] += lx[idx][idy];
            }
        }
    }
};

inline PExecute TransferNode::generate(bool bTrain) {
    TransferExecute* exec = new TransferExecute();
    exec->batch.push_back(this);
    exec->inDim = param->nInSize;
    exec->outDim = param->nOutSize;
    exec->param = param;
    exec->xid = xid;
    exec->bTrain = bTrain;
    return exec;
}
#endif

#endif /* TransferOP_H_ */
