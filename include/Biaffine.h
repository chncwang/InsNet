#ifndef Biaffine_H_
#define Biaffine_H_

/*
*  Biaffine_H_.h:
*  a simple feed forward neural operation, binary input.
*
*  Created on: June 11, 2017
*      Author: yue zhang (suda)
*/
/*
   This file will be modified by meishan zhang, currently better not use it
*/


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class BiaffineParams {
  public:
    vector<Param> W;
    Param b;
    bool bUseB;
    int classDim;

  public:
    BiaffineParams() {
        bUseB = true;
        classDim = 0;
    }

    void exportAdaParams(ModelUpdate& ada) {
        for (int i = 0; i < classDim; i++) {
            ada.addParam(&W[i]);
        }
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    void initial(int nISize1, int nISize2, bool useB = true, int classDims = 1) {
        classDim = classDims;
        W.resize(classDim);
        for (int i = 0; i < classDim; i++) {
            W[i].initial(nISize1, nISize2);
        }
        bUseB = useB;
        if (bUseB) {
            b.initial(classDim, 1);
        }
    }

    void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        for (int i = 0; i < classDim; i++)
            W[i].save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    void load(std::ifstream &is) {
        is >> bUseB;
        for (int i = 0; i < classDim; i++)
            W[i].load(is);
        if (bUseB) {
            b.load(is);
        }
    }
};

class BiaffineNode : public Node {
  public:
    vector<PNode> in1, in2;
    Tensor2D x1, x2;//concat nodes in in1 by y

    bool expandIn1, expandIn2;
    BiaffineParams* param;
    int nSize;
    int classDim;
    int inDim1, inDim2;

    vector<Tensor2D> vals;
    vector<Tensor2D> losses;
    vector<Tensor2D> y1;

  public:
    BiaffineNode() : Node() {
        nSize = 0;
        classDim = 0;
        in1.clear();
        in2.clear();
        vals.clear();
        losses.clear();
        param = NULL;
        node_type = "biaffine";
    }

    void setParam(BiaffineParams* paramInit, bool expandIns1, bool expandIns2) {
        param = paramInit;
        classDim = paramInit->classDim;
        expandIn1 = expandIns1;
        expandIn2 = expandIns2;
    }

    void init(int dim) {
        this->dim = dim;
        vals.resize(classDim);
        losses.resize(classDim);
        for (int i = 0; i < classDim; i++) {
            vals[i].init(dim, dim);
            losses[i].init(dim, dim);
        }
        parents.clear();
    }

  public:
    void forward(Graph *cg, vector<PNode> x1, vector<PNode> x2) {
        assert(x1.size() == x2.size());
        nSize = x1.size();
        for (int i = 0; i < nSize; i++) {
            in1.push_back(x1[i]);
            in2.push_back(x2[i]);
        }
        degree = in1.size() + in2.size();
        for (int i = 0; i < nSize; i++) {
            in1[i]->parents.push_back(this);
            in2[i]->parents.push_back(this);
        }
        cg->addNode(this);
    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;
        BiaffineNode* conv_other = (BiaffineNode*)other;
        if (param == conv_other->param) {
            return true;
        } else
            return false;
    }
  public:
    void compute() {
        inDim1 = in1[0]->dim;
        inDim2 = in2[0]->dim;
        x1.init(inDim1 + (expandIn1 ? 1 : 0), nSize);
        x2.init(inDim2 + (expandIn2 ? 1 : 0), nSize);
        y1.resize(classDim);
        for (int i = 0; i < classDim; i++) {
            y1[i].init(nSize, inDim2 + (expandIn2 ? 1 : 0));
        }
        for (int i = 0; i < nSize; ++i) {
            for (int j = 0; j < inDim1; j++)
                x1[i][j] = in1[i]->val[j];
            if (expandIn1)
                x1[i][inDim1] = 1;

            for (int j = 0; j < inDim2; j++)
                x2[i][j] = in2[i]->val[j];
            if (expandIn2)
                x2[i][inDim2] = 1;
        }
        for (int i = 0; i < classDim; i++) {
            y1[i].mat() = x1.mat().transpose() * param->W[i].val.mat();
            vals[i].mat() = y1[i].mat() * x2.mat();
            if (param->bUseB) {
                for (int idx = 0; idx < nSize; idx++)
                    for (int idy = 0; idy < nSize; idy++)
                        vals[i].mat()(idx, idy) += param->b.val.mat()(i, 0);
            }
        }
    }

    void backward() {
        vector<Tensor2D> lx1, lx2, ly1;
        lx1.resize(classDim);
        lx2.resize(classDim);
        ly1.resize(classDim);
        for (int i = 0; i < classDim; i++) {
            lx1[i].init(inDim1 + (expandIn1 ? 1 : 0), nSize);
            lx2[i].init(inDim2 + (expandIn2 ? 1 : 0), nSize);
            ly1[i].init(nSize, inDim2 + (expandIn2 ? 1 : 0));
        }

        for (int i = 0; i < classDim; i++) {
            lx2[i].mat() = y1[i].mat().transpose() * losses[i].mat();
            ly1[i].mat() = losses[i].mat() * x2.mat().transpose();
            lx1[i].mat() = param->W[i].val.mat() * ly1[i].mat().transpose();
            param->W[i].grad.mat() += x1.mat() * ly1[i].mat();
        }

        if (param->bUseB) {
            for (int i = 0; i < classDim; i++) {
                for (int idx = 0; idx < nSize; idx++) {
                    for (int idy = 0; idy < nSize; idy++)
                        param->b.grad.v[i] += losses[i][idx][idy];
                }
            }
        }
        for (int i = 0; i < classDim; i++) {
            for (int idx = 0; idx < nSize; idx++) {
                for (int idy = 0; idy < inDim1; idy++) {
                    in1[idx]->loss[idy] += lx1[i][idx][idy];
                }
                for (int idy = 0; idy < inDim2; idy++) {
                    in2[idx]->loss[idy] += lx2[i][idx][idy];
                }
            }
        }
    }
};

class BiaffineExecute :public Execute {
  public:
    void  forward() {
        int count = batch.size();

        for (int idx = 0; idx < count; idx++) {
            BiaffineNode* ptr = (BiaffineNode*)batch[idx];
            ptr->compute();
        }
    }

    void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            BiaffineNode* ptr = (BiaffineNode*)batch[idx];
            ptr->backward();
        }
    }
};

PExecute BiaffineNode::generate(bool bTrain, dtype cur_drop_factor) {
    BiaffineExecute* exec = new BiaffineExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};

#endif /* Biaffine_H_ */
