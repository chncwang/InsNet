#ifndef TRIOP_H_
#define TRIOP_H_

/*
*  TriOP.h:
*  a simple feed forward neural operation, triple input.
*
*  Created on: June 11, 2017
*      Author: mszhang
*/


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class TriParams {
  public:
    Param W1;
    Param W2;
    Param W3;
    Param b;

    bool bUseB;

  public:
    TriParams() {
        bUseB = true;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W1);
        ada.addParam(&W2);
        ada.addParam(&W3);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    inline void initial(int nOSize, int nISize1, int nISize2, int nISize3, bool useB = true) {
        W1.initial(nOSize, nISize1);
        W2.initial(nOSize, nISize2);
        W3.initial(nOSize, nISize3);

        bUseB = useB;
        if (bUseB) {
            b.initial(nOSize, 1);
        }
    }

    inline void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        W1.save(os);
        W2.save(os);
        W3.save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    inline void load(std::ifstream &is) {
        is >> bUseB;
        W1.load(is);
        W2.load(is);
        W3.load(is);
        if (bUseB) {
            b.load(is);
        }
    }

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class TriNode : public Node {
  public:
    PNode in1, in2, in3;
    TriParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    Tensor1D ty, lty;


  public:
    TriNode() : Node() {
        in1 = in2 = in3 = NULL;
        activate = ftanh;
        derivate = dtanh;
        param = NULL;
        node_type = "tri";
    }

    ~TriNode() {
        in1 = in2 = in3 = NULL;
    }

    inline void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
        ty.init(ndim);
        lty.init(ndim);
    }

    inline void setParam(TriParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in1 = in2 = in3 = NULL;
        ty = 0;
        lty = 0;
    }

    // define the activate function and its derivation form
    inline void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
        activate = f;
        derivate = f_deri;
    }

  public:
    void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
        in1 = x1;
        in2 = x2;
        in3 = x3;
        degree = 0;
        in1->addParent(this);
        in2->addParent(this);
        in3->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        ty.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat() + param->W3.val.mat() * in3->val.mat();
        if (param->bUseB) {
            ty.vec() += param->b.val.vec();
        }
        val.vec() = ty.vec().unaryExpr(ptr_fun(activate));
    }

    inline void backward() {
        lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));

        param->W1.grad.mat() += lty.mat() * in1->val.tmat();
        param->W2.grad.mat() += lty.mat() * in2->val.tmat();
        param->W3.grad.mat() += lty.mat() * in3->val.tmat();

        if (param->bUseB) {
            param->b.grad.vec() += lty.vec();
        }

        in1->loss.mat() += param->W1.val.mat().transpose() * lty.mat();
        in2->loss.mat() += param->W2.val.mat().transpose() * lty.mat();
        in3->loss.mat() += param->W3.val.mat().transpose() * lty.mat();
    }

  public:
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        TriNode* conv_other = (TriNode*)other;
        if (param != conv_other->param) {
            return false;
        }
        if (activate != conv_other->activate || derivate != conv_other->derivate) {
            return false;
        }

        return true;
    }

};


// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class LinearTriNode : public Node {
  public:
    PNode in1, in2, in3;
    TriParams* param;

  public:
    LinearTriNode() : Node() {
        in1 = in2 = in3 = NULL;
        param = NULL;
        node_type = "linear_tri";
    }

    inline void setParam(TriParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in1 = in2 = in3 = NULL;
    }

  public:
    void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
        in1 = x1;
        in2 = x2;
        in3 = x3;
        degree = 0;
        in1->addParent(this);
        in2->addParent(this);
        in3->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        val.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat() + param->W3.val.mat() * in3->val.mat();

        if (param->bUseB) {
            val.vec() += param->b.val.vec();
        }
    }

    inline void backward() {
        param->W1.grad.mat() += loss.mat() * in1->val.tmat();
        param->W2.grad.mat() += loss.mat() * in2->val.tmat();
        param->W3.grad.mat() += loss.mat() * in3->val.tmat();

        if (param->bUseB) {
            param->b.grad.vec() += loss.vec();
        }

        in1->loss.mat() += param->W1.val.mat().transpose() * loss.mat();
        in2->loss.mat() += param->W2.val.mat().transpose() * loss.mat();
        in3->loss.mat() += param->W3.val.mat().transpose() * loss.mat();
    }

  public:
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        LinearTriNode* conv_other = (LinearTriNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};


class TriExecute :public Execute {
  public:
    bool bTrain;
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

inline PExecute TriNode::generate(bool bTrain, dtype cur_drop_factor) {
    TriExecute* exec = new TriExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};

class LinearTriExecute :public Execute {
  public:
    bool bTrain;
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

inline PExecute LinearTriNode::generate(bool bTrain, dtype cur_drop_factor) {
    LinearTriExecute* exec = new LinearTriExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};

#endif /* TRIOP_H_ */
