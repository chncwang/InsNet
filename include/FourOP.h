#ifndef FOUROP_H_
#define FOUROP_H_

/*
*  FourOP.h:
*  a simple feed forward neural operation, four variable input.
*
*  Created on: June 11, 2017
*      Author: mszhang
*/

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class FourParams {
  public:
    Param W1;
    Param W2;
    Param W3;
    Param W4;
    Param b;

    bool bUseB;

  public:
    FourParams() {
        bUseB = true;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W1);
        ada.addParam(&W2);
        ada.addParam(&W3);
        ada.addParam(&W4);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    inline void initial(int nOSize, int nISize1, int nISize2, int nISize3, int nISize4, bool useB = true) {
        W1.initial(nOSize, nISize1);
        W2.initial(nOSize, nISize2);
        W3.initial(nOSize, nISize3);
        W4.initial(nOSize, nISize4);

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
        W4.save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    inline void load(std::ifstream &is) {
        is >> bUseB;
        W1.load(is);
        W2.load(is);
        W3.load(is);
        W4.load(is);
        if (bUseB) {
            b.load(is);
        }
    }

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class FourNode : public Node {
  public:
    PNode in1, in2, in3, in4;
    FourParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    Tensor1D ty, lty;


  public:
    FourNode() : Node() {
        in1 = in2 = in3 = in4 = NULL;
        activate = ftanh;
        derivate = dtanh;
        param = NULL;
        node_type = "four";
    }

    ~FourNode() {
        in1 = in2 = in3 = in4 = NULL;
    }

    inline void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
        ty.init(ndim);
        lty.init(ndim);
    }

    inline void setParam(FourParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in1 = in2 = in3 = in4 = NULL;
        ty = 0;
        lty = 0;
    }

    // define the activate function and its derivation form
    inline void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
        activate = f;
        derivate = f_deri;
    }

  public:
    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
        in1 = x1;
        in2 = x2;
        in3 = x3;
        in4 = x4;
        degree = 0;
        in1->addParent(this);
        in2->addParent(this);
        in3->addParent(this);
        in4->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        ty.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat()
                   + param->W3.val.mat() * in3->val.mat() + param->W4.val.mat() * in4->val.mat();
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
        param->W4.grad.mat() += lty.mat() * in4->val.tmat();

        if (param->bUseB) {
            param->b.grad.vec() += lty.vec();
        }

        in1->loss.mat() += param->W1.val.mat().transpose() * lty.mat();
        in2->loss.mat() += param->W2.val.mat().transpose() * lty.mat();
        in3->loss.mat() += param->W3.val.mat().transpose() * lty.mat();
        in4->loss.mat() += param->W4.val.mat().transpose() * lty.mat();
    }

  public:
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        FourNode* conv_other = (FourNode*)other;
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
class LinearFourNode : public Node {
  public:
    PNode in1, in2, in3, in4;
    FourParams* param;

  public:
    LinearFourNode() : Node() {
        in1 = in2 = in3 = in4 = NULL;
        param = NULL;
        node_type = "linear_four";
    }

    inline void setParam(FourParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in1 = in2 = in3 = in4 = NULL;
    }

  public:
    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
        in1 = x1;
        in2 = x2;
        in3 = x3;
        in4 = x4;
        degree = 0;
        in1->addParent(this);
        in2->addParent(this);
        in3->addParent(this);
        in4->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        val.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat()
                    + param->W3.val.mat() * in3->val.mat() + param->W4.val.mat() * in4->val.mat();

        if (param->bUseB) {
            val.vec() += param->b.val.vec();
        }
    }

    inline void backward() {
        param->W1.grad.mat() += loss.mat() * in1->val.tmat();
        param->W2.grad.mat() += loss.mat() * in2->val.tmat();
        param->W3.grad.mat() += loss.mat() * in3->val.tmat();
        param->W4.grad.mat() += loss.mat() * in4->val.tmat();

        if (param->bUseB) {
            param->b.grad.vec() += loss.vec();
        }

        in1->loss.mat() += param->W1.val.mat().transpose() * loss.mat();
        in2->loss.mat() += param->W2.val.mat().transpose() * loss.mat();
        in3->loss.mat() += param->W3.val.mat().transpose() * loss.mat();
        in4->loss.mat() += param->W4.val.mat().transpose() * loss.mat();
    }

  public:
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        LinearFourNode* conv_other = (LinearFourNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};


class FourExecute :public Execute {
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

inline PExecute FourNode::generate(bool bTrain, dtype cur_drop_factor) {
    FourExecute* exec = new FourExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};

class LinearFourExecute :public Execute {
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

inline PExecute LinearFourNode::generate(bool bTrain, dtype cur_drop_factor) {
    LinearFourExecute* exec = new LinearFourExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};

#endif /* FOUROP_H_ */
