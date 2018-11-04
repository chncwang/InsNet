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

    void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W1);
        ada.addParam(&W2);
        ada.addParam(&W3);
        ada.addParam(&W4);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    void initial(int nOSize, int nISize1, int nISize2, int nISize3, int nISize4, bool useB = true) {
        W1.initial(nOSize, nISize1);
        W2.initial(nOSize, nISize2);
        W3.initial(nOSize, nISize3);
        W4.initial(nOSize, nISize4);

        bUseB = useB;
        if (bUseB) {
            b.initial(nOSize, 1);
        }
    }

    void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        W1.save(os);
        W2.save(os);
        W3.save(os);
        W4.save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    void load(std::ifstream &is) {
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

    void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
        ty.init(ndim);
        lty.init(ndim);
    }

    void setParam(FourParams* paramInit) {
        param = paramInit;
    }

    // define the activate function and its derivation form
    void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
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
    void compute() {
        ty.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat()
                   + param->W3.val.mat() * in3->val.mat() + param->W4.val.mat() * in4->val.mat();
        if (param->bUseB) {
            ty.vec() += param->b.val.vec();
        }
        val.vec() = ty.vec().unaryExpr(ptr_fun(activate));
    }

    void backward() {
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
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
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

    void setParam(FourParams* paramInit) {
        param = paramInit;
    }

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
    void compute() {
        val.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat()
                    + param->W3.val.mat() * in3->val.mat() + param->W4.val.mat() * in4->val.mat();

        if (param->bUseB) {
            val.vec() += param->b.val.vec();
        }
    }

    void backward() {
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
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
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
    Tensor2D x1, x2, x3, x4, ty, y, b;
    Tensor2D drop_mask;
    int inDim1, inDim2, inDim3, inDim4, outDim;
    FourParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
public:

    void  forward() {
        int count = batch.size();
        x1.init(inDim1, count);
        x2.init(inDim2, count);
        x3.init(inDim3, count);
        x4.init(inDim4, count);
        b.init(outDim, count);
        ty.init(outDim, count);
        y.init(outDim, count);
        drop_mask.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            FourNode* ptr = (FourNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                x1[idx][idy] = ptr->in1->val[idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                x2[idx][idy] = ptr->in2->val[idy];
            }
            for (int idy = 0; idy < inDim3; idy++) {
                x3[idx][idy] = ptr->in3->val[idy];
            }
            for (int idy = 0; idy < inDim4; idy++) {
                x4[idx][idy] = ptr->in4->val[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idx][idy] = param->b.val.v[idy];
                }
            }
        }
        ty.mat() = param->W1.val.mat() * x1.mat() + param->W2.val.mat() * x2.mat() + param->W3.val.mat() * x3.mat() + param->W4.val.mat() * x4.mat();
        if (param->bUseB) {
            ty.vec() = ty.vec() + b.vec();
        }
        y.vec() = ty.vec().unaryExpr(ptr_fun(activate));
        for (int idx = 0; idx < count; idx++) {
            FourNode* ptr = (FourNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
            ptr->forward_drop(bTrain, drop_factor / batch.at(0)->drop_value);
        }
    }

    void backward() {
        int count = batch.size();
        Tensor2D lx1, lx2, lx3, lx4, lty, ly;
        lx1.init(inDim1, count);
        lx2.init(inDim2, count);
        lx3.init(inDim3, count);
        lx4.init(inDim4, count);
        lty.init(outDim, count);
        ly.init(outDim, count);
        for (int idx = 0; idx < count; idx++) {
            FourNode* ptr = (FourNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }
        }
        lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));
        param->W1.grad.mat() += lty.mat() * x1.mat().transpose();
        param->W2.grad.mat() += lty.mat() * x2.mat().transpose();
        param->W3.grad.mat() += lty.mat() * x3.mat().transpose();
        param->W4.grad.mat() += lty.mat() * x4.mat().transpose();
        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += lty[idx][idy];
                }
            }
        }
        lx1.mat() += param->W1.val.mat().transpose() * lty.mat();
        lx2.mat() += param->W2.val.mat().transpose() * lty.mat();
        lx3.mat() += param->W3.val.mat().transpose() * lty.mat();
        lx4.mat() += param->W4.val.mat().transpose() * lty.mat();
        for (int idx = 0; idx < count; idx++) {
            FourNode* ptr = (FourNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                ptr->in1->loss[idy] += lx1[idx][idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                ptr->in2->loss[idy] += lx2[idx][idy];
            }
            for (int idy = 0; idy < inDim3; idy++) {
                ptr->in3->loss[idy] += lx3[idx][idy];
            }
            for (int idy = 0; idy < inDim4; idy++) {
                ptr->in4->loss[idy] += lx4[idx][idy];
            }
        }
    }
};

PExecute FourNode::generate(bool bTrain, dtype cur_drop_factor) {
    FourExecute* exec = new FourExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor * drop_value;
    exec->inDim1 = param->W1.inDim();
    exec->inDim2 = param->W2.inDim();
    exec->inDim3 = param->W3.inDim();
    exec->inDim4 = param->W4.inDim();
    exec->outDim = param->W1.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;
    return exec;
};

class LinearFourExecute :public Execute {
public:
    Tensor2D x1, x2, x3, x4, y, b;
    int inDim1, inDim2, inDim3, inDim4, outDim, count;
    FourParams* param;

public:
    void  forward() {
        count = batch.size();
        x1.init(inDim1, count);
        x2.init(inDim2, count);
        x3.init(inDim3, count);
        x4.init(inDim4, count);
        b.init(outDim, count);
        y.init(outDim, count);


        for (int idx = 0; idx < count; idx++) {
            LinearFourNode* ptr = (LinearFourNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                x1[idx][idy] = ptr->in1->val[idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                x2[idx][idy] = ptr->in2->val[idy];
            }
            for (int idy = 0; idy < inDim3; idy++) {
                x3[idx][idy] = ptr->in3->val[idy];
            }
            for (int idy = 0; idy < inDim4; idy++) {
                x4[idx][idy] = ptr->in4->val[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idx][idy] = param->b.val.v[idy];
                }
            }
        }

        y.mat() = param->W1.val.mat() * x1.mat() + param->W2.val.mat() * x2.mat() + param->W3.val.mat() * x3.mat() + param->W4.val.mat() * x4.mat();

        if (param->bUseB) {
            y.vec() += b.vec();
        }

        for (int idx = 0; idx < count; idx++) {
            LinearFourNode* ptr = (LinearFourNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
            ptr->forward_drop(bTrain, drop_factor);
        }
    }

    void backward() {
        Tensor2D lx1, lx2, lx3, lx4, ly;
        lx1.init(inDim1, count);
        lx2.init(inDim2, count);
        lx3.init(inDim3, count);
        lx4.init(inDim4, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            LinearFourNode* ptr = (LinearFourNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }
        }

        param->W1.grad.mat() += ly.mat() * x1.mat().transpose();
        param->W2.grad.mat() += ly.mat() * x2.mat().transpose();
        param->W3.grad.mat() += ly.mat() * x3.mat().transpose();
        param->W4.grad.mat() += ly.mat() * x4.mat().transpose();

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += ly[idx][idy];
                }
            }
        }

        lx1.mat() += param->W1.val.mat().transpose() * ly.mat();
        lx2.mat() += param->W2.val.mat().transpose() * ly.mat();
        lx3.mat() += param->W3.val.mat().transpose() * ly.mat();
        lx4.mat() += param->W4.val.mat().transpose() * ly.mat();

        for (int idx = 0; idx < count; idx++) {
            LinearFourNode* ptr = (LinearFourNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                ptr->in1->loss[idy] += lx1[idx][idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                ptr->in2->loss[idy] += lx2[idx][idy];
            }
            for (int idy = 0; idy < inDim3; idy++) {
                ptr->in3->loss[idy] += lx3[idx][idy];
            }
            for (int idy = 0; idy < inDim4; idy++) {
                ptr->in4->loss[idy] += lx4[idx][idy];
            }
        }

    }
};

PExecute LinearFourNode::generate(bool bTrain, dtype cur_drop_factor) {
    LinearFourExecute* exec = new LinearFourExecute();
    exec->batch.push_back(this);
    exec->drop_factor = cur_drop_factor * drop_value;
    exec->inDim1 = param->W1.inDim();
    exec->inDim2 = param->W2.inDim();
    exec->inDim3 = param->W3.inDim();
    exec->inDim4 = param->W4.inDim();
    exec->outDim = param->W1.outDim();
    exec->param = param;
    exec->bTrain = bTrain;
    return exec;
};

#endif /* FOUROP_H_ */
