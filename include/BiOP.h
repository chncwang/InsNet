#ifndef BIOP_H_
#define BIOP_H_

/*
*  BiOP.h:
*  a simple feed forward neural operation, binary input.
*
*  Created on: June 11, 2017
*      Author: mszhang
*/


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class BiParams {
  public:
    Param W1;
    Param W2;
    Param b;

    bool bUseB;

  public:
    BiParams() {
        bUseB = true;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W1);
        ada.addParam(&W2);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    inline void initial(int nOSize, int nISize1, int nISize2, bool useB = true) {
        W1.initial(nOSize, nISize1);
        W2.initial(nOSize, nISize2);
        bUseB = useB;
        if (bUseB) {
            b.initial(nOSize, 1);
        }
    }

    inline void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        W1.save(os);
        W2.save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    inline void load(std::ifstream &is) {
        is >> bUseB;
        W1.load(is);
        W2.load(is);
        if (bUseB) {
            b.load(is);
        }
    }

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class BiNode : public Node {
  public:
    PNode in1, in2;
    BiParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function


  public:
    BiNode() : Node() {
        in1 = in2 = NULL;
        activate = ftanh;
        derivate = dtanh;
        param = NULL;
        node_type = "bi";
    }

    ~BiNode() {
        in1 = in2 = NULL;
    }


    inline void setParam(BiParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in1 = in2 = NULL;
    }

    // define the activate function and its derivation form
    inline void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
        activate = f;
        derivate = f_deri;
    }


  public:
    void forward(Graph *cg, PNode x1, PNode x2) {
        in1 = x1;
        in2 = x2;
        degree = 0;
        in1->addParent(this);
        in2->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute(Tensor1D& ty) {
        ty.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat();
        if (param->bUseB) {
            ty.vec() += param->b.val.vec();
        }
        val.vec() = ty.vec().unaryExpr(ptr_fun(activate));
    }

    inline void backward(Tensor1D& ty, Tensor1D& lty) {
        lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));

        param->W1.grad.mat() += lty.mat() * in1->val.tmat();
        param->W2.grad.mat() += lty.mat() * in2->val.tmat();

        if (param->bUseB) {
            param->b.grad.vec() += lty.vec();
        }

        in1->loss.mat() += param->W1.val.mat().transpose() * lty.mat();
        in2->loss.mat() += param->W2.val.mat().transpose() * lty.mat();
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        BiNode* conv_other = (BiNode*)other;
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
class LinearBiNode : public Node {
  public:
    PNode in1, in2;
    BiParams* param;

  public:
    LinearBiNode() : Node() {
        in1 = in2 = NULL;
        param = NULL;
        node_type = "linear_bi";
    }

    inline void setParam(BiParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in1 = in2 = NULL;
    }


  public:
    void forward(Graph *cg, PNode x1, PNode x2) {
        in1 = x1;
        in2 = x2;
        degree = 0;
        in1->addParent(this);
        in2->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        val.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat();

        if (param->bUseB) {
            val.vec() += param->b.val.vec();
        }
    }

    inline void backward() {
        param->W1.grad.mat() += loss.mat() * in1->val.tmat();
        param->W2.grad.mat() += loss.mat() * in2->val.tmat();

        if (param->bUseB) {
            param->b.grad.vec() += loss.vec();
        }

        in1->loss.mat() += param->W1.val.mat().transpose() * loss.mat();
        in2->loss.mat() += param->W2.val.mat().transpose() * loss.mat();
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        LinearBiNode* conv_other = (LinearBiNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};


#if USE_GPU
class BiExecute :public Execute {
  public:
    Tensor2D x1, x2, ty, y, b;
    int inDim1, inDim2, outDim;
    BiParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    bool bTrain;

  public:
    ~BiExecute() {
        param = NULL;
        activate = NULL;
        derivate = NULL;
        inDim1 = inDim2 = outDim = 0;
    }


  public:
    inline void  forward() {
        int count = batch.size();
        x1.init(inDim1, count);
        x2.init(inDim2, count);
        b.init(outDim, count);
        ty.init(outDim, count);
        y.init(outDim, count);


        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                x1[idx][idy] = ptr->in1->val[idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                x2[idx][idy] = ptr->in2->val[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idx][idy] = param->b.val.v[idy];
                }
            }
        }

        ty.mat() = param->W1.val.mat() * x1.mat() + param->W2.val.mat() * x2.mat();

        if (param->bUseB) {
            ty.vec() = ty.vec() + b.vec();
        }

        y.vec() = ty.vec().unaryExpr(ptr_fun(activate));

        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        Tensor2D lx1, lx2, lty, ly;
        lx1.init(inDim1, count);
        lx2.init(inDim2, count);
        lty.init(outDim, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }
        }

        lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));

        param->W1.grad.mat() += lty.mat() * x1.mat().transpose();
        param->W2.grad.mat() += lty.mat() * x2.mat().transpose();

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += lty[idx][idy];
                }
            }
        }

        lx1.mat() += param->W1.val.mat().transpose() * lty.mat();
        lx2.mat() += param->W2.val.mat().transpose() * lty.mat();

        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                ptr->in1->loss[idy] += lx1[idx][idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                ptr->in2->loss[idy] += lx2[idx][idy];
            }
        }
    }
};

class LinearBiExecute :public Execute {
  public:
    Tensor2D x1, x2, y, b;
    int inDim1, inDim2, outDim, count;
    BiParams* param;
    bool bTrain;

  public:
    inline void  forward() {
        count = batch.size();
        x1.init(inDim1, count);
        x2.init(inDim2, count);
        b.init(outDim, count);
        y.init(outDim, count);


        for (int idx = 0; idx < count; idx++) {
            LinearBiNode* ptr = (LinearBiNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                x1[idx][idy] = ptr->in1->val[idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                x2[idx][idy] = ptr->in2->val[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idx][idy] = param->b.val.v[idy];
                }
            }
        }

        y.mat() = param->W1.val.mat() * x1.mat() + param->W2.val.mat() * x2.mat();

        if (param->bUseB) {
            y.vec() += b.vec();
        }

        for (int idx = 0; idx < count; idx++) {
            LinearBiNode* ptr = (LinearBiNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        Tensor2D lx1, lx2, ly;
        lx1.init(inDim1, count);
        lx2.init(inDim2, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            LinearBiNode* ptr = (LinearBiNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }
        }

        param->W1.grad.mat() += ly.mat() * x1.mat().transpose();
        param->W2.grad.mat() += ly.mat() * x2.mat().transpose();

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += ly[idx][idy];
                }
            }
        }

        lx1.mat() += param->W1.val.mat().transpose() * ly.mat();
        lx2.mat() += param->W2.val.mat().transpose() * ly.mat();

        for (int idx = 0; idx < count; idx++) {
            LinearBiNode* ptr = (LinearBiNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                ptr->in1->loss[idy] += lx1[idx][idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                ptr->in2->loss[idy] += lx2[idx][idy];
            }
        }

    }
};


inline PExecute BiNode::generate(bool bTrain) {
    BiExecute* exec = new BiExecute();
    exec->batch.push_back(this);
    exec->inDim1 = param->W1.inDim();
    exec->inDim2 = param->W2.inDim();
    exec->outDim = param->W1.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;
    exec->bTrain = bTrain;
    return exec;
}


inline PExecute LinearBiNode::generate(bool bTrain) {
    LinearBiExecute* exec = new LinearBiExecute();
    exec->batch.push_back(this);
    exec->inDim1 = param->W1.inDim();
    exec->inDim2 = param->W2.inDim();
    exec->outDim = param->W1.outDim();
    exec->param = param;
    exec->bTrain = bTrain;
    return exec;
}
#elif USE_BASE
class BiExecute :public Execute {
  public:
    bool bTrain;
    int dim;
    vector<Tensor1D> tys, ltys;
  public:
    inline void  forward() {
        int count = batch.size();
        tys.resize(count);
        for (int idx = 0; idx < count; idx++) {
            tys[idx].init(dim, NULL);
        }
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            ptr->compute(tys[idx]);
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        ltys.resize(count);
        for (int idx = 0; idx < count; idx++) {
            ltys[idx].init(dim, NULL);
        }
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward(tys[idx], ltys[idx]);
        }
    }
};

inline PExecute BiNode::generate(bool bTrain) {
    BiExecute* exec = new BiExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->dim = dim;
    return exec;
};

class LinearBiExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            LinearBiNode* ptr = (LinearBiNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            LinearBiNode* ptr = (LinearBiNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute LinearBiNode::generate(bool bTrain) {
    LinearBiExecute* exec = new LinearBiExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};
#else
class BiExecute :public Execute {
  public:
    Tensor2D x1, x2, ty, y, b;
    int inDim1, inDim2, outDim;
    BiParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    bool bTrain;

  public:
    ~BiExecute() {
        param = NULL;
        activate = NULL;
        derivate = NULL;
        inDim1 = inDim2 = outDim = 0;
    }


  public:
    inline void  forward() {
        int count = batch.size();
        x1.init(inDim1, count);
        x2.init(inDim2, count);
        b.init(outDim, count);
        ty.init(outDim, count);
        y.init(outDim, count);


        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                x1[idx][idy] = ptr->in1->val[idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                x2[idx][idy] = ptr->in2->val[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idx][idy] = param->b.val.v[idy];
                }
            }
        }

        ty.mat() = param->W1.val.mat() * x1.mat() + param->W2.val.mat() * x2.mat();

        if (param->bUseB) {
            ty.vec() = ty.vec() + b.vec();
        }

        y.vec() = ty.vec().unaryExpr(ptr_fun(activate));

        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        Tensor2D lx1, lx2, lty, ly;
        lx1.init(inDim1, count);
        lx2.init(inDim2, count);
        lty.init(outDim, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }
        }

        lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));

        param->W1.grad.mat() += lty.mat() * x1.mat().transpose();
        param->W2.grad.mat() += lty.mat() * x2.mat().transpose();

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += lty[idx][idy];
                }
            }
        }

        lx1.mat() += param->W1.val.mat().transpose() * lty.mat();
        lx2.mat() += param->W2.val.mat().transpose() * lty.mat();

        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                ptr->in1->loss[idy] += lx1[idx][idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                ptr->in2->loss[idy] += lx2[idx][idy];
            }
        }
    }
};

class LinearBiExecute :public Execute {
  public:
    Tensor2D x1, x2, y, b;
    int inDim1, inDim2, outDim, count;
    BiParams* param;
    bool bTrain;

  public:
    inline void  forward() {
        count = batch.size();
        x1.init(inDim1, count);
        x2.init(inDim2, count);
        b.init(outDim, count);
        y.init(outDim, count);


        for (int idx = 0; idx < count; idx++) {
            LinearBiNode* ptr = (LinearBiNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                x1[idx][idy] = ptr->in1->val[idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                x2[idx][idy] = ptr->in2->val[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idx][idy] = param->b.val.v[idy];
                }
            }
        }

        y.mat() = param->W1.val.mat() * x1.mat() + param->W2.val.mat() * x2.mat();

        if (param->bUseB) {
            y.vec() += b.vec();
        }

        for (int idx = 0; idx < count; idx++) {
            LinearBiNode* ptr = (LinearBiNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        Tensor2D lx1, lx2, ly;
        lx1.init(inDim1, count);
        lx2.init(inDim2, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            LinearBiNode* ptr = (LinearBiNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }
        }

        param->W1.grad.mat() += ly.mat() * x1.mat().transpose();
        param->W2.grad.mat() += ly.mat() * x2.mat().transpose();

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += ly[idx][idy];
                }
            }
        }

        lx1.mat() += param->W1.val.mat().transpose() * ly.mat();
        lx2.mat() += param->W2.val.mat().transpose() * ly.mat();

        for (int idx = 0; idx < count; idx++) {
            LinearBiNode* ptr = (LinearBiNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                ptr->in1->loss[idy] += lx1[idx][idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                ptr->in2->loss[idy] += lx2[idx][idy];
            }
        }

    }
};


inline PExecute BiNode::generate(bool bTrain) {
    BiExecute* exec = new BiExecute();
    exec->batch.push_back(this);
    exec->inDim1 = param->W1.inDim();
    exec->inDim2 = param->W2.inDim();
    exec->outDim = param->W1.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;
    exec->bTrain = bTrain;
    return exec;
}


inline PExecute LinearBiNode::generate(bool bTrain) {
    LinearBiExecute* exec = new LinearBiExecute();
    exec->batch.push_back(this);
    exec->inDim1 = param->W1.inDim();
    exec->inDim2 = param->W2.inDim();
    exec->outDim = param->W1.outDim();
    exec->param = param;
    exec->bTrain = bTrain;
    return exec;
}
#endif


#endif /* BIOP_H_ */
