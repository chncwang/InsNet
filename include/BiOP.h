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

    void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W1);
        ada.addParam(&W2);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    void initial(int nOSize, int nISize1, int nISize2, bool useB = true) {
        W1.initial(nOSize, nISize1);
        W2.initial(nOSize, nISize2);
        bUseB = useB;
        if (bUseB) {
            b.initial(nOSize, 1);
        }
    }

    void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        W1.save(os);
        W2.save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    void load(std::ifstream &is) {
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
    Tensor1D ty, lty;


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

    void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
        ty.init(ndim);
        lty.init(ndim);
    }

    void setParam(BiParams* paramInit) {
        param = paramInit;
    }

    // define the activate function and its derivation form
    void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
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
    void compute() {
        ty.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat();
        if (param->bUseB) {
            ty.vec() += param->b.val.vec();
        }
        val.vec() = ty.vec().unaryExpr(ptr_fun(activate));
    }

    void backward() {
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
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) override {
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

    size_t typeHashCode() const override {
        void *act = reinterpret_cast<void*>(activate);
        void *de = reinterpret_cast<void*>(derivate);
        return Node::typeHashCode() ^ ::typeHashCode(param) ^
            ::typeHashCode(act) ^ (::typeHashCode(de) << 1);
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

    void setParam(BiParams* paramInit) {
        param = paramInit;
    }

    void forward(Graph *cg, PNode x1, PNode x2) {
        in1 = x1;
        in2 = x2;
        degree = 0;
        in1->addParent(this);
        in2->addParent(this);
        cg->addNode(this);
    }

  public:
    void compute() {
        val.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat();

        if (param->bUseB) {
            val.vec() += param->b.val.vec();
        }
    }

    void backward() {
        param->W1.grad.mat() += loss.mat() * in1->val.tmat();
        param->W2.grad.mat() += loss.mat() * in2->val.tmat();

        if (param->bUseB) {
            param->b.grad.vec() += loss.vec();
        }

        in1->loss.mat() += param->W1.val.mat().transpose() * loss.mat();
        in2->loss.mat() += param->W2.val.mat().transpose() * loss.mat();
    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        LinearBiNode* conv_other = (LinearBiNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};


class BiExecute :public Execute {
  public:
    Tensor2D x1, x2, ty, y, b;
    Tensor2D drop_mask;
    int inDim1, inDim2, outDim;
    BiParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
#if USE_GPU
    void forward() {
        int count = batch.size();
        ty.init(outDim, count);
        x1.init(inDim1, count);
        x2.init(inDim2, count);
        y.init(outDim, count);
        drop_mask.init(outDim, count);
#if TEST_CUDA
        b.init(outDim, count);
#endif
        std::vector<dtype*> x1s, x2s, ys;
        x1s.reserve(count);
        x2s.reserve(count);
        ys.reserve(count);

        for (int i = 0; i < batch.size(); ++i) {
            BiNode *n = static_cast<BiNode*>(batch.at(i));
            x1s.push_back(n->in1->val.value);
            x2s.push_back(n->in2->val.value);
            ys.push_back(n->val.value);
        }

        n3ldg_cuda::CopyForBiNodeForward(x1s, x2s, param->b.val.value,
                x1.value, x2.value, ty.value, count, inDim1, inDim2, outDim);
#if TEST_CUDA
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
        n3ldg_cuda::Assert(x1.verify("BiExecute forward x1"));
        n3ldg_cuda::Assert(x2.verify("BiExecute forward x2"));
#endif
        n3ldg_cuda::MatrixMultiplyMatrix(param->W1.val.value, x1.value,
                ty.value, outDim, inDim1, count, param->bUseB);
        n3ldg_cuda::MatrixMultiplyMatrix(param->W2.val.value, x2.value,
                ty.value, outDim, inDim2, count, true);
        if (bTrain) {
            n3ldg_cuda::CalculateDropoutMask(dynamicDropValue(), count, outDim,
                    drop_mask.value);
        }
        n3ldg_cuda::ActivatedEnum activatedEnum = ToActivatedEnum(activate);
        n3ldg_cuda::Activated(activatedEnum, ty.value, ys, y.value, outDim,
                bTrain, dynamicDropValue(), drop_mask.value);
#if TEST_CUDA
        ty.mat() = param->W1.val.mat() * x1.mat() + param->W2.val.mat() * x2.mat();

        if (param->bUseB) {
            ty.vec() = ty.vec() + b.vec();
        }
        n3ldg_cuda::Assert(ty.verify("BiExecute forward ty"));

        y.vec() = ty.vec().unaryExpr(ptr_fun(activate));
        n3ldg_cuda::Assert(y.verify("BiExecute forward y"));

        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
        }

        drop_mask.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < outDim; ++j) {
                dtype v = drop_mask[i][j];
                batch[i]->drop_mask[j] = v <= dynamicDropValue() ? 0 : 1;
            }
        }

        for (int i = 0; i < count; ++i) {
            dtype drop_value = batch[0]->drop_value;
            batch[i]->forward_drop(bTrain, drop_factor);
            n3ldg_cuda::Assert(batch[i]->val.verify(
                        "BiExecute forward batch i val"));
        }
#endif
    }

#else
    void  forward() {
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
            ptr->forward_drop(bTrain, drop_factor);
        }
    }
#endif

#if USE_GPU
    void backward() {
        int count = batch.size();
        Tensor2D lx1, lx2, lty, ly;
        lx1.init(inDim1, count);
        lx2.init(inDim2, count);
        lty.init(outDim, count);
        ly.init(outDim, count);

        std::vector<dtype*> ly_vec;
        ly_vec.reserve(count);
        for (int i = 0; i < count; ++i) {
            BiNode* ptr = (BiNode*)batch[i];
            ly_vec.push_back(ptr->loss.value);
        }
        n3ldg_cuda::ActivatedEnum activated = ToActivatedEnum(activate);
        n3ldg_cuda::CalculateLtyForUniBackward(activated, ly_vec, ty.value,
                y.value, drop_mask.value, dynamicDropValue(), lty.value, count,
                outDim);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->loss[idy];
            }
        }

        n3ldg_cuda::Assert(ty.verify("BiExecute backward ty"));
        n3ldg_cuda::Assert(y.verify("BiExecute backward y"));
        lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));
        n3ldg_cuda::Assert(lty.verify("BiExecute backward lty"));
#endif
#if TEST_CUDA
        n3ldg_cuda::Assert(param->W1.grad.verify("bi backward W grad initial"));
        n3ldg_cuda::Assert(param->W2.grad.verify("bi backward W grad initial"));
#endif
        n3ldg_cuda::MatrixMultiplyMatrix(lty.value, x1.value,
                param->W1.grad.value, outDim, count, inDim1, true, true, false);
        n3ldg_cuda::MatrixMultiplyMatrix(lty.value, x2.value,
                param->W2.grad.value, outDim, count, inDim2, true, true, false);
#if TEST_CUDA
        n3ldg_cuda::Assert(param->W1.val.verify("bi W1.val initial"));
        n3ldg_cuda::Assert(param->W2.val.verify("bi W2.val initial"));
#endif
        n3ldg_cuda::MatrixMultiplyMatrix(param->W1.val.value, lty.value,
                lx1.value, inDim1, outDim, count, false, false, true);
        n3ldg_cuda::MatrixMultiplyMatrix(param->W2.val.value, lty.value,
                lx2.value, inDim2, outDim, count, false, false, true);
        std::vector<dtype*> losses1, losses2;
        losses1.reserve(count);
        losses2.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
#if TEST_CUDA
            n3ldg_cuda::Assert(ptr->in1->loss.verify("bi backward in loss"));
            n3ldg_cuda::Assert(ptr->in2->loss.verify("bi backward in loss"));
#endif
            losses1.push_back(ptr->in1->loss.value);
            losses2.push_back(ptr->in2->loss.value);
        }
#if TEST_CUDA
        n3ldg_cuda::Assert(param->b.grad.verify(
                    "bi backward param b initial"));
#endif
        n3ldg_cuda::AddLtyToParamBiasAndAddLxToInputLossesForBiBackward(
                lty.value, lx1.value, lx2.value, param->b.grad.value,
                losses1, losses2, count, outDim, inDim1, inDim2);
#if TEST_CUDA
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
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            n3ldg_cuda::Assert(ptr->in1->loss.verify("bi in1 loss"));
            n3ldg_cuda::Assert(ptr->in2->loss.verify("bi in2 loss"));
        }
#endif
    }
#else
    void backward() {
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
#endif
};

PExecute BiNode::generate(bool bTrain, dtype cur_drop_factor) {
    BiExecute* exec = new BiExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    exec->inDim1 = param->W1.inDim();
    exec->inDim2 = param->W2.inDim();
    exec->outDim = param->W1.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;
    return exec;
};

class LinearBiExecute :public Execute {
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

PExecute LinearBiNode::generate(bool bTrain, dtype cur_drop_factor) {
    LinearBiExecute* exec = new LinearBiExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};

#endif /* BIOP_H_ */
