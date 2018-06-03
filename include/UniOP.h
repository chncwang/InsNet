#ifndef UNIOP_H_
#define UNIOP_H_

/*
*  UniOP.h:
*  a simple feed forward neural operation, unary input.
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"
#include <cstdlib>
#include "profiler.h"

class UniParams {
  public:
    Param W;
    Param b;
    bool bUseB;

  public:
    UniParams() {
        bUseB = true;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    inline void initial(int nOSize, int nISize, bool useB = true) {
        W.initial(nOSize, nISize);

        bUseB = useB;
        if (bUseB) {
            b.initial(nOSize, 1);
        }
    }

    inline void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        W.save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    inline void load(std::ifstream &is) {
        is >> bUseB;
        W.load(is);
        if (bUseB) {
            b.load(is);
        }
    }

};


// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class UniNode : public Node {
  public:
    PNode in;
    UniParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    Tensor1D ty, lty;


  public:
    UniNode() : Node() {
        in = NULL;
        activate = ftanh;
        derivate = dtanh;
        param = NULL;
        node_type = "uni";
    }

    ~UniNode() {
        in = NULL;
    }

    inline void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
        ty.init(ndim);
        lty.init(ndim);
    }


    inline void setParam(UniParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
    }

    // define the activate function and its derivation form
    inline void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
        activate = f;
        derivate = f_deri;
    }

    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

    inline void compute() {
        ty.mat() = param->W.val.mat() * in->val.mat();
        if (param->bUseB) {
            ty.vec() += param->b.val.vec();
        }
        val.vec() = ty.vec().unaryExpr(ptr_fun(activate));
    }

    inline void backward() {
        lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));
        param->W.grad.mat() += lty.mat() * in->val.tmat();
        if (param->bUseB) {
            param->b.grad.vec() += lty.vec();
        }
        in->loss.mat() += param->W.val.mat().transpose() * lty.mat();
    }

    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) override {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        UniNode* conv_other = (UniNode*)other;
        if (param != conv_other->param) {
            return false;
        }
        if (activate != conv_other->activate || derivate != conv_other->derivate) {
            return false;
        }

        return true;
    }

//    size_t typeHashCode() const override {
//        void *act = reinterpret_cast<void*>(activate);
//        void *de = reinterpret_cast<void*>(derivate);
//        return Node::typeHashCode() ^ ::typeHashCode(param) ^ ::typeHashCode(act) ^
//            (::typeHashCode(de) << 1);
//    }
};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class LinearUniNode : public Node {
  public:
    PNode in;
    UniParams* param;

  public:
    LinearUniNode() : Node() {
        in = NULL;
        param = NULL;
        node_type = "linear_uni";
    }


    inline void setParam(UniParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
    }


  public:
    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        val.mat() = param->W.val.mat() * in->val.mat();
        if (param->bUseB) {
            val.vec() += param->b.val.vec();
        }
    }

    inline void backward() {
        param->W.grad.mat() += loss.mat() * in->val.tmat();
        if (param->bUseB) {
            param->b.grad.vec() += loss.vec();
        }
        in->loss.mat() += param->W.val.mat().transpose() * loss.mat();
    }

  public:
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        LinearUniNode* conv_other = (LinearUniNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};



// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class LinearNode : public Node {
  public:
    PNode in;
    UniParams* param;

  public:
    LinearNode() : Node() {
        in = NULL;
        param = NULL;
        node_type = "linear";
    }


    inline void setParam(UniParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
    }


  public:
    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        val.mat() = param->W.val.mat() * in->val.mat();
    }

    inline void backward() {
        param->W.grad.mat() += loss.mat() * in->val.tmat();
        in->loss.mat() += param->W.val.mat().transpose() * loss.mat();
    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) override {
        bool result = Node::typeEqual(other);
        if (!result) return false;
        LinearNode* conv_other = (LinearNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

//    size_t typeHashCode() const override {
//        return Node::typeHashCode() ^ ::typeHashCode(param);
//    }
};


class UniExecute :public Execute {
  public:
      bool bTrain;
    Tensor2D x, ty, y, b;
    int inDim, outDim;
    UniParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    Tensor2D drop_mask;

    inline void  forward() {
        int count = batch.size();
        ty.init(outDim, count);
        x.init(inDim, count);
        y.init(outDim, count);
        drop_mask.init(outDim, count);
#if TEST_CUDA || !USE_GPU
        b.init(outDim, count);
#endif

#if USE_GPU
        std::vector<dtype*> xs, ys;
        xs.reserve(batch.size());
        ys.reserve(batch.size());


        for (int i = 0; i < batch.size(); ++i) {
            UniNode *n = static_cast<UniNode*>(batch.at(i));

            xs.push_back(n->in->val.value);
            ys.push_back(n->val.value);
        }

        n3ldg_cuda::CopyForUniNodeForward(xs, param->b.val.value,
                x.value, ty.value, count, inDim, outDim, param->bUseB);

        n3ldg_cuda::MatrixMultiplyMatrix(param->W.val.value, x.value,
                ty.value, outDim, inDim, count, param->bUseB);

        CalculateDropMask(count, outDim, drop_mask);

        n3ldg_cuda::ActivatedEnum activatedEnum = ToActivatedEnum(activate);
        n3ldg_cuda::Activated(activatedEnum, ty.value, ys, y.value, outDim,
                bTrain, dynamicDropValue(), drop_mask.value);

        for (int i = 0; i<batch.size(); ++i) {
            UniNode *n = static_cast<UniNode*>(batch.at(i));
        }

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                x[idy][idx] = ptr->in->val[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idy][idx] = param->b.val.v[idy];
                }
            }
        }
        n3ldg_cuda::Assert(x.verify("forward x"));

        ty.mat() = param->W.val.mat() * x.mat();

        if (param->bUseB) {
            ty.vec() = ty.vec() + b.vec();
        }

        y.vec() = ty.vec().unaryExpr(ptr_fun(activate));

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idy][idx];
            }
        }

        drop_mask.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < outDim; ++j) {
                dtype v = drop_mask[j][i];
                batch[i]->drop_mask[j] = v <= dynamicDropValue() ? 0 : 1;
            }
        }

        for (int i = 0; i < count; ++i) {
            batch[i]->forward_drop(bTrain, drop_factor);
            n3ldg_cuda::Assert(batch[i]->val.verify("forward batch i val"));
        }

        n3ldg_cuda::Assert(ty.verify("forward ty"));
        n3ldg_cuda::Assert(y.verify("forward y"));
#endif
#else
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                x[idy][idx] = ptr->in->val[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idy][idx] = param->b.val.v[idy];
                }
            }
        }

        ty.mat() = param->W.val.mat() * x.mat();

        if (param->bUseB) {
            ty.vec() = ty.vec() + b.vec();
        }

        y.vec() = ty.vec().unaryExpr(ptr_fun(activate));

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idy][idx];
            }
        }

        for (int i = 0; i < count; ++i) {
            dtype drop_value = batch[0]->drop_value;
            batch[i]->forward_drop(bTrain, drop_factor);
        }
#endif
    }

    void backward() {
        int count = batch.size();
        Tensor2D lx, lty, ly;
#if USE_GPU
        lx.init(inDim, count);
        lty.init(outDim, count);
        ly.init(outDim, count);

        std::vector<dtype*> ly_vec;
        ly_vec.reserve(count);
        for (int i = 0; i < count; ++i) {
            UniNode* ptr = (UniNode*)batch[i];
            ly_vec.push_back(ptr->loss.value);
        }
        n3ldg_cuda::ActivatedEnum activatedEnum = ToActivatedEnum(activate);
        n3ldg_cuda::CalculateLtyForUniBackward(activatedEnum, ly_vec, ty.value,
                y.value, drop_mask.value, dynamicDropValue(), lty.value, count,
                outDim);
#if TEST_CUDA
        n3ldg_cuda::Assert(param->W.grad.verify(
                    "uni backward W grad initial"));
#endif
        n3ldg_cuda::MatrixMultiplyMatrix(lty.value, x.value,
                param->W.grad.value, outDim, count, inDim, true, true, false);
#if TEST_CUDA
        n3ldg_cuda::Assert(param->W.val.verify("uni W.val initial"));
#endif
        n3ldg_cuda::MatrixMultiplyMatrix(param->W.val.value, lty.value,
                lx.value, inDim, outDim, count, false, false, true);
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            losses.push_back(ptr->in->loss.value);
        }

#if TEST_CUDA
        n3ldg_cuda::Assert(
                param->b.grad.verify("uni backward param b initial"));
#endif
        n3ldg_cuda::AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(
                lty.value, lx.value, param->b.grad.value, losses, count,
                outDim, inDim, param->bUseB);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idy][idx] = ptr->loss[idy];
            }
        }

        n3ldg_cuda::Assert(x.verify("backward x"));
        lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));
        n3ldg_cuda::Assert(lty.verify("backward lty"));

        param->W.grad.mat() += lty.mat() * x.mat().transpose();
        n3ldg_cuda::Assert(param->W.grad.verify("backward W grad"));

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += lty[idy][idx];
                }
            }
        }
        n3ldg_cuda::Assert(param->b.grad.verify("backward b grad"));

        lx.mat() += param->W.val.mat().transpose() * lty.mat();
        n3ldg_cuda::Assert(lx.verify("backward lx"));

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss[idy] += lx[idy][idx];
            }
        }

        for (Node * n : batch) {
            UniNode *ptr = static_cast<UniNode *>(n);
            n3ldg_cuda::Assert(ptr->in->loss.verify("uni backward loss"));
        }
#endif
#else
        lx.init(inDim, count);
        lty.init(outDim, count);
        ly.init(outDim, count);
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idy][idx] = ptr->loss[idy];
            }
        }

        lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));
        param->W.grad.mat() += lty.mat() * x.mat().transpose();

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += lty[idy][idx];
                }
            }
        }

        lx.mat() += param->W.val.mat().transpose() * lty.mat();

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss[idy] += lx[idy][idx];
            }
        }
#endif
    }
};

inline PExecute UniNode::generate(bool bTrain, dtype cur_drop_factor) {
    UniExecute* exec = new UniExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    exec->inDim = param->W.inDim();
    exec->outDim = param->W.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;
    return exec;
};

class LinearUniExecute :public Execute {
  public:
      bool bTrain;
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

inline PExecute LinearUniNode::generate(bool bTrain, dtype cur_drop_factor) {
    LinearUniExecute* exec = new LinearUniExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};

#if USE_GPU
class LinearExecute :public Execute {
public:
    bool bTrain;
    Tensor2D x, y, b;
    int inDim, outDim, count;
    UniParams* param;

    void  forward() {
        int count = batch.size();

        x.init(inDim, count);
        y.init(outDim, count);
#if TEST_CUDA
        b.init(outDim, count);
#endif
        std::vector<dtype*> xs, ys;
        xs.reserve(batch.size());
        ys.reserve(batch.size());

        for (int i = 0; i < batch.size(); ++i) {
            LinearNode *n = static_cast<LinearNode*>(batch.at(i));

            xs.push_back(n->in->val.value);
            ys.push_back(n->val.value);
        }

        n3ldg_cuda::CopyForUniNodeForward(xs, param->b.val.value,
                x.value, y.value, count, inDim, outDim, param->bUseB);

        n3ldg_cuda::MatrixMultiplyMatrix(param->W.val.value, x.value, y.value,
                outDim, inDim, count, false);

        std::vector<dtype*> vals;
        vals.reserve(count);
        for (Node *node : batch) {
            vals.push_back(node->val.value);
        }

        n3ldg_cuda::CopyFromOneVectorToMultiVals(y.value, vals, count, outDim);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                x[idy][idx] = ptr->in->val[idy];
            }
        }

        y.mat() = param->W.val.mat() * x.mat();
        n3ldg_cuda::Assert(x.verify("forward x"));
        n3ldg_cuda::Assert(y.verify("forward y"));

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idy][idx];
            }
            n3ldg_cuda::Assert(ptr->val.verify("linear forward val"));
        }
#endif

    }

    void backward() {
        int count = batch.size();
        Tensor2D lx, ly;
        lx.init(inDim, count);
        ly.init(outDim, count);

        std::vector<dtype*> ly_vec;
        ly_vec.reserve(count);
        for (int i = 0; i < count; ++i) {
            UniNode* ptr = (UniNode*)batch[i];
            ly_vec.push_back(ptr->loss.value);
        }
        n3ldg_cuda::CalculateLyForLinearBackward(ly_vec, ly.value, count,
                outDim);
        n3ldg_cuda::MatrixMultiplyMatrix(ly.value, x.value,
                param->W.grad.value, outDim, count, inDim, true, true, false);
        n3ldg_cuda::MatrixMultiplyMatrix(param->W.val.value, ly.value,
                lx.value, inDim, outDim, count, false, false, true);
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            losses.push_back(ptr->in->loss.value);
        }

        n3ldg_cuda::AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(
                ly.value, lx.value, param->b.grad.value, losses, count,
                outDim, inDim, param->bUseB);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idy][idx] = ptr->loss[idy];
            }
        }

        assert(x.verify("backward x"));

        param->W.grad.mat() += ly.mat() * x.mat().transpose();
        param->W.grad.verify("backward W grad");

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += ly[idy][idx];
                }
            }
        }
        n3ldg_cuda::Assert(param->b.grad.verify("backward b grad"));

        lx.mat() += param->W.val.mat().transpose() * ly.mat();
        lx.verify("backward lx");

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss[idy] += lx[idy][idx];
            }
        }

        for (Node * n : batch) {
            UniNode *ptr = static_cast<UniNode *>(n);
            n3ldg_cuda::Assert(ptr->in->loss.verify("backward loss"));
        }
#endif
    }
};
#else
class LinearExecute :public Execute {
  public:
    Tensor2D x, y;
    int inDim, outDim, count;
    UniParams* param;
    bool bTrain;
    inline void  forward() {
        count = batch.size();
        x.init(inDim, count);
        y.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                x[idy][idx] = ptr->in->val[idy];
            }
        }

        y.mat() = param->W.val.mat() * x.mat();

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idy][idx];
            }
            ptr->forward_drop(bTrain, drop_factor);
        }
    }

    inline void backward() {
        Tensor2D lx, ly;
        lx.init(inDim, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idy][idx] = ptr->loss[idy];
            }
        }

        param->W.grad.mat() += ly.mat() * x.mat().transpose();

        lx.mat() += param->W.val.mat().transpose() * ly.mat();

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss[idy] += lx[idy][idx];
            }
        }
    }
};
#endif

inline PExecute LinearNode::generate(bool bTrain, dtype cur_drop_factor) {
    LinearExecute* exec = new LinearExecute();
    exec->batch.push_back(this);
    exec->inDim = param->W.inDim();
    exec->outDim = param->W.outDim();
    exec->param = param;
    exec->bTrain = bTrain;
    return exec;
};


#endif /* UNIOP_H_ */
