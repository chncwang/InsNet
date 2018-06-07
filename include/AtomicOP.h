#ifndef ATOMICIOP_H_
#define ATOMICIOP_H_

/*
*  AtomicOP.h:
*  a list of atomic operations
*
*  Created on: June 11, 2017
*      Author: yue_zhang(suda), mszhang
*/

/*
ActivateNode
TanhNode
SigmoidNode
ReluNode
IndexNode
PSubNode
PDotNode
*/

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class ActivateNode :public Node {
  public:
    PNode in;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function

  public:
    ActivateNode() : Node() {
        in = NULL;
        activate = ftanh;
        derivate = dtanh;
        node_type = "activate";
    }

    ~ActivateNode() {
        in = NULL;
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

  public:
    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        val.vec() = in->val.vec().unaryExpr(ptr_fun(activate));
    }

    void backward() {
        in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(derivate));
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

#if USE_GPU
class ActivateExecute :public Execute {
  public:
    Tensor1D x, y;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    int sumDim;
    bool bTrain;

  public:
    ~ActivateExecute() {
        sumDim = 0;
        activate = NULL;
        derivate = NULL;
    }

  public:
    inline void  forward() {
        int count = batch.size();

        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
            sumDim += batch[idx]->dim;
        }

        x.init(sumDim, NULL);
        y.init(sumDim, NULL);

        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ActivateNode* ptr = (ActivateNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                x[offset + idy] = ptr->in->val[idy];
            }
            offset += ptr->dim;
        }

        y.vec() = x.vec().unaryExpr(ptr_fun(activate));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ActivateNode* ptr = (ActivateNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->val[idy] = y[offset + idy];
            }
            offset += ptr->dim;
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        Tensor1D lx, ly;
        lx.init(sumDim, NULL);
        ly.init(sumDim, NULL);

        int count = batch.size();
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ActivateNode* ptr = (ActivateNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < ptr->dim; idy++) {
                ly[offset + idy] = ptr->loss[idy];
            }
            offset += ptr->dim;
        }

        lx.vec() = ly.vec() * x.vec().binaryExpr(y.vec(), ptr_fun(derivate));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ActivateNode* ptr = (ActivateNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->in->loss[idy] += lx[offset + idy];
            }
            offset += ptr->dim;
        }
    }
};

inline PExecute ActivateNode::generate(bool bTrain) {
    ActivateExecute* exec = new ActivateExecute();
    exec->batch.push_back(this);
    exec->activate = activate;
    exec->derivate = derivate;
    exec->bTrain = bTrain;
    return exec;
};

#elif USE_BASE
class ActivateExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            ActivateNode* ptr = (ActivateNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            ActivateNode* ptr = (ActivateNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute ActivateNode::generate(bool bTrain) {
    ActivateExecute* exec = new ActivateExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};

#else
class ActivateExecute :public Execute {
  public:
    Tensor1D x, y;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    int sumDim;
    bool bTrain;

  public:
    ~ActivateExecute() {
        sumDim = 0;
        activate = NULL;
        derivate = NULL;
    }

  public:
    inline void  forward() {
        int count = batch.size();

        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
            sumDim += batch[idx]->dim;
        }

        x.init(sumDim);
        y.init(sumDim);

        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ActivateNode* ptr = (ActivateNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                x[offset + idy] = ptr->in->val[idy];
            }
            offset += ptr->dim;
        }

        y.vec() = x.vec().unaryExpr(ptr_fun(activate));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ActivateNode* ptr = (ActivateNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->val[idy] = y[offset + idy];
            }
            offset += ptr->dim;
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        Tensor1D lx, ly;
        lx.init(sumDim);
        ly.init(sumDim);

        int count = batch.size();
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ActivateNode* ptr = (ActivateNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < ptr->dim; idy++) {
                ly[offset + idy] = ptr->loss[idy];
            }
            offset += ptr->dim;
        }

        lx.vec() = ly.vec() * x.vec().binaryExpr(y.vec(), ptr_fun(derivate));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ActivateNode* ptr = (ActivateNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->in->loss[idy] += lx[offset + idy];
            }
            offset += ptr->dim;
        }
    }
};

inline PExecute ActivateNode::generate(bool bTrain) {
    ActivateExecute* exec = new ActivateExecute();
    exec->batch.push_back(this);
    exec->activate = activate;
    exec->derivate = derivate;
    exec->bTrain = bTrain;
    return exec;
};
#endif





class TanhNode :public Node {
  public:
    PNode in;

  public:
    TanhNode() : Node() {
        in = NULL;
        node_type = "tanh";
    }

    ~TanhNode() {
        in = NULL;
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
        val.vec() = in->val.vec().unaryExpr(ptr_fun(ftanh));
    }

    void backward() {
        in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(dtanh));
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

#if USE_GPU
class TanhExecute :public Execute {
  public:
    Tensor1D x, y;
    int sumDim;
    bool bTrain;

  public:
    ~TanhExecute() {
        sumDim = 0;
    }

  public:
    inline void  forward() {
        int count = batch.size();

        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
            sumDim += batch[idx]->dim;
        }

        x.init(sumDim, NULL);
        y.init(sumDim, NULL);

        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                x[offset + idy] = ptr->in->val[idy];
            }
            offset += ptr->dim;
        }

        y.vec() = x.vec().unaryExpr(ptr_fun(ftanh));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->val[idy] = y[offset + idy];
            }
            offset += ptr->dim;
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        Tensor1D lx, ly;
        lx.init(sumDim, NULL);
        ly.init(sumDim, NULL);

        int count = batch.size();
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < ptr->dim; idy++) {
                ly[offset + idy] = ptr->loss[idy];
            }
            offset += ptr->dim;
        }

        lx.vec() = ly.vec() * x.vec().binaryExpr(y.vec(), ptr_fun(dtanh));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->in->loss[idy] += lx[offset + idy];
            }
            offset += ptr->dim;
        }
    }
};

inline PExecute TanhNode::generate(bool bTrain) {
    TanhExecute* exec = new TanhExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};

#elif USE_BASE
class TanhExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute TanhNode::generate(bool bTrain) {
    TanhExecute* exec = new TanhExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};

#else
class TanhExecute :public Execute {
  public:
    Tensor1D x, y;
    int sumDim;
    bool bTrain;

  public:
    ~TanhExecute() {
        sumDim = 0;
    }

  public:
    inline void  forward() {
        int count = batch.size();

        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
            sumDim += batch[idx]->dim;
        }

        x.init(sumDim);
        y.init(sumDim);

        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                x[offset + idy] = ptr->in->val[idy];
            }
            offset += ptr->dim;
        }

        y.vec() = x.vec().unaryExpr(ptr_fun(ftanh));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->val[idy] = y[offset + idy];
            }
            offset += ptr->dim;
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        Tensor1D lx, ly;
        lx.init(sumDim);
        ly.init(sumDim);

        int count = batch.size();
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < ptr->dim; idy++) {
                ly[offset + idy] = ptr->loss[idy];
            }
            offset += ptr->dim;
        }

        lx.vec() = ly.vec() * x.vec().binaryExpr(y.vec(), ptr_fun(dtanh));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->in->loss[idy] += lx[offset + idy];
            }
            offset += ptr->dim;
        }
    }
};

inline PExecute TanhNode::generate(bool bTrain) {
    TanhExecute* exec = new TanhExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};
#endif


class SigmoidNode :public Node {
  public:
    PNode in;

  public:
    SigmoidNode() : Node() {
        in = NULL;
        node_type = "sigmoid";
    }

    ~SigmoidNode() {
        in = NULL;
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
        val.vec() = in->val.vec().unaryExpr(ptr_fun(fsigmoid));
    }

    void backward() {
        in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(dsigmoid));
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

#if USE_GPU
class SigmoidExecute :public Execute {
  public:
    Tensor1D x, y;
    int sumDim;
    bool bTrain;

  public:
    ~SigmoidExecute() {
        sumDim = 0;
    }

  public:
    inline void  forward() {
        int count = batch.size();

        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
            sumDim += batch[idx]->dim;
        }

        x.init(sumDim, NULL);
        y.init(sumDim, NULL);

        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            SigmoidNode* ptr = (SigmoidNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                x[offset + idy] = ptr->in->val[idy];
            }
            offset += ptr->dim;
        }

        y.vec() = x.vec().unaryExpr(ptr_fun(fsigmoid));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            SigmoidNode* ptr = (SigmoidNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->val[idy] = y[offset + idy];
            }
            offset += ptr->dim;
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        Tensor1D lx, ly;
        lx.init(sumDim, NULL);
        ly.init(sumDim, NULL);

        int count = batch.size();
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            SigmoidNode* ptr = (SigmoidNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < ptr->dim; idy++) {
                ly[offset + idy] = ptr->loss[idy];
            }
            offset += ptr->dim;
        }

        lx.vec() = ly.vec() * x.vec().binaryExpr(y.vec(), ptr_fun(dsigmoid));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            SigmoidNode* ptr = (SigmoidNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->in->loss[idy] += lx[offset + idy];
            }
            offset += ptr->dim;
        }
    }
};

inline PExecute SigmoidNode::generate(bool bTrain) {
    SigmoidExecute* exec = new SigmoidExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};

#elif USE_BASE
class SigmoidExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            SigmoidNode* ptr = (SigmoidNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            SigmoidNode* ptr = (SigmoidNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute SigmoidNode::generate(bool bTrain) {
    SigmoidExecute* exec = new SigmoidExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};

#else
class SigmoidExecute :public Execute {
  public:
    Tensor1D x, y;
    int sumDim;
    bool bTrain;

  public:
    ~SigmoidExecute() {
        sumDim = 0;
    }

  public:
    inline void  forward() {
        int count = batch.size();

        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
            sumDim += batch[idx]->dim;
        }

        x.init(sumDim);
        y.init(sumDim);

        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            SigmoidNode* ptr = (SigmoidNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                x[offset + idy] = ptr->in->val[idy];
            }
            offset += ptr->dim;
        }

        y.vec() = x.vec().unaryExpr(ptr_fun(fsigmoid));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            SigmoidNode* ptr = (SigmoidNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->val[idy] = y[offset + idy];
            }
            offset += ptr->dim;
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        Tensor1D lx, ly;
        lx.init(sumDim);
        ly.init(sumDim);

        int count = batch.size();
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            SigmoidNode* ptr = (SigmoidNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < ptr->dim; idy++) {
                ly[offset + idy] = ptr->loss[idy];
            }
            offset += ptr->dim;
        }

        lx.vec() = ly.vec() * x.vec().binaryExpr(y.vec(), ptr_fun(dsigmoid));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            SigmoidNode* ptr = (SigmoidNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->in->loss[idy] += lx[offset + idy];
            }
            offset += ptr->dim;
        }
    }
};

inline PExecute SigmoidNode::generate(bool bTrain) {
    SigmoidExecute* exec = new SigmoidExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};
#endif



class ReluNode :public Node {
  public:
    PNode in;

  public:
    ReluNode() : Node() {
        in = NULL;
        node_type = "relu";
    }

    ~ReluNode() {
        in = NULL;
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
        val.vec() = in->val.vec().unaryExpr(ptr_fun(frelu));
    }

    void backward() {
        in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(drelu));
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

#if USE_GPU
class ReluExecute :public Execute {
  public:
    Tensor1D x, y;
    int sumDim;
    bool bTrain;

  public:
    ~ReluExecute() {
        sumDim = 0;
    }

  public:
    inline void  forward() {
        int count = batch.size();

        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
            sumDim += batch[idx]->dim;
        }

        x.init(sumDim, NULL);
        y.init(sumDim, NULL);

        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ReluNode* ptr = (ReluNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                x[offset + idy] = ptr->in->val[idy];
            }
            offset += ptr->dim;
        }

        y.vec() = x.vec().unaryExpr(ptr_fun(frelu));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ReluNode* ptr = (ReluNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->val[idy] = y[offset + idy];
            }
            offset += ptr->dim;
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        Tensor1D lx, ly;
        lx.init(sumDim, NULL);
        ly.init(sumDim, NULL);

        int count = batch.size();
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ReluNode* ptr = (ReluNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < ptr->dim; idy++) {
                ly[offset + idy] = ptr->loss[idy];
            }
            offset += ptr->dim;
        }

        lx.vec() = ly.vec() * x.vec().binaryExpr(y.vec(), ptr_fun(drelu));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ReluNode* ptr = (ReluNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->in->loss[idy] += lx[offset + idy];
            }
            offset += ptr->dim;
        }
    }
};

inline PExecute ReluNode::generate(bool bTrain) {
    ReluExecute* exec = new ReluExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};

#elif USE_BASE
class ReluExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            ReluNode* ptr = (ReluNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            ReluNode* ptr = (ReluNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute ReluNode::generate(bool bTrain) {
    ReluExecute* exec = new ReluExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};

#else
class ReluExecute :public Execute {
  public:
    Tensor1D x, y;
    int sumDim;
    bool bTrain;

  public:
    ~ReluExecute() {
        sumDim = 0;
    }

  public:
    inline void  forward() {
        int count = batch.size();

        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
            sumDim += batch[idx]->dim;
        }

        x.init(sumDim);
        y.init(sumDim);

        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ReluNode* ptr = (ReluNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                x[offset + idy] = ptr->in->val[idy];
            }
            offset += ptr->dim;
        }

        y.vec() = x.vec().unaryExpr(ptr_fun(frelu));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ReluNode* ptr = (ReluNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->val[idy] = y[offset + idy];
            }
            offset += ptr->dim;
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        Tensor1D lx, ly;
        lx.init(sumDim);
        ly.init(sumDim);

        int count = batch.size();
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ReluNode* ptr = (ReluNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < ptr->dim; idy++) {
                ly[offset + idy] = ptr->loss[idy];
            }
            offset += ptr->dim;
        }

        lx.vec() = ly.vec() * x.vec().binaryExpr(y.vec(), ptr_fun(drelu));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            ReluNode* ptr = (ReluNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->in->loss[idy] += lx[offset + idy];
            }
            offset += ptr->dim;
        }
    }
};

inline PExecute ReluNode::generate(bool bTrain) {
    ReluExecute* exec = new ReluExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};
#endif


class IndexNode :public Node {
  public:
    PNode in;
    int index_id;

  public:
    IndexNode() : Node() {
        in = NULL;
        index_id = -1;
        dim = 1;
        node_type = "index";
    }

    ~IndexNode() {
        in = NULL;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
        index_id = -1;
    }

    //can not be dropped since the output is a scalar
    inline void init(int ndim, dtype dropout) {
        dim = 1;
        Node::init(dim, -1);
    }

  public:
    void forward(Graph *cg, PNode x, int index) {
        in = x;
        index_id = index;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    void compute() {
        val[0] = in->val[index_id];
    }

    void backward() {
        in->loss[index_id] += loss[0];
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

class IndexExecute : public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            IndexNode* ptr = (IndexNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            IndexNode* ptr = (IndexNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute IndexNode::generate(bool bTrain) {
    IndexExecute* exec = new IndexExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}



class PSubNode : public Node {
  public:
    PNode in1, in2;
  public:
    PSubNode() : Node() {
        in1 = NULL;
        in2 = NULL;
        node_type = "point-subtraction";
    }
  public:
    virtual inline void clearValue() {
        Node::clearValue();
        in1 = NULL;
        in2 = NULL;
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
        val.vec() = in1->val.vec() - in2->val.vec();
    }

    void backward() {
        in1->loss.vec() += loss.vec();
        in2->loss.vec() -= loss.vec();
    }

  public:
    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

    inline PExecute generate(bool bTrain);
};


class PSubExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            PSubNode* ptr = (PSubNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            PSubNode* ptr = (PSubNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute PSubNode::generate(bool bTrain) {
    PSubExecute* exec = new PSubExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}


class PDotNode : public Node {
  public:
    PNode in1, in2;
  public:
    PDotNode() : Node() {
        in1 = NULL;
        in2 = NULL;
        dim = 1;
        node_type = "point-dot";
    }
  public:
    virtual inline void clearValue() {
        Node::clearValue();
        in1 = NULL;
        in2 = NULL;
    }

    //can not be dropped since the output is a scalar
    inline void init(int ndim, dtype dropout) {
        dim = 1;
        Node::init(dim, -1);
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
        val[0] = 0.0;
        for (int idx = 0; idx < in1->dim; idx++) {
            val[0] += in1->val[idx] * in2->val[idx];
        }
    }

    void backward() {
        for (int idx = 0; idx < in1->dim; idx++) {
            in1->loss[idx] += loss[0] * in2->val[idx];
            in2->loss[idx] += loss[0] * in1->val[idx];
        }
    }

  public:
    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

    inline PExecute generate(bool bTrain);
};

class PDotExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            PDotNode* ptr = (PDotNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            PDotNode* ptr = (PDotNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute PDotNode::generate(bool bTrain) {
    PDotExecute* exec = new PDotExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}

#endif
