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

    // define the activate function and its derivation form
    void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
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
    void compute() {
        val.vec() = in->val.vec().unaryExpr(ptr_fun(activate));
    }

    void backward() {
        in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(derivate));
    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};


class ActivateExecute :public Execute {
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

PExecute ActivateNode::generate(bool bTrain, dtype cur_drop_factor) {
    ActivateExecute* exec = new ActivateExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};

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

  public:
    void forward(Graph &graph, Node &input) {
        this->forward(&graph, &input);
    }

    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    void compute() {
        val.vec() = in->val.vec().unaryExpr(ptr_fun(ftanh));
    }

    void backward() {
        in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(dtanh));
    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

class TanhExecute :public Execute {
  public:
    Tensor2D drop_mask;
    int dim;
public:
    Tensor1D y, x;
    int sumDim;

#if USE_GPU
    void forward() {
        int count = batch.size();
        std::vector<dtype*> xs, ys;
        xs.reserve(count);
        ys.reserve(count);
        drop_mask.init(dim, count);
        for (Node *n : batch) {
            TanhNode *tanh = static_cast<TanhNode*>(n);
#if TEST_CUDA
            tanh->in->val.copyFromHostToDevice();
#endif
            xs.push_back(tanh->in->val.value);
            ys.push_back(tanh->val.value);
        }

        CalculateDropMask(count, dim, drop_mask);
        n3ldg_cuda::TanhForward(n3ldg_cuda::ActivatedEnum::TANH, xs, count, dim, drop_mask.value,
                this->dynamicDropValue(), ys);
#if TEST_CUDA
        drop_mask.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < dim; ++j) {
                dtype v = drop_mask[i][j];
                batch[i]->drop_mask[j] = v <= dynamicDropValue() ? 0 : 1;
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
            n3ldg_cuda::Assert(batch.at(idx)->val.verify("Tanh forward"));
        }
#endif
    }
#else
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for  
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
            ptr->forward_drop(bTrain,drop_factor);
        }
    }
#endif

#if USE_GPU
    void backward() {
        int count = batch.size();
        std::vector<dtype*> vals, losses, in_losses;
        vals.reserve(count);
        losses.reserve(count);
        in_losses.reserve(count);
        for (Node *n : batch) {
            TanhNode *tanh = static_cast<TanhNode*>(n);
#if TEST_CUDA
            tanh->loss.copyFromHostToDevice();
            tanh->in->loss.copyFromHostToDevice();
#endif
            vals.push_back(tanh->val.value);
            losses.push_back(tanh->loss.value);
            in_losses.push_back(tanh->in->loss.value);
        }
        n3ldg_cuda::TanhBackward(n3ldg_cuda::ActivatedEnum::TANH, losses, vals, count, dim, drop_mask.value,
                dynamicDropValue(), in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward_drop();
            n->backward();
        }
        for (Node *n : batch) {
            TanhNode *tanh = static_cast<TanhNode*>(n);
            n3ldg_cuda::Assert(tanh->in->loss.verify("TanhExecute backward"));
        }
#endif
    }
#else
    void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        Tensor1D lx, ly;
        lx.init(sumDim);
        ly.init(sumDim);

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
#endif
};

PExecute TanhNode::generate(bool bTrain, dtype cur_drop_factor) {
    TanhExecute* exec = new TanhExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    exec->dim = dim;
    return exec;
};


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

    void forward(Graph &graph, Node &input) {
        this->forward(&graph, &input);
    }

    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    void compute() {
        val.vec() = in->val.vec().unaryExpr(ptr_fun(fsigmoid));
    }

    void backward() {
        in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(dsigmoid));
    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};


class SigmoidExecute :public Execute {
  public:
    Tensor2D drop_mask;
    int dim;
public:
    Tensor1D x, y;
    int sumDim;

#if USE_GPU
    void forward() {
        int count = batch.size();
        std::vector<dtype*> xs, ys;
        xs.reserve(count);
        ys.reserve(count);
        drop_mask.init(dim, count);
        for (Node *n : batch) {
            SigmoidNode *tanh = static_cast<SigmoidNode*>(n);
#if TEST_CUDA
            tanh->in->val.copyFromHostToDevice();
#endif
            xs.push_back(tanh->in->val.value);
            ys.push_back(tanh->val.value);
        }

        CalculateDropMask(count, dim, drop_mask);
        n3ldg_cuda::TanhForward(n3ldg_cuda::ActivatedEnum::SIGMOID, xs, count, dim, drop_mask.value,
                this->dynamicDropValue(), ys);
#if TEST_CUDA
        drop_mask.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < dim; ++j) {
                dtype v = drop_mask[i][j];
                batch[i]->drop_mask[j] = v <= dynamicDropValue() ? 0 : 1;
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
            n3ldg_cuda::Assert(batch.at(idx)->val.verify("Sigmoid forward"));
        }
#endif
    }
#else
#endif

#if USE_GPU
    void backward() {
        int count = batch.size();
        std::vector<dtype*> vals, losses, in_losses;
        vals.reserve(count);
        losses.reserve(count);
        in_losses.reserve(count);
        for (Node *n : batch) {
            SigmoidNode *tanh = static_cast<SigmoidNode*>(n);
#if TEST_CUDA
            tanh->loss.copyFromHostToDevice();
            tanh->in->loss.copyFromHostToDevice();
#endif
            vals.push_back(tanh->val.value);
            losses.push_back(tanh->loss.value);
            in_losses.push_back(tanh->in->loss.value);
        }
        n3ldg_cuda::TanhBackward(n3ldg_cuda::ActivatedEnum::SIGMOID, losses, vals, count, dim, drop_mask.value,
                dynamicDropValue(), in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward_drop();
            n->backward();
        }
        for (Node *n : batch) {
            SigmoidNode *tanh = static_cast<SigmoidNode*>(n);
            n3ldg_cuda::Assert(tanh->in->loss.verify("SigmoidExecute backward"));
        }
#endif
    }
#else
#endif
};

PExecute SigmoidNode::generate(bool bTrain, dtype cur_drop_factor) {
    SigmoidExecute* exec = new SigmoidExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    exec->dim = dim;
    return exec;
};


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

    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    void compute() {
        val.vec() = in->val.vec().unaryExpr(ptr_fun(frelu));
    }

    void backward() {
        in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(drelu));
    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

class ReluExecute :public Execute {
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

PExecute ReluNode::generate(bool bTrain, dtype cur_drop_factor) {
    ReluExecute* exec = new ReluExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};


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

    //can not be dropped since the output is a scalar
    void init(int ndim, dtype dropout) {
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
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

class IndexExecute : public Execute {
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

PExecute IndexNode::generate(bool bTrain, dtype cur_drop_factor) {
    IndexExecute* exec = new IndexExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
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
        val.vec() = in1->val.vec() - in2->val.vec();
    }

    void backward() {
        in1->loss.vec() += loss.vec();
        in2->loss.vec() -= loss.vec();
    }

  public:
    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

    PExecute generate(bool bTrain, dtype cur_drop_factor);
};


class PSubExecute :public Execute {
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

PExecute PSubNode::generate(bool bTrain, dtype cur_drop_factor) {
    PSubExecute* exec = new PSubExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
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

    //can not be dropped since the output is a scalar
    void init(int ndim, dtype dropout) {
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
    void compute() {
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
    bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

    PExecute generate(bool bTrain, dtype cur_drop_factor);
};

#if USE_GPU
class PDotExecute :public Execute {
public:
    void  forward() {
        int count = batch.size();
        std::vector<dtype*> vals;
        ins1.reserve(count);
        ins2.reserve(count);
        vals.reserve(count);
        for (Node *node : batch) {
            PDotNode *dot = static_cast<PDotNode*>(node);
            ins1.push_back(dot->in1->val.value);
            ins2.push_back(dot->in2->val.value);
            vals.push_back(dot->val.value);
        }

        n3ldg_cuda::PDotForward(ins1, ins2, count,
                static_cast<PDotNode*>(batch.at(0))->in1->dim, vals);
#if TEST_CUDA
        for (Node *node : batch) {
            PDotNode *dot = static_cast<PDotNode*>(node);
            n3ldg_cuda::Assert(dot->in1->val.verify("PDot in1"));
            n3ldg_cuda::Assert(dot->in2->val.verify("PDot in2"));
            node->compute();
            n3ldg_cuda::Assert(node->val.verify("PDot forward"));
        }
#endif
    }

    void backward() {
        int count = batch.size();
        std::vector<dtype*> losses, in_losses1, in_losses2;
        losses.reserve(count);
        in_losses1.reserve(count);
        in_losses2.reserve(count);
        for (Node *node : batch) {
            PDotNode *dot = static_cast<PDotNode*>(node);
            losses.push_back(dot->loss.value);
            in_losses1.push_back(dot->in1->loss.value);
            in_losses2.push_back(dot->in2->loss.value);
        }
        n3ldg_cuda::PDotBackward(losses, ins1, ins2, count,
                static_cast<PDotNode*>(batch.at(0))->in1->dim, in_losses1,
                in_losses2);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
            n3ldg_cuda::Assert(batch[idx]->loss.verify(
                        "PDotExecute backward"));
        }

        for (Node *node : batch) {
            PDotNode *dot = static_cast<PDotNode*>(node);
            n3ldg_cuda::Assert(dot->in1->loss.verify(
                        "PDotExecute backward in1"));
            n3ldg_cuda::Assert(dot->in2->loss.verify(
                        "PDotExecute backward in2"));
        }
#endif
    }

private:
    std::vector<dtype*> ins1;
    std::vector<dtype*> ins2;
};
#else
class PDotExecute :public Execute {
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
#endif


PExecute PDotNode::generate(bool bTrain, dtype cur_drop_factor) {
    PDotExecute* exec = new PDotExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
}

class DropoutNode : public Node {
public:
    PNode in = NULL;

    DropoutNode() {
        node_type = "dropout";
    }

    PExecute generate(bool bTrain, dtype cur_drop_factor);
};

class DropoutExecute :public Execute {
  public:
    Tensor2D drop_mask;
    int dim;

#if USE_GPU
    void forward() {
        int count = batch.size();
        std::vector<dtype*> xs, ys;
        xs.reserve(count);
        ys.reserve(count);
        drop_mask.init(dim, count);
        for (Node *n : batch) {
            DropoutNode *tanh = static_cast<DropoutNode*>(n);
#if TEST_CUDA
            tanh->in->val.copyFromHostToDevice();
#endif
            xs.push_back(tanh->in->val.value);
            ys.push_back(tanh->val.value);
        }

        CalculateDropMask(count, dim, drop_mask);
        n3ldg_cuda::DropoutForward(xs, count, dim, drop_mask.value,
                this->dynamicDropValue(), ys);
#if TEST_CUDA
        drop_mask.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < dim; ++j) {
                dtype v = drop_mask[i][j];
                batch[i]->drop_mask[j] = v <= dynamicDropValue() ? 0 : 1;
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
            n3ldg_cuda::Assert(batch.at(idx)->val.verify("Dropout forward"));
        }
#endif
    }
#else
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }
#endif

#if USE_GPU
    void backward() {
        int count = batch.size();
        std::vector<dtype*> vals, losses, in_losses;
        vals.reserve(count);
        losses.reserve(count);
        in_losses.reserve(count);
        for (Node *n : batch) {
            DropoutNode *tanh = static_cast<DropoutNode*>(n);
#if TEST_CUDA
            tanh->loss.copyFromHostToDevice();
            tanh->in->loss.copyFromHostToDevice();
#endif
            vals.push_back(tanh->val.value);
            losses.push_back(tanh->loss.value);
            in_losses.push_back(tanh->in->loss.value);
        }
        n3ldg_cuda::DropoutBackward(losses, vals, count, dim, drop_mask.value,
                dynamicDropValue(), in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward_drop();
            n->backward();
        }
        for (Node *n : batch) {
            DropoutNode *tanh = static_cast<DropoutNode*>(n);
            n3ldg_cuda::Assert(tanh->in->loss.verify("DropoutExecute backward"));
        }
#endif
    }
#else
    void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
#endif
};



#endif
