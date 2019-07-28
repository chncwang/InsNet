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
    Node* in;
    dtype(*activate)(const dtype&);
    dtype(*derivate)(const dtype&, const dtype&);

    ActivateNode() : Node("activate") {
        in = nullptr;
        activate = ftanh;
        derivate = dtanh;
    }

    ~ActivateNode() = default;

    void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
        activate = f;
        derivate = f_deri;
    }

    void forward(Graph *cg, Node* x) {
        in = x;
        in->addParent(this);
        cg->addNode(this);
    }

    void compute() {
        val().vec() = in->val().vec().unaryExpr(ptr_fun(activate));
    }

    void backward() {
        in->loss().vec() += loss().vec() * in->val().vec().binaryExpr(val().vec(),
                ptr_fun(derivate));
    }

    PExecutor generate();

    bool typeEqual(Node* other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};


class ActivateExecutor :public Executor {
};

PExecutor ActivateNode::generate() {
    ActivateExecutor* exec = new ActivateExecutor();
    exec->batch.push_back(this);
    return exec;
};

class TanhNode :public Node {
  public:
    Node* in;

  public:
    TanhNode() : Node("tanh") {
        in = nullptr;
    }

    ~TanhNode() {
        in = nullptr;
    }

  public:
    void forward(Graph &graph, Node &input) {
        this->forward(&graph, &input);
    }

    void forward(Graph *cg, Node* x) {
        in = x;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    void compute() {
        val().vec() = in->val().vec().unaryExpr(ptr_fun(ftanh));
    }

    void backward() {
        in->loss().vec() += loss().vec() * in->val().vec().binaryExpr(val().vec(), ptr_fun(dtanh));
    }

  public:
    PExecutor generate();

    // better to rewrite for deep understanding
    bool typeEqual(Node* other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

class TanhExecutor :public Executor {
public:
    int dim;
    Tensor1D y, x;
    int sumDim;

#if USE_GPU
    void forward() {
        int count = batch.size();
        std::vector<dtype*> xs, ys;
        xs.reserve(count);
        ys.reserve(count);
        for (Node *n : batch) {
            TanhNode *tanh = static_cast<TanhNode*>(n);
#if TEST_CUDA
            tanh->in->val().copyFromHostToDevice();
#endif
            xs.push_back(tanh->in->val().value);
            ys.push_back(tanh->val().value);
        }

        n3ldg_cuda::TanhForward(n3ldg_cuda::ActivatedEnum::TANH, xs, count, dim, ys);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch.at(idx)->getVal().verify("Tanh forward"));
        }
#endif
    }
#else
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for  
        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
            sumDim += batch[idx]->getDim();
        }

        x.init(sumDim);
        y.init(sumDim);

        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            for (int idy = 0; idy < ptr->getDim(); idy++) {
                x[offset + idy] = ptr->in->val()[idy];
            }
            offset += ptr->getDim();
        }

        y.vec() = x.vec().unaryExpr(ptr_fun(ftanh));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            for (int idy = 0; idy < ptr->getDim(); idy++) {
                ptr->val()[idy] = y[offset + idy];
            }
            offset += ptr->getDim();
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
            tanh->loss().copyFromHostToDevice();
            tanh->in->loss().copyFromHostToDevice();
#endif
            vals.push_back(tanh->val().value);
            losses.push_back(tanh->loss().value);
            in_losses.push_back(tanh->in->loss().value);
        }
        n3ldg_cuda::TanhBackward(n3ldg_cuda::ActivatedEnum::TANH, losses, vals, count, dim,
                in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            TanhNode *tanh = static_cast<TanhNode*>(n);
            n3ldg_cuda::Assert(tanh->in->getLoss().verify("TanhExecutor backward"));
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
            for (int idy = 0; idy < ptr->getDim(); idy++) {
                ly[offset + idy] = ptr->loss()[idy];
            }
            offset += ptr->getDim();
        }

        lx.vec() = ly.vec() * x.vec().binaryExpr(y.vec(), ptr_fun(dtanh));

        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            TanhNode* ptr = (TanhNode*)batch[idx];
            for (int idy = 0; idy < ptr->getDim(); idy++) {
                ptr->in->loss()[idy] += lx[offset + idy];
            }
            offset += ptr->getDim();
        }
    }
#endif
};

PExecutor TanhNode::generate() {
    TanhExecutor* exec = new TanhExecutor();
    exec->batch.push_back(this);
    exec->dim = getDim();
    return exec;
};


class SigmoidNode :public Node {
public:
    Node* in;

    SigmoidNode() : Node("sigmoid") {
        in = nullptr;
    }

    ~SigmoidNode() {
        in = nullptr;
    }

    void forward(Graph &graph, Node &input) {
        this->forward(&graph, &input);
    }

    void forward(Graph *cg, Node* x) {
        in = x;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    void compute() {
        val().vec() = in->val().vec().unaryExpr(ptr_fun(fsigmoid));
    }

    void backward() {
        in->loss().vec() += loss().vec() * in->val().vec().binaryExpr(val().vec(),
                ptr_fun(dsigmoid));
    }

  public:
    PExecutor generate();

    // better to rewrite for deep understanding
    bool typeEqual(Node* other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};


class SigmoidExecutor :public Executor {
  public:
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
        for (Node *n : batch) {
            SigmoidNode *tanh = static_cast<SigmoidNode*>(n);
#if TEST_CUDA
            tanh->in->val().copyFromHostToDevice();
#endif
            xs.push_back(tanh->in->val().value);
            ys.push_back(tanh->val().value);
        }

        n3ldg_cuda::TanhForward(n3ldg_cuda::ActivatedEnum::SIGMOID, xs, count, dim, ys);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch.at(idx)->getVal().verify("Sigmoid forward"));
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
            tanh->loss().copyFromHostToDevice();
            tanh->in->loss().copyFromHostToDevice();
#endif
            vals.push_back(tanh->val().value);
            losses.push_back(tanh->loss().value);
            in_losses.push_back(tanh->in->loss().value);
        }
        n3ldg_cuda::TanhBackward(n3ldg_cuda::ActivatedEnum::SIGMOID, losses, vals, count, dim,
                in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            SigmoidNode *tanh = static_cast<SigmoidNode*>(n);
            n3ldg_cuda::Assert(tanh->in->getLoss().verify("SigmoidExecutor backward"));
        }
#endif
    }
#else
#endif
};

PExecutor SigmoidNode::generate() {
    SigmoidExecutor* exec = new SigmoidExecutor();
    exec->batch.push_back(this);
    exec->dim = getDim();
    return exec;
};


class ReluNode :public Node {
  public:
    Node* in;

  public:
    ReluNode() : Node("relu") {
        in = nullptr;
    }

    ~ReluNode() {
        in = nullptr;
    }

    void forward(Graph *cg, Node* x) {
        in = x;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    void compute() {
        val().vec() = in->val().vec().unaryExpr(ptr_fun(frelu));
    }

    void backward() {
        in->loss().vec() += loss().vec() * in->val().vec().binaryExpr(val().vec(), ptr_fun(drelu));
    }

  public:
    PExecutor generate();

    // better to rewrite for deep understanding
    bool typeEqual(Node* other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

class ReluExecutor :public Executor {};

PExecutor ReluNode::generate() {
    ReluExecutor* exec = new ReluExecutor();
    exec->batch.push_back(this);
    return exec;
};


class PDotNode : public Node {
public:
    Node* in1, *in2;

    PDotNode() : Node("point-dot", 1) {
        in1 = nullptr;
        in2 = nullptr;
    }

    void init(int dim = 1){
        if (dim != 1) {
            abort();
        }
        Node::init(dim);
    }

    void forward(Graph *cg, Node* x1, Node* x2) {
        in1 = x1;
        in2 = x2;
        in1->addParent(this);
        in2->addParent(this);
        cg->addNode(this);
    }

    void compute() {
        val()[0] = 0.0;
        for (int idx = 0; idx < in1->getDim(); idx++) {
            val()[0] += in1->val()[idx] * in2->val()[idx];
        }
    }

    void backward() {
        for (int idx = 0; idx < in1->getDim(); idx++) {
            in1->loss()[idx] += loss()[0] * in2->val()[idx];
            in2->loss()[idx] += loss()[0] * in1->val()[idx];
        }
    }

    PExecutor generate();
};

#if USE_GPU
class PDotExecutor :public Executor {
public:
    void  forward() {
        int count = batch.size();
        std::vector<dtype*> vals;
        ins1.reserve(count);
        ins2.reserve(count);
        vals.reserve(count);
        for (Node *node : batch) {
            PDotNode *dot = static_cast<PDotNode*>(node);
            ins1.push_back(dot->in1->val().value);
            ins2.push_back(dot->in2->val().value);
            vals.push_back(dot->val().value);
        }

        n3ldg_cuda::PDotForward(ins1, ins2, count,
                static_cast<PDotNode*>(batch.at(0))->in1->getDim(), vals);
#if TEST_CUDA
        for (Node *node : batch) {
            PDotNode *dot = static_cast<PDotNode*>(node);
            n3ldg_cuda::Assert(dot->in1->getVal().verify("PDot in1"));
            n3ldg_cuda::Assert(dot->in2->getVal().verify("PDot in2"));
            node->compute();
            n3ldg_cuda::Assert(node->getVal().verify("PDot forward"));
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
            losses.push_back(dot->loss().value);
            in_losses1.push_back(dot->in1->loss().value);
            in_losses2.push_back(dot->in2->loss().value);
        }
        n3ldg_cuda::PDotBackward(losses, ins1, ins2, count,
                static_cast<PDotNode*>(batch.at(0))->in1->getDim(), in_losses1,
                in_losses2);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
            n3ldg_cuda::Assert(batch[idx]->getLoss().verify("PDotExecutor backward"));
        }

        for (Node *node : batch) {
            PDotNode *dot = static_cast<PDotNode*>(node);
            n3ldg_cuda::Assert(dot->in1->getLoss().verify("PDotExecutor backward in1"));
            n3ldg_cuda::Assert(dot->in2->getLoss().verify("PDotExecutor backward in2"));
        }
#endif
    }

private:
    std::vector<dtype*> ins1;
    std::vector<dtype*> ins2;
};
#else
class PDotExecutor :public Executor {
};
#endif


PExecutor PDotNode::generate() {
    PDotExecutor* exec = new PDotExecutor();
    exec->batch.push_back(this);
    return exec;
}

class DropoutNode : public Node {
public:
    DropoutNode(dtype dropout, bool is_training) : Node("dropout"), drop_value_(dropout),
    is_training_(is_training) {}

    void init(int dimm) override {
        Node::init(dimm);
        drop_mask_.init(dimm);
    }

#if USE_GPU
    void initOnHostAndDevice(int ndim) override {
        Node::initOnHostAndDevice(ndim);
        drop_mask_.init(ndim);
    }
#endif

    virtual void generate_dropmask() {
        int dropNum = (int)(getDim() * drop_value_);
        std::vector<int> tmp_masks(getDim());
        for (int idx = 0; idx < getDim(); idx++) {
            tmp_masks[idx] = idx < dropNum ? 0 : 1;
        }
        random_shuffle(tmp_masks.begin(), tmp_masks.end());
        for (int idx = 0; idx < getDim(); idx++) {
            drop_mask_[idx] = tmp_masks[idx];
        }
    }

    void forward(Graph &graph, Node &x) {
        in_ = &x;
        in_->addParent(this);
        graph.addNode(this);
    }

    void compute() override {
        if (is_training_) {
#if !TEST_CUDA
            generate_dropmask();
#endif
        } else {
            drop_mask_ = 1 - drop_value_;
        }
//        cout << boost::format("compute is_training:%1%\n") % is_training_;
//        std::cout << "before compute:" << in_->val().toString() << std::endl;
        val().vec() = in_->val().vec() * drop_mask_.vec();
//        std::cout << "after compute:" << val().toString() << std::endl;
    }

    void backward() override {
//        cout << boost::format("backward is_training:%1%\n") % is_training_;
//        std::cout << "before backward:" << loss().toString() << std::endl;
        in_->loss().vec() += loss().vec() * drop_mask_.vec();
//        std::cout << "after backward:" << in_->loss().toString() << std::endl;
    }

    bool typeEqual(Node *other) override {
        DropoutNode *o = static_cast<DropoutNode*>(other);
        if (o->is_training_ != is_training_) {
            std::cerr << "is_training not equal" << std::endl;
            abort();
        }
        return Node::typeEqual(other) && abs(drop_value_ - o->drop_value_) < 0.001f;
    }

    size_t typeHashCode() const override {
        return Node::typeHashCode() ^ (std::hash<int>{}((int)(10000 * drop_value_)) << 1);
    }

    PExecutor generate() override;

    Node* in() {
        return in_;
    }

    bool isTraning() {
        return is_training_;
    }

    Tensor1D &dropMask() {
        return drop_mask_;
    }

private:
    Node* in_ = nullptr;
    Tensor1D drop_mask_;
    dtype drop_value_ = 0.0f;
    bool is_training_ = true;
};

class DropoutExecutor :public Executor {
  public:
    Tensor2D drop_mask;
    dtype drop_value;
    int dim;
    bool is_training;

#if USE_GPU
    void CalculateDropMask(int count, int dim, const Tensor2D &mask) {
        if (is_training) {
            n3ldg_cuda::CalculateDropoutMask(drop_value, count, dim, mask.value);
        }
    }

    void forward() {
        int count = batch.size();
        std::vector<dtype*> xs, ys;
        xs.reserve(count);
        ys.reserve(count);
        drop_mask.init(dim, count);
        for (Node *n : batch) {
            DropoutNode *tanh = static_cast<DropoutNode*>(n);
#if TEST_CUDA
            tanh->in()->val().copyFromHostToDevice();
#endif
            xs.push_back(tanh->in()->getVal().value);
            ys.push_back(tanh->getVal().value);
        }

        CalculateDropMask(count, dim, drop_mask);
        n3ldg_cuda::DropoutForward(xs, count, dim, is_training, drop_mask.value, drop_value, ys);
#if TEST_CUDA
        drop_mask.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < dim; ++j) {
                dtype v = drop_mask[i][j];
                static_cast<DropoutNode*>(batch.at(i))->dropMask()[j] = v <= drop_value ? 0 : 1;
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch.at(idx)->val().verify("Dropout forward"));
        }
#endif
    }

    void backward() {
        int count = batch.size();
        std::vector<dtype*> vals, losses, in_losses;
        vals.reserve(count);
        losses.reserve(count);
        in_losses.reserve(count);
        for (Node *n : batch) {
            DropoutNode *tanh = static_cast<DropoutNode*>(n);
#if TEST_CUDA
            tanh->loss().copyFromHostToDevice();
            tanh->in()->loss().copyFromHostToDevice();
#endif
            vals.push_back(tanh->val().value);
            losses.push_back(tanh->loss().value);
            in_losses.push_back(tanh->in()->loss().value);
        }
        n3ldg_cuda::DropoutBackward(losses, vals, count, dim, is_training, drop_mask.value,
                drop_value, in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            DropoutNode *tanh = static_cast<DropoutNode*>(n);
            n3ldg_cuda::Assert(tanh->in()->loss().verify("DropoutExecutor backward"));
        }
#endif
    }
#endif
};

PExecutor DropoutNode::generate() {
    DropoutExecutor* exec = new DropoutExecutor();
    exec->batch.push_back(this);
    exec->is_training = isTraning();
    exec->dim = getDim();
    return exec;
}

#endif
