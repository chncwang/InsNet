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

class TanhNode : public UniInputNode {
public:
    TanhNode() : UniInputNode("tanh") {}

    void compute() {
        val().vec() = getInput()->val().vec().unaryExpr(ptr_fun(ftanh));
    }

    void backward() {
        getInput()->loss().vec() += loss().vec() * getInput()->val().vec().binaryExpr(val().vec(),
                ptr_fun(dtanh));
    }

    PExecutor generate();

protected:
    virtual bool isDimLegal(const Node &input) const {
        return input.getDim() == getDim();
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
            tanh->getInput()->val().copyFromHostToDevice();
#endif
            xs.push_back(tanh->getInput()->val().value);
            ys.push_back(tanh->val().value);
        }

        n3ldg_cuda::TanhForward(ActivatedEnum::TANH, xs, count, dim, ys);
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
                x[offset + idy] = ptr->getInput()->val()[idy];
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
            tanh->getInput()->loss().copyFromHostToDevice();
#endif
            vals.push_back(tanh->val().value);
            losses.push_back(tanh->loss().value);
            in_losses.push_back(tanh->getInput()->loss().value);
        }
        n3ldg_cuda::TanhBackward(ActivatedEnum::TANH, losses, vals, count, dim,
                in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            TanhNode *tanh = static_cast<TanhNode*>(n);
            n3ldg_cuda::Assert(tanh->getInput()->getLoss().verify("TanhExecutor backward"));
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
                ptr->getInput()->loss()[idy] += lx[offset + idy];
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


class SigmoidNode :public UniInputNode {
public:
    SigmoidNode() : UniInputNode("sigmoid") {}

    void compute() {
        val().vec() = getInput()->val().vec().unaryExpr(ptr_fun(fsigmoid));
    }

    void backward() {
        getInput()->loss().vec() += loss().vec() * getInput()->val().vec().binaryExpr(val().vec(),
                ptr_fun(dsigmoid));
    }

    PExecutor generate();

protected:
    virtual bool isDimLegal(const Node &input) const {
        return input.getDim() == getDim();
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
            tanh->getInput()->val().copyFromHostToDevice();
#endif
            xs.push_back(tanh->getInput()->val().value);
            ys.push_back(tanh->val().value);
        }

        n3ldg_cuda::TanhForward(ActivatedEnum::SIGMOID, xs, count, dim, ys);
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
            tanh->getInput()->loss().copyFromHostToDevice();
#endif
            vals.push_back(tanh->val().value);
            losses.push_back(tanh->loss().value);
            in_losses.push_back(tanh->getInput()->loss().value);
        }
        n3ldg_cuda::TanhBackward(ActivatedEnum::SIGMOID, losses, vals, count, dim,
                in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            SigmoidNode *tanh = static_cast<SigmoidNode*>(n);
            n3ldg_cuda::Assert(tanh->getInput()->getLoss().verify("SigmoidExecutor backward"));
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
        val().vec() = in_->val().vec() * drop_mask_.vec();
    }

    void backward() override {
        in_->loss().vec() += loss().vec() * drop_mask_.vec();
    }

    bool typeEqual(Node *other) override {
        DropoutNode *o = static_cast<DropoutNode*>(other);
        if (o->is_training_ != is_training_) {
            std::cerr << "is_training not equal" << std::endl;
            abort();
        }
        return Node::typeEqual(other) && abs(drop_value_ - o->drop_value_) < 0.001f;
    }

    string typeSignature() const override {
        return Node::typeSignature() + "-" + to_string(drop_value_);
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

class MaxScalarNode : public UniInputNode {
public:
    MaxScalarNode() : UniInputNode("max_scalar_node") {}

    void initAsScalar() {
        init(1);
    }

    void compute() override {
        float max = getInput()->getVal()[0];
        int max_i = 0;
        for (int i = 1; i < getInput()->getDim(); ++i) {
            if (getInput()->getVal()[i] > max) {
                max = getInput()->getVal()[i];
                max_i = i;
            }
        }
        max_i_ = max_i;
        val()[0] = max;
    }

    void backward() override {
        getInput()->loss()[max_i_] += getLoss()[0];
    }

    Executor* generate() override;

protected:
    bool isDimLegal(const Node &input) const override {
        return true;
    }

private:
    int max_i_;
    friend class MaxScalarExecutor;
};

#if USE_GPU
class MaxScalarExecutor : public UniInputExecutor {
public:
    void forward() override {
        vector<const dtype*> inputs;
        vector<dtype*> results;
        max_indexes.resize(batch.size());
        for (int i = 0; i < batch.size(); ++i) {
            MaxScalarNode *node = static_cast<MaxScalarNode*>(batch.at(i));
            inputs.push_back(node->getInput()->getVal().value);
            results.push_back(node->getVal().value);
        }
        int input_dim = static_cast<MaxScalarNode*>(batch.front())->getInput()->getDim();
        n3ldg_cuda::MaxScalarForward(inputs, batch.size(), input_dim, results, max_indexes);

        for (int i = 0; i < batch.size(); ++i) {
            MaxScalarNode *node = static_cast<MaxScalarNode*>(batch.at(i));
            node->max_i_ = max_indexes.at(i);
        }
#if TEST_CUDA
        Executor::forward();
        int i = 0;
        for (Node *node : batch) {
            MaxScalarNode *max_scalar = static_cast<MaxScalarNode*>(node);
            n3ldg_cuda::Assert(max_scalar->getInput()->getVal().verify("max scalar forward input"));
            n3ldg_cuda::Assert(max_scalar->getVal().verify("max scalar forward"));
            ++i;
        }
        cout << "max scalar tested:" << endl;
#endif
    }

    void backward() override {
        vector<const dtype*> losses;
        vector<dtype *> input_losses;

        for (Node *node : batch) {
            MaxScalarNode *max_scalar = static_cast<MaxScalarNode*>(node);
            losses.push_back(max_scalar->getLoss().value);
            input_losses.push_back(max_scalar->getInput()->getLoss().value);
        }

        n3ldg_cuda::MaxScalarBackward(losses, max_indexes, batch.size(), input_losses);
#if TEST_CUDA
        UniInputExecutor::testBackward();
        cout << "tested" << endl;
#endif
    }

private:
    vector<int> max_indexes;
};
#else
class MaxScalarExecutor : public Executor {};
#endif

Executor *MaxScalarNode::generate() {
    MaxScalarExecutor * executor = new MaxScalarExecutor();
    return executor;
}

class ScalarToVectorNode : public UniInputNode {
public:
    ScalarToVectorNode() : UniInputNode("scalar_to_vector") {}

    void compute() override {
        for (int i = 0; i < getDim(); ++i) {
            val()[i] = getInput()->getVal()[0];
        }
    }

    void backward() override {
        int dim = getDim();
        dtype sum = 0;
        for (int i = 0; i < dim; ++i) {
            sum += getLoss()[i];
        }
        getInput()->loss()[0] += sum;
    }

    Executor* generate() override;

protected:
    bool isDimLegal(const Node &input) const override {
        return input.getDim() == 1;
    }

private:
    friend class ScalarToVectorExecutor;
};

namespace n3ldg_plus {

Node *scalarToVector(Graph &graph, int dim, Node &input) {
    ScalarToVectorNode *node = new ScalarToVectorNode;
    node->init(dim);
    node->forward(graph, input);
    return node;
}

}

#if USE_GPU
class ScalarToVectorExecutor : public UniInputExecutor {
public:
    void forward() override {
#if TEST_CUDA
        UniInputExecutor::testForwardInpputs();
#endif
        vector<const dtype*> inputs;
        vector<dtype*> results;
        for (Node *node : batch) {
            ScalarToVectorNode * n = static_cast<ScalarToVectorNode*>(node);
            inputs.push_back(n->getInput()->getVal().value);
            results.push_back(n->getVal().value);
        }
        n3ldg_cuda::ScalarToVectorForward(inputs, batch.size(), getDim(), results);
#if TEST_CUDA
        Executor::testForward();
        cout << "scalarToVector tested" << endl;
#endif
    }

    void backward() override {
        vector<const dtype*> losses;
        vector<dtype*> input_losses;
        for (Node *node : batch) {
            ScalarToVectorNode * n = static_cast<ScalarToVectorNode*>(node);
            losses.push_back(n->getLoss().value);
            input_losses.push_back(n->getInput()->getLoss().value);
        }
        n3ldg_cuda::ScalarToVectorBackward(losses, batch.size(), getDim(), input_losses);
#if TEST_CUDA
        UniInputExecutor::testBackward();
        cout << "ScalarToVectorNode backward tested" << endl;
#endif
    }
};
#else
class ScalarToVectorExecutor : public Executor {};
#endif

Executor *ScalarToVectorNode::generate() {
    ScalarToVectorExecutor * executor = new ScalarToVectorExecutor();
    return executor;
}

class ExpNode : public UniInputNode {
public:
    ExpNode() : UniInputNode("exp") {}

    Executor* generate() override;

    void compute() override {
        val().vec() = getInput()->getVal().vec().exp();
    }

    void backward() override {
        getInput()->loss().vec() += getLoss().vec() * getVal().vec();
    }

protected:
    bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }

private:
    friend class ExpExecutor;
};

#if USE_GPU
class ExpExecutor : public UniInputExecutor {
public:
    vector<dtype*> vals;

    void forward() override {
        vector<const dtype*> inputs;
        for (Node *node : batch) {
            ExpNode *expnode = static_cast<ExpNode*>(node);
            inputs.push_back(expnode->getInput()->getVal().value);
            vals.push_back(expnode->getVal().value);
        }
        n3ldg_cuda::ExpForward(inputs, batch.size(), getDim(), vals);
#if TEST_CUDA
        Executor::testForward();
        cout << "exp forward tested" << endl;
#endif
    }

    void backward() override {
        vector<const dtype*> losses, vals;
        vector<dtype*> input_losses;

        for (Node *node : batch) {
            ExpNode *exp = static_cast<ExpNode*>(node);
            vals.push_back(node->getVal().value);
            losses.push_back(exp->getLoss().value);
            input_losses.push_back(exp->getInput()->getLoss().value);
        }

        n3ldg_cuda::ExpBackward(losses, vals, batch.size(), getDim(), input_losses);
#if TEST_CUDA
        UniInputExecutor::testBackward();
        cout << "exp backward tested" << endl;
#endif
    }
};
#else
class ExpExecutor : public Executor {};
#endif

Executor *ExpNode::generate() {
    ExpExecutor * executor = new ExpExecutor();
    return executor;
}

class SumNode : public UniInputNode {
public:
    SumNode(): UniInputNode("sum") {}

    Executor* generate() override;

    void initAsScalar() {
        init(1);
    }

    void compute() override {
        dtype sum = 0;
        for (int i = 0; i < getInput()->getDim(); ++i) {
            sum += getInput()->getVal()[i];
        }
        val()[0] = sum;
    }

    void backward() override {
        for (int i = 0; i < getInput()->getDim(); ++i) {
            getInput()->loss()[i] += getLoss()[0];
        }
    }

protected:
    bool isDimLegal(const Node &input) const override {
        return true;
    }

private:
    friend class SumExecutor;
};

namespace n3ldg_plus {

Node *vectorSum(Graph &graph, Node &input) {
    SumNode *sum = new SumNode;
    sum->initAsScalar();
    sum->forward(graph, input);
    return sum;
}

}

#if USE_GPU
class SumExecutor : public UniInputExecutor {
    void forward() override {
        vector<const dtype*> inputs;
        vector<dtype*> results;
        for (Node *node : batch) {
            SumNode *sum = static_cast<SumNode*>(node);
            inputs.push_back(sum->getInput()->getVal().value);
            results.push_back(sum->getVal().value);
        }
        n3ldg_cuda::VectorSumForward(inputs, batch.size(),
                static_cast<SumNode*>(batch.front())->getInput()->getDim(), results);
#if TEST_CUDA
        Executor::testForward();
        cout << "sum tested" << endl;
#endif
    }

    void backward() override {
        vector<const dtype*> losses;
        vector<dtype*> input_losses;
        for (Node *node : batch) {
#if TEST_CUDA
            node->loss().copyFromDeviceToHost();
#endif
            losses.push_back(node->getLoss().value);
            SumNode *sum = static_cast<SumNode*>(node);
            input_losses.push_back(sum->getInput()->getLoss().value);
        }

        int dim = static_cast<SumNode*>(batch.front())->getInput()->getDim();
        n3ldg_cuda::VectorSumBackward(losses, batch.size(), dim, input_losses);
#if TEST_CUDA
        UniInputExecutor::testBackward();
        cout << "sum backward tested" << endl;
#endif
    }
};
#else
class SumExecutor : public Executor {};
#endif

Executor *SumNode::generate() {
    SumExecutor *e = new SumExecutor();
    return e;
}

#endif
