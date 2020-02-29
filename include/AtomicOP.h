#ifndef ATOMICIOP_H_
#define ATOMICIOP_H_

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

#if USE_GPU
template<ActivatedEnum activation>
class ActivationExecutor : public UniInputExecutor {
public:
    vector<dtype*> vals;

    void forward() override {
        vector<const dtype*> inputs;
        for (Node *node : batch) {
            UniInputNode *expnode = static_cast<UniInputNode*>(node);
            inputs.push_back(expnode->getInput()->getVal().value);
            vals.push_back(expnode->getVal().value);
            dims_.push_back(node->getDim());
        }
        n3ldg_cuda::ActivationForward(activation, inputs, batch.size(), dims_, vals);
#if TEST_CUDA
        Executor::testForward();
        cout << "exp forward tested" << endl;
#endif
    }

    void backward() override {
        vector<const dtype*> losses;
        vector<dtype*> vals;
        vector<dtype*> input_losses;

        for (Node *node : batch) {
            UniInputNode *exp = static_cast<UniInputNode*>(node);
            vals.push_back(node->getVal().value);
            losses.push_back(exp->getLoss().value);
            input_losses.push_back(exp->getInput()->getLoss().value);
        }

        n3ldg_cuda::ActivationBackward(activation, losses, vals, batch.size(), dims_,
                input_losses);
#if TEST_CUDA
        UniInputExecutor::testBackward();
        cout << "exp backward tested" << endl;
#endif
    }

private:
    vector<int> dims_;
};
#else
template<ActivatedEnum activation>
class ActivationExecutor : public UniInputExecutor {};
#endif

class TanhNode : public UniInputNode, public Poolable<TanhNode> {
public:
    TanhNode() : UniInputNode("tanh") {}

    void initNode(int dim) override {
        init(dim);
    }

    int getKey() const override {
        return getDim();
    }

    void compute() override {
        val().vec() = getInput()->val().vec().unaryExpr(ptr_fun(ftanh));
    }

    void backward() override {
        getInput()->loss().vec() += loss().vec() * getInput()->val().vec().binaryExpr(val().vec(),
                ptr_fun(dtanh));
    }

    PExecutor generate() override;

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }
};

PExecutor TanhNode::generate() {
    return new ActivationExecutor<ActivatedEnum::TANH>;
};


class SigmoidNode :public UniInputNode, public Poolable<SigmoidNode> {
public:
    SigmoidNode() : UniInputNode("sigmoid") {}

    void initNode(int dim) override {
        init(dim);
    }

    int getKey() const override {
        return getDim();
    }

    void compute() override {
        val().vec() = getInput()->val().vec().unaryExpr(ptr_fun(fsigmoid));
    }

    void backward() override {
        getInput()->loss().vec() += loss().vec() * getInput()->val().vec().binaryExpr(val().vec(),
                ptr_fun(dsigmoid));
    }

    PExecutor generate() override {
        return new ActivationExecutor<ActivatedEnum::SIGMOID>;
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }
};

class ReluNode :public UniInputNode, public Poolable<ReluNode> {
public:
    ReluNode() : UniInputNode("relu") {}

    void initNode(int dim) override {
        init(dim);
    }

    int getKey() const override {
        return getDim();
    }

    void compute() override {
        val().vec() = getInput()->val().vec().unaryExpr(ptr_fun(frelu));
    }

    void backward() override {
        getInput()->loss().vec() += loss().vec() * getInput()->val().vec().binaryExpr(val().vec(),
                ptr_fun(drelu));
    }

    PExecutor generate() override {
        return new ActivationExecutor<ActivatedEnum::RELU>;
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }
};

class SqrtNode :public UniInputNode, public Poolable<SqrtNode> {
public:
    SqrtNode() : UniInputNode("sqrt") {}

    void initNode(int dim) override {
        init(dim);
    }

    int getKey() const override {
        return getDim();
    }

    void compute() override {
        val().vec() = getInput()->val().vec().unaryExpr(ptr_fun(fsqrt));
    }

    void backward() override {
        getInput()->loss().vec() += loss().vec() * val().vec().unaryExpr(ptr_fun(dsqrt));
    }

    PExecutor generate() override {
        return new ActivationExecutor<ActivatedEnum::SQRT>;
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }
};

class DropoutNode : public Node, public Poolable<DropoutNode> {
public:
    DropoutNode() : Node("dropout") {}

    void initNode(int dim) override {
        init(dim);
    }

    int getKey() const override {
        return getDim();
    }

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

    void setIsTraining(bool is_training) {
        is_training_ = is_training;
    }

    void setDropValue(dtype drop_value) {
        drop_value_ = drop_value;
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
        if (is_training) {
            drop_mask.copyFromDeviceToHost();
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < dim; ++j) {
                    dtype v = drop_mask[i][j];
                    static_cast<DropoutNode*>(batch.at(i))->dropMask()[j] = v <= drop_value ?
                        0 : 1;
                }
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch.at(idx)->val().verify("Dropout forward"));
            batch[idx]->getVal().print();
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
    exec->drop_value = drop_value_;
    exec->dim = getDim();
    return exec;
}

class MaxScalarNode : public UniInputNode, public Poolable<MaxScalarNode> {
public:
    MaxScalarNode() : UniInputNode("max_scalar_node") {}

    void initNode(int dim) override {
        init(dim);
    }

    int getKey() const override {
        return getDim();
    }

    bool typeEqual(Node *other) override {
        return Node::typeEqual(other);
    }

    string typeSignature() const override {
        return Node::typeSignature();
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
        vector<int> dims;
        for (int i = 0; i < batch.size(); ++i) {
            MaxScalarNode *node = static_cast<MaxScalarNode*>(batch.at(i));
            inputs.push_back(node->getInput()->getVal().value);
            results.push_back(node->getVal().value);
            dims.push_back(node->getInput()->getDim());
        }
        n3ldg_cuda::MaxScalarForward(inputs, batch.size(), dims, results, max_indexes);

        for (int i = 0; i < batch.size(); ++i) {
            MaxScalarNode *node = static_cast<MaxScalarNode*>(batch.at(i));
            if (max_indexes.at(i) > 100000 || max_indexes.at(i) < 0) {
                cerr << "illegal max index returned " << max_indexes.at(i) << endl;
                node->getInput()->getVal().print();
                node->getVal().print();
                abort();
            }
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
        cout << "max scalar forward tested:" << endl;
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
        cout << "max scalar backward tested" << endl;
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

class ScalarToVectorNode : public UniInputNode, public Poolable<ScalarToVectorNode> {
public:
    ScalarToVectorNode() : UniInputNode("scalar_to_vector") {}

    void initNode(int dim) override {
        init(dim);
    }

    int getKey() const override {
        return getDim();
    }

    bool typeEqual(Node *other) override {
        return getNodeType() == other->getNodeType();
    }

    string typeSignature() const override {
        return getNodeType();
    }

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
            ScalarToVectorNode *n = static_cast<ScalarToVectorNode*>(node);
            inputs.push_back(n->getInput()->getVal().value);
            results.push_back(n->getVal().value);
            dims_.push_back(n->getDim());
        }
        n3ldg_cuda::ScalarToVectorForward(inputs, batch.size(), dims_, results);
#if TEST_CUDA
        Executor::testForward();
        cout << "scalarToVector tested" << endl;
#endif
    }

    void backward() override {
#if TEST_CUDA
        cout << "scalarToVector test before backward..." << endl;
        for (Node *node : batch) {
            ScalarToVectorNode * n = static_cast<ScalarToVectorNode*>(node);
            cout << n->getInput()->getNodeType() << endl;
        }
        UniInputExecutor::testBeforeBackward();
#endif
        vector<const dtype*> losses;
        vector<dtype*> input_losses;
        for (Node *node : batch) {
            ScalarToVectorNode * n = static_cast<ScalarToVectorNode*>(node);
            losses.push_back(n->getLoss().value);
            input_losses.push_back(n->getInput()->getLoss().value);
        }
        n3ldg_cuda::ScalarToVectorBackward(losses, batch.size(), dims_, input_losses);
#if TEST_CUDA
        cout << "scalarToVector test backward..." << endl;
        UniInputExecutor::testBackward();
        cout << "ScalarToVectorNode backward tested" << endl;
#endif
    }

private:
    vector<int> dims_;
};
#else
class ScalarToVectorExecutor : public Executor {};
#endif

Executor *ScalarToVectorNode::generate() {
    ScalarToVectorExecutor * executor = new ScalarToVectorExecutor();
    return executor;
}

class ExpNode : public UniInputNode, public Poolable<ExpNode> {
public:
    ExpNode() : UniInputNode("exp") {}

    void initNode(int dim) override {
        init(dim);
    }

    int getKey() const override {
        return getDim();
    }

    Executor* generate() override {
        return new ActivationExecutor<ActivatedEnum::EXP>;
    }

    bool typeEqual(Node *other) override {
        return getNodeType() == other->getNodeType();
    }

    string typeSignature() const override {
        return getNodeType();
    }

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

class SumNode : public UniInputNode, public Poolable<SumNode> {
public:
    SumNode(): UniInputNode("sum") {}

    void initNode(int dim) override {
        init(dim);
    }

    int getKey() const override {
        return getDim();
    }

    Executor* generate() override;

    virtual bool typeEqual(Node *other) override {
        return Node::typeEqual(other);
    }

    virtual string typeSignature() const override {
        return Node::typeSignature();
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

#if USE_GPU
class SumExecutor : public UniInputExecutor {
    void forward() override {
        vector<const dtype*> inputs;
        vector<dtype*> results;
        for (Node *node : batch) {
            SumNode *sum = static_cast<SumNode*>(node);
            inputs.push_back(sum->getInput()->getVal().value);
            results.push_back(sum->getVal().value);
            dims_.push_back(sum->getInput()->getDim());
        }
        n3ldg_cuda::VectorSumForward(inputs, batch.size(), dims_, results);
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

        n3ldg_cuda::VectorSumBackward(losses, batch.size(), dims_, input_losses);
#if TEST_CUDA
        UniInputExecutor::testBackward();
        cout << "sum backward tested" << endl;
#endif
    }

private:
    vector<int> dims_;
};
#else
class SumExecutor : public Executor {};
#endif

Executor *SumNode::generate() {
    SumExecutor *e = new SumExecutor();
    return e;
}

namespace n3ldg_plus {

Node *maxScalar(Graph &graph, Node &input) {
    MaxScalarNode *node = MaxScalarNode::newNode(1);
    node->forward(graph, input);
    return node;
}

Node *tanh(Graph &graph, Node &input) {
    TanhNode *result = TanhNode::newNode(input.getDim());
    result->forward(graph, input);
    return result;
}

Node *sigmoid(Graph &graph, Node &input) {
    SigmoidNode *result = SigmoidNode::newNode(input.getDim());
    result->forward(graph, input);
    return result;
}

Node *relu(Graph &graph, Node &input) {
    ReluNode *result = ReluNode::newNode(input.getDim());
    result->forward(graph, input);
    return result;
}

Node *sqrt(Graph &graph, Node &input) {
    SqrtNode *result = SqrtNode::newNode(input.getDim());
    result->forward(graph, input);
    return result;
}

Node *scalarToVector(Graph &graph, int dim, Node &input) {
    ScalarToVectorNode *node = ScalarToVectorNode::newNode(dim);
    node->forward(graph, input);
    return node;
}

Node *vectorSum(Graph &graph, Node &input) {
    SumNode *sum = SumNode::newNode(1);
    sum->forward(graph, input);
    return sum;
}

Node *exp(Graph &graph, Node &input) {
    ExpNode *node = ExpNode::newNode(input.getDim());
    node->forward(graph, input);
    return node;
}

Node *dropout(Graph &graph, Node &input, dtype dropout, bool is_training) {
    DropoutNode *node = DropoutNode::newNode(input.getDim());
    node->setIsTraining(is_training);
    node->setDropValue(dropout);
    node->forward(graph, input);
    return node;
}

}


#endif
