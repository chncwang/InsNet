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
        vector<dtype*> inputs;
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
        vector<dtype*> losses;
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
class ActivationExecutor : public UniInputExecutor {
public:
    int calculateFLOPs() override {
        return defaultFLOPs();
    }

};
#endif

class TanhNode : public UniInputNode, public Poolable<TanhNode> {
public:
    TanhNode() : UniInputNode("tanh") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
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

    void setNodeDim(int dim) override {
        setDim(dim);
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

    void setNodeDim(int dim) override {
        setDim(dim);
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

    void setNodeDim(int dim) override {
        setDim(dim);
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

    void setNodeDim(int dim) override {
        setDim(dim);
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
#else
    int calculateFLOPs() override {
        return defaultFLOPs();
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

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    bool typeEqual(Node *other) override {
        return Node::typeEqual(other);
    }

    string typeSignature() const override {
        return Node::typeSignature();
    }

    void clear() override {
        max_indexes_.clear();
        Node::clear();
    }

    void compute() override {
        int input_row = getInput()->getDim() / getDim();
        for (int i = 0; i < getDim(); ++i) {
            float max = getInput()->getVal()[input_row * i];
            int max_i = 0;
            for (int j = 1; j < input_row; ++j) {
                if (getInput()->getVal()[input_row * i + j] > max) {
                    max = getInput()->getVal()[input_row * i + j];
                    max_i = j;
                }
            }
            max_indexes_.push_back(max_i);
            val()[i] = max;
        }
    }

    void backward() override {
        int input_row = getInput()->getDim() / getDim();
        for (int i = 0; i < getDim(); ++i) {
            getInput()->loss()[i * input_row + max_indexes_.at(i)] += getLoss()[i];
        }
    }

    Executor* generate() override;

protected:
    bool isDimLegal(const Node &input) const override {
        return input.getDim() % getDim() == 0;
    }

private:
    vector<int> max_indexes_;
    friend class MaxScalarExecutor;
};

#if USE_GPU
class MaxScalarExecutor : public UniInputExecutor {
public:
    void forward() override {
        vector<dtype*> inputs;
        vector<dtype*> results;
        max_indexes.resize(batch.size() * batch.front()->getDim());
        vector<int> head_dims;
        int dim = Executor::getDim();
        for (int i = 0; i < batch.size(); ++i) {
            MaxScalarNode *node = static_cast<MaxScalarNode*>(batch.at(i));
            inputs.push_back(node->getInput()->getVal().value);
            results.push_back(node->getVal().value);
            int head_dim = node->getInput()->getDim() / dim;
            if (head_dim * dim != node->getInput()->getDim()) {
                cerr << boost::format(
                        "MaxScalarExecutor forward head_dim:%1% dim:%2% input dim:%3%") % head_dim
                    % dim % node->getInput()->getDim() << endl;
                abort();
            }
            head_dims.push_back(head_dim);
        }
        n3ldg_cuda::MaxScalarForward(inputs, batch.size(), dim, head_dims, results, max_indexes);

#if TEST_CUDA
        Executor::forward();
        for (Node *node : batch) {
            MaxScalarNode *max_scalar = static_cast<MaxScalarNode*>(node);
            n3ldg_cuda::Assert(max_scalar->getInput()->getVal().verify(
                        "max scalar forward input"));
            n3ldg_cuda::Assert(max_scalar->getVal().verify("max scalar forward"));
        }
        cout << "max scalar forward tested:" << endl;
#endif
    }

    void backward() override {
        vector<dtype*> losses;
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
class MaxScalarExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return defaultFLOPs();
    }
};
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

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    bool typeEqual(Node *other) override {
        return getNodeType() == other->getNodeType() &&
            getInput()->getDim() == dynamic_cast<UniInputNode *>(other)->getInput()->getDim();
    }

    string typeSignature() const override {
        return getNodeType() + to_string(getInput()->getDim());
    }

    void compute() override {
        for (int i = 0; i < getColumn(); ++i) {
            for (int j = 0; j < getRow(); ++j) {
                val()[i * getRow() + j] = getInput()->getVal()[i];
            }
        }
    }

    void backward() override {
        int row = getRow();
        for (int i = 0; i < getColumn(); ++i) {
            dtype sum = 0;
            for (int j = 0; j < row; ++j) {
                sum += getLoss()[i * row + j];
            }
            getInput()->loss()[i] += sum;
        }
    }

    Executor* generate() override;

protected:
    bool isDimLegal(const Node &input) const override {
        return true;
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
        vector<dtype*> inputs;
        vector<dtype*> results;
        for (Node *node : batch) {
            ScalarToVectorNode *n = static_cast<ScalarToVectorNode*>(node);
            inputs.push_back(n->getInput()->getVal().value);
            results.push_back(n->getVal().value);
            dims_.push_back(n->getDim() / n->getInput()->getDim());
        }
        int dim = dynamic_cast<ScalarToVectorNode *>(batch.front())->getInput()->getDim();
        n3ldg_cuda::ScalarToVectorForward(inputs, batch.size(), dim, dims_, results);
#if TEST_CUDA
        Executor::testForward();
        cout << "scalarToVector tested" << endl;
#endif
    }

    void backward() override {
#if TEST_CUDA
        cout << "scalarToVector test before backward..." << endl;
        UniInputExecutor::testBeforeBackward();
        for (Node *node : batch) {
            ScalarToVectorNode * n = static_cast<ScalarToVectorNode*>(node);
            n->loss().copyFromHostToDevice();
        }
#endif
        vector<dtype*> losses;
        vector<dtype*> input_losses;
        for (Node *node : batch) {
            ScalarToVectorNode * n = static_cast<ScalarToVectorNode*>(node);
            losses.push_back(n->getLoss().value);
            input_losses.push_back(n->getInput()->getLoss().value);
        }
        int dim = dynamic_cast<ScalarToVectorNode *>(batch.front())->getInput()->getDim();
        n3ldg_cuda::ScalarToVectorBackward(losses, batch.size(), dim, dims_, input_losses);
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
class ScalarToVectorExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }
};
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

    void setNodeDim(int dim) override {
        setDim(dim);
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

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    Executor* generate() override;

    virtual bool typeEqual(Node *other) override {
        return Node::typeEqual(other);
    }

    virtual string typeSignature() const override {
        return Node::typeSignature();
    }

    void compute() override {
        for (int i = 0; i < getDim(); ++i) {
            dtype sum = 0;
            int input_row = getInput()->getDim() / getDim();
            for (int j = 0; j < input_row; ++j) {
                sum += getInput()->getVal()[i * input_row + j];
            }
            val()[i] = sum;
        }
    }

    void backward() override {
        for (int i = 0; i < getDim(); ++i) {
            int input_row = getInput()->getDim() / getDim();
            for (int j = 0; j < input_row; ++j) {
                getInput()->loss()[i * input_row + j] += getLoss()[i];
            }
        }
    }

protected:
    bool isDimLegal(const Node &input) const override {
        return input.getDim() % getDim() == 0;
    }

private:
    friend class SumExecutor;
};

#if USE_GPU
class SumExecutor : public UniInputExecutor {
    void forward() override {
        vector<dtype*> inputs;
        vector<dtype*> results;
        for (Node *node : batch) {
            SumNode *sum = static_cast<SumNode*>(node);
            inputs.push_back(sum->getInput()->getVal().value);
            results.push_back(sum->getVal().value);
            int row = sum->getInput()->getDim()/ getDim();
            if (row * getDim() != sum->getInput()->getDim()) {
                cerr << boost::format("SumExecutor forward row:%1% dim:%2% input dim:%3%") %
                    row % getDim() % sum->getInput()->getDim() << endl;
                abort();
            }
            dims_.push_back(row);
        }
        n3ldg_cuda::VectorSumForward(inputs, batch.size(), getDim(), dims_, results);
#if TEST_CUDA
        Executor::testForward();
        cout << "sum tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype*> losses;
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
class SumExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return defaultFLOPs();
    }
};
#endif

Executor *SumNode::generate() {
    SumExecutor *e = new SumExecutor();
    return e;
}

class ScaledExecutor;

class ScaledNode : public UniInputNode, public Poolable<ScaledNode> {
public:
    ScaledNode() : UniInputNode("ScaledNode") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void setFactor(dtype factor) {
        factor_ = factor;
    }

    void compute() override {
        val().vec() = factor_ * getInput()->getVal().vec();
    }

    void backward() override {
        getInput()->loss().vec() += factor_ * getLoss().vec();
    }

    Executor* generate() override;

    virtual bool typeEqual(Node *other) override {
        return getNodeType() == other->getNodeType();
    }

    virtual string typeSignature() const override {
        return getNodeType();
    }

protected:
    bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }

private:
    dtype factor_ = 1;
    friend class ScaledExecutor;
};

#if USE_GPU
class ScaledExecutor : public UniInputExecutor {
public:
    void forward() override {
        vector<dtype *> in_vals;
        for (Node *node : batch) {
            ScaledNode *scaled = static_cast<ScaledNode *>(node);
            in_vals.push_back(scaled->getInput()->getVal().value);
            dims.push_back(scaled->getDim());
            factors.push_back(scaled->factor_);
        }
        auto vals = getVals();
        n3ldg_cuda::ScaledForward(in_vals, batch.size(), dims, factors, vals);
#if TEST_CUDA
        testForward();
        cout << "ScaledExecutor forward tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype *> in_grads;
        for (Node *node : batch) {
            ScaledNode *scaled = static_cast<ScaledNode *>(node);
            in_grads.push_back(scaled->getInput()->getLoss().value);
        }
        auto grads = getGrads();
        n3ldg_cuda::ScaledBackward(grads, batch.size(), dims, factors, in_grads);
#if TEST_CUDA
        testBackward();
        cout << "ScaledExecutor backward tested" << endl;
#endif
    }

private:
    vector<int> dims;
    vector<dtype> factors;
};
#else
class ScaledExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return defaultFLOPs();
    }
};
#endif

Executor *ScaledNode::generate() {
    return new ScaledExecutor;
}

namespace n3ldg_plus {

Node *maxScalar(Graph &graph, Node &input, int input_col) {
    MaxScalarNode *node = MaxScalarNode::newNode(input_col);
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

Node *scalarToVector(Graph &graph, int row, Node &input) {
    ScalarToVectorNode *node = ScalarToVectorNode::newNode(row * input.getDim());
    node->setColumn(input.getDim());
    node->forward(graph, input);
    return node;
}

Node *vectorSum(Graph &graph, Node &input,  int input_col) {
    SumNode *sum = SumNode::newNode(input_col);
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

Node *scaled(Graph &graph, Node &input, dtype factor) {
    ScaledNode *node = ScaledNode::newNode(input.getDim());
    node->setFactor(factor);
    node->forward(graph, input);
    return node;
}

}


#endif
