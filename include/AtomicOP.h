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

    string typeSignature() const override {
        return Node::typeSignature();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }
};

class BatchedReluNode : public BatchedNodeImpl<ReluNode> {
public:
    void init(Graph &graph, BatchedNode &input) {
        allocateBatch(input.getDim(), input.batch().size());
        setInputsPerNode({&input});
        afterInit(graph, {&input});
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

class BatchedSqrtNode : public BatchedNodeImpl<SqrtNode> {
public:
    void init(Graph &graph, BatchedNode &input) {
        allocateBatch(input.getDim(), input.batch().size());
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
};

class DropoutNode : public UniInputNode, public Poolable<DropoutNode> {
public:
    DropoutNode() : UniInputNode("dropout") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        Node::setDim(dim);
    }

    void init(int dimm) override {
        Node::init(dimm);
        drop_mask_.init(dimm);
    }

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

    void compute() override {
        if (is_training_) {
#if !TEST_CUDA
            generate_dropmask();
#endif
        } else {
            drop_mask_ = 1 - drop_value_;
        }
        val().vec() = getInput()->val().vec() * drop_mask_.vec();
    }

    void backward() override {
        getInput()->loss().vec() += loss().vec() * drop_mask_.vec();
    }

    string typeSignature() const override {
        return Node::typeSignature() + "-" + to_string(drop_value_);
    }

    PExecutor generate() override;

    bool isTraining() {
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

    dtype dropoutValue() const {
        return drop_value_;
    }

    Json::Value toJson() const override {
        Json::Value json = UniInputNode::toJson();
        json["dropout"] = drop_value_;
        json["is_training"] = is_training_;
        return json;
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return true;
    }

private:
    Tensor1D drop_mask_;
    dtype drop_value_ = 0.0f;
    bool is_training_ = true;
};

class BatchedDropoutNode : public BatchedNodeImpl<DropoutNode> {
public:
    void init(Graph &graph, BatchedNode &input, dtype dropout, bool is_traning) {
        allocateBatch(input.getDim(), input.batch().size());
        for (Node *node : batch()) {
            DropoutNode *d = dynamic_cast<DropoutNode *>(node);
            d->setIsTraining(is_traning);
            d->setDropValue(dropout);
        }
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
};

class DropoutExecutor :public Executor {
public:
    Tensor2D drop_mask;

    bool isTraining() const {
        DropoutNode *node = dynamic_cast<DropoutNode *>(batch.front());
        return node->isTraining();
    }

    dtype dropoutValue() const {
        DropoutNode *node = dynamic_cast<DropoutNode *>(batch.front());
        return node->dropoutValue();
    }

#if USE_GPU
    void CalculateDropMask(int count, int dim, const Tensor2D &mask) {
        if (isTraining()) {
            n3ldg_cuda::CalculateDropoutMask(dropoutValue(), count, dim, mask.value);
        }
    }

    void forward() {
        int count = batch.size();
        std::vector<dtype*> xs, ys;
        xs.reserve(count);
        ys.reserve(count);
        drop_mask.init(getDim(), count);
        for (Node *n : batch) {
            DropoutNode *dropout_node = static_cast<DropoutNode*>(n);
#if TEST_CUDA
            dropout_node->getInput()->val().copyFromHostToDevice();
#endif
            xs.push_back(dropout_node->getInput()->getVal().value);
            ys.push_back(dropout_node->getVal().value);
        }

        CalculateDropMask(count, getDim(), drop_mask);
        n3ldg_cuda::DropoutForward(xs, count, getDim(), isTraining(), drop_mask.value,
                dropoutValue(), ys);
#if TEST_CUDA
        if (isTraining()) {
            drop_mask.copyFromDeviceToHost();
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < getDim(); ++j) {
                    dtype v = drop_mask[i][j];
                    static_cast<DropoutNode*>(batch.at(i))->dropMask()[j] = v <= dropoutValue() ?
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
            DropoutNode *dropout_node = static_cast<DropoutNode*>(n);
#if TEST_CUDA
            dropout_node->loss().copyFromHostToDevice();
            dropout_node->getInput()->loss().copyFromHostToDevice();
#endif
            vals.push_back(dropout_node->val().value);
            losses.push_back(dropout_node->loss().value);
            in_losses.push_back(dropout_node->getInput()->loss().value);
        }
        n3ldg_cuda::DropoutBackward(losses, vals, count, getDim(), isTraining(), drop_mask.value,
                dropoutValue(), in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            DropoutNode *dropout_node = static_cast<DropoutNode*>(n);
            n3ldg_cuda::Assert(dropout_node->getInput()->loss().verify("DropoutExecutor backward"));
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
    return new DropoutExecutor();
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

class BatchedMaxScalarNode : public BatchedNodeImpl<MaxScalarNode> {
public:
    void init(Graph &graph, BatchedNode &input, int input_col) {
        allocateBatch(input_col, input.batch().size());
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
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
    return new MaxScalarExecutor();
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

class BatchedScalarToVectorNode : public BatchedNodeImpl<ScalarToVectorNode> {
public:
    void init(Graph &graph, BatchedNode &input, int row) {
        allocateBatch(input.getDim() * row, input.batch().size());
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }

    void init(Graph &graph, BatchedNode &input, const vector<int> &rows) {
        vector<int> dims;
        for (int row : rows) {
            dims.push_back(row * input.getDim());
        }
        allocateBatch(dims);
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
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
    return new ScalarToVectorExecutor();
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
};

class BatchedExpNode : public BatchedNodeImpl<ExpNode> {
public:
    void init(Graph &graph, BatchedNode &input) {
        allocateBatch(input.getDims());
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
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

class BatchedSumNode : public BatchedNodeImpl<SumNode> {
public:
    void init(Graph &graph, BatchedNode &input, dtype dim) {
        allocateBatch(dim, input.batch().size());
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
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
            n3ldg_cuda::Assert(node->loss().verify("input loss"));
            node->loss().copyFromDeviceToHost();
#endif
            losses.push_back(node->getLoss().value);
            SumNode *sum = static_cast<SumNode*>(node);
            input_losses.push_back(sum->getInput()->getLoss().value);
        }

        n3ldg_cuda::VectorSumBackward(losses, batch.size(), getDim(), dims_, input_losses);
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
    return new SumExecutor();
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

class BatchedScaledNode : public BatchedNodeImpl<ScaledNode> {
public:
    void init(Graph &graph, BatchedNode &input, const vector<dtype> &factors) {
        const auto &dims = input.getDims();
        allocateBatch(dims);

        int i = 0;
        for (Node *node : batch()) {
            ScaledNode *s = dynamic_cast<ScaledNode *>(node);
            s->setFactor(factors.at(i++));
        }

        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
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

BatchedNode *maxScalar(Graph &graph, BatchedNode &input, int input_col) {
    BatchedMaxScalarNode *node = new BatchedMaxScalarNode;
    node->init(graph, input, input_col);
    return node;
};

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

BatchedNode *relu(Graph &graph, BatchedNode &input) {
    BatchedReluNode *node = new BatchedReluNode;
    node->init(graph, input);
    return node;
}

Node *sqrt(Graph &graph, Node &input) {
    SqrtNode *result = SqrtNode::newNode(input.getDim());
    result->forward(graph, input);
    return result;
}

BatchedNode *sqrt(Graph &graph, BatchedNode &input) {
    BatchedSqrtNode *node = new BatchedSqrtNode;
    node->init(graph, input);
    return node;
}

Node *scalarToVector(Graph &graph, Node &input, int row) {
    ScalarToVectorNode *node = ScalarToVectorNode::newNode(row * input.getDim());
    node->setColumn(input.getDim());
    node->forward(graph, input);
    return node;
}

BatchedNode *scalarToVector(Graph &graph, BatchedNode &input, int row) {
    BatchedScalarToVectorNode *node = new BatchedScalarToVectorNode;
    node->init(graph, input, row);
    return node;
}

BatchedNode *scalarToVector(Graph &graph, BatchedNode &input, const vector<int> &rows) {
    BatchedScalarToVectorNode *node = new BatchedScalarToVectorNode;
    node->init(graph, input, rows);
    return node;
}

Node *vectorSum(Graph &graph, Node &input,  int input_col) {
    SumNode *sum = SumNode::newNode(input_col);
    sum->forward(graph, input);
    return sum;
}

BatchedNode *vectorSum(Graph &graph, BatchedNode &input,  int input_col) {
    BatchedSumNode *node = new BatchedSumNode;
    node->init(graph, input, input_col);
    return node;
}

Node *exp(Graph &graph, Node &input) {
    ExpNode *node = ExpNode::newNode(input.getDim());
    node->forward(graph, input);
    return node;
}

BatchedNode *exp(Graph &graph, BatchedNode &input) {
    BatchedExpNode *node = new BatchedExpNode;
    node->init(graph, input);
    return node;
}

Node *dropout(Graph &graph, Node &input, dtype dropout, bool is_training) {
    DropoutNode *node = DropoutNode::newNode(input.getDim());
    node->setIsTraining(is_training);
    node->setDropValue(dropout);
    node->forward(graph, input);
    return node;
}

BatchedNode *dropout(Graph &graph, BatchedNode &input, dtype dropout, bool is_training) {
    BatchedDropoutNode *node = new BatchedDropoutNode;
    node->init(graph, input, dropout, is_training);
    return node;
}

Node *scaled(Graph &graph, Node &input, dtype factor) {
    ScaledNode *node = ScaledNode::newNode(input.getDim());
    node->setFactor(factor);
    node->forward(graph, input);
    return node;
}

BatchedNode *scaled(Graph &graph, BatchedNode &input, const vector<dtype> &factors) {
    BatchedScaledNode *node = new BatchedScaledNode;
    node->init(graph, input, factors);
    return node;
}

BatchedNode *scaled(Graph &graph, BatchedNode &input, dtype factor) {
    vector<dtype> factors;
    factors.reserve(input.batch().size());
    for (int i = 0; i < input.batch().size(); ++i) {
        factors.push_back(factor);
    }
    return scaled(graph, input, factors);
}

}


#endif
