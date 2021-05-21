#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/def.h"

using std::ptr_fun;
using std::string;
using std::to_string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using std::shared_ptr;

namespace n3ldg_plus {

#if USE_GPU
template<ActivatedEnum activation>
class ActivationExecutor : public Executor {
public:
    vector<dtype*> vals;

    void forward() override {
        vals.reserve(batch.size());
        dims_.reserve(batch.size());
        vector<dtype*> inputs(batch.size());
        int i = 0;
        for (Node *node : batch) {
            UniInputNode *expnode = dynamic_cast<UniInputNode*>(node);
            inputs.at(i++) = expnode->inputVal().value;
            vals.push_back(expnode->getVal().value);
            dims_.push_back(node->size());
        }
        cuda::ActivationForward(activation, inputs, batch.size(), dims_, vals);
#if TEST_CUDA
        Executor::testForward();
        cout << "exp forward tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype*> losses(batch.size());
        vector<dtype*> input_losses(batch.size());
#if TEST_CUDA
        Executor::testBeforeBackward();
        for (Node *node : batch) {
            node->grad().copyFromHostToDevice();
            UniInputNode *i = dynamic_cast<UniInputNode *>(node);
            i->inputGrad().copyFromHostToDevice();
            i->val().copyFromHostToDevice();
        }
#endif

        int i = 0;
        for (Node *node : batch) {
            UniInputNode *exp = dynamic_cast<UniInputNode*>(node);
            losses.at(i) = exp->getGrad().value;
            input_losses.at(i++) = exp->inputGrad().value;
        }

        cuda::ActivationBackward(activation, losses, vals, batch.size(), dims_,
                input_losses);
#if TEST_CUDA
        Executor::testBackward();
        cout << "exp backward tested" << endl;
#endif
    }

private:
    vector<int> dims_;
};
#else
template<ActivatedEnum activation>
class ActivationExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return defaultFLOPs();
    }

};
#endif

class TanhNode : public UniInputNode, public Poolable<TanhNode> {
public:
    TanhNode() : UniInputNode("tanh") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void compute() override {
        val().vec() = inputVal().vec().tanh();
    }

    void backward() override {
        inputGrad().vec() += grad().vec() * getVal().vec().unaryExpr(ptr_fun(dtanh));
    }

    Executor *generate() override;

    string typeSignature() const override {
        return Node::getNodeType();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.size() == size();
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return false;
    }
};

Executor *TanhNode::generate() {
    return new ActivationExecutor<ActivatedEnum::TANH>;
};


class SigmoidNode :public UniInputNode, public Poolable<SigmoidNode> {
public:
    SigmoidNode() : UniInputNode("sigmoid") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void compute() override {
        val().vec() = inputVal().vec().unaryExpr(ptr_fun(fsigmoid));
    }

    void backward() override {
        inputGrad().vec() += grad().vec() * getVal().vec().unaryExpr(ptr_fun(dsigmoid));
    }

    Executor *generate() override {
        return new ActivationExecutor<ActivatedEnum::SIGMOID>;
    }

    string typeSignature() const override {
        return Node::getNodeType();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.size() == size();
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return false;
    }
};

class ReluNode :public UniInputNode, public Poolable<ReluNode> {
public:
    ReluNode() : UniInputNode("relu") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void compute() override {
        val().vec() = inputVal().vec().unaryExpr(ptr_fun(frelu));
    }

    void backward() override {
        inputGrad().vec() += grad().vec() * val().vec().unaryExpr(ptr_fun(drelu));
    }

    Executor *generate() override {
        return new ActivationExecutor<ActivatedEnum::RELU>;
    }

    string typeSignature() const override {
        return Node::getNodeType();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.size() == size();
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return false;
    }
};

class SqrtNode :public UniInputNode, public Poolable<SqrtNode> {
public:
    SqrtNode() : UniInputNode("sqrt") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void compute() override {
        val().vec() = inputVal().vec().unaryExpr(ptr_fun(fsqrt));
    }

    void backward() override {
        inputGrad().vec() += grad().vec() * val().vec().unaryExpr(ptr_fun(dsqrt));
    }

    string typeSignature() const override {
        return Node::getNodeType();
    }

    Executor *generate() override {
        return new ActivationExecutor<ActivatedEnum::SQRT>;
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.size() == size();
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return false;
    }
};

class BatchedSqrtNode : public BatchedNodeImpl<SqrtNode> {
public:
    void init(BatchedNode &input) {
        allocateBatch(input.size(), input.batch().size());
        setInputsPerNode({&input});
        afterInit({&input});
    }
};

class DropoutNode : public UniInputNode, public Poolable<DropoutNode> {
public:
    DropoutNode() : UniInputNode("dropout") {}

    void setNodeDim(int dim) override {
        Node::setDim(dim);
    }

    void init(int dimm) {
#if !USE_GPU || TEST_CUDA
        drop_mask_.init(dimm);
#endif
    }

#if !USE_GPU || TEST_CUDA
    virtual void generate_dropmask() {
        int dropNum = (int)(size() * drop_value_);
        vector<int> tmp_masks(size());
        for (int idx = 0; idx < size(); idx++) {
            tmp_masks[idx] = idx < dropNum ? 0 : 1;
        }
        random_shuffle(tmp_masks.begin(), tmp_masks.end());
        for (int idx = 0; idx < size(); idx++) {
            drop_mask_[idx] = tmp_masks[idx];
        }
    }
#endif

#if !USE_GPU || TEST_CUDA
    void compute() override {
        if (isTraining()) {
#if !TEST_CUDA
            generate_dropmask();
#endif
        } else {
            drop_mask_ = 1 - drop_value_;
        }
        val().vec() = inputVal().vec() * drop_mask_.vec();
    }

    void backward() override {
        inputGrad().vec() += grad().vec() * drop_mask_.vec();
    }
#else
    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }
#endif

    string typeSignature() const override {
        return Node::getNodeType() + "-" + to_string(drop_value_);
    }

    Executor *generate() override;

    bool isTraining() {
        return getNodeContainer().getModelStage() == ModelStage::TRAINING;
    }

#if !USE_GPU || TEST_CUDA
    Tensor1D &dropMask() {
        return drop_mask_;
    }
#endif

    void setDropValue(dtype drop_value) {
        drop_value_ = drop_value;
    }

    dtype dropoutValue() const {
        return drop_value_;
    }

    void clear() override {
#if !USE_GPU || TEST_CUDA
        drop_mask_.releaseMemory();
#endif
        Node::clear();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.size() == size();;
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
#if !USE_GPU || TEST_CUDA
    Tensor1D drop_mask_;
#endif
    dtype drop_value_ = 0.0f;
};

class BatchedDropoutNode : public BatchedNodeImpl<DropoutNode> {
public:
    void init(BatchedNode &input, dtype dropout) {
        allocateBatch(input.size(), input.batch().size());
        for (Node *node : batch()) {
            DropoutNode *d = dynamic_cast<DropoutNode *>(node);
            d->setDropValue(dropout);
        }
        setInputsPerNode({&input});
        afterInit({&input});
    }
};

class DropoutExecutor :public Executor {
public:
    bool isTraining() const {
        DropoutNode *node = dynamic_cast<DropoutNode *>(batch.front());
        return node->isTraining();
    }

    dtype dropoutValue() const {
        DropoutNode *node = dynamic_cast<DropoutNode *>(batch.front());
        return node->dropoutValue();
    }

#if USE_GPU
    void CalculateDropMask() {
        if (isTraining()) {
            cuda::CalculateDropoutMask(dropoutValue(), dim_sum_, drop_mask.value);
        }
    }

    void forward() {
        int count = batch.size();
        vector<dtype*> xs(count), ys(count);
        dims_.reserve(count);
        offsets_.reserve(count);

        for (Node *node : batch) {
            DropoutNode &dropout_node = dynamic_cast<DropoutNode&>(*node);
            offsets_.push_back(dim_sum_);
            dim_sum_ += dropout_node.size();
            dims_.push_back(dropout_node.size());
        }

        drop_mask.init(dim_sum_);
        int i = 0;
        for (Node *n : batch) {
            DropoutNode *dropout_node = dynamic_cast<DropoutNode*>(n);
#if TEST_CUDA
            dropout_node->inputVal().copyFromHostToDevice();
#endif
            xs.at(i) = dropout_node->inputVal().value;
            ys.at(i++) = dropout_node->getVal().value;
        }

        CalculateDropMask();
        max_dim_ = *max_element(dims_.begin(), dims_.end());
        cuda::DropoutForward(xs, count, dims_, max_dim_, offsets_, isTraining(),
                drop_mask.value, dropoutValue(), ys);
#if TEST_CUDA
        if (isTraining()) {
            drop_mask.copyFromDeviceToHost();
            int offset = 0;
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < batch.at(i)->size(); ++j) {
                    dtype v = drop_mask[offset + j];
                    dynamic_cast<DropoutNode*>(batch.at(i))->dropMask()[j] = v < dropoutValue() ?
                        0 : 1;
                }
                offset += batch.at(i)->size();
            }
        }
        Executor::forward();

        i = 0;
        int offset = 0;
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            cout << fmt::format("i:{} dim:{}", i, node->size()) << endl;
            if(!x->getVal().verify((getNodeType() + " forward").c_str())) {
                cout << "cpu:" << endl;
                cout << x->getVal().toString();
                cout << "gpu:" << endl;
                x->getVal().print();
                cout << "dropout mask:" << endl;
                for (int j = 0; j < node->size(); ++j) {
                    cout << j << " " << drop_mask[offset + j] << endl;
                }
                throw cuda::CudaVerificationException(i);
            }
            offset += node->size();
            ++i;
        }
#endif
    }

    void backward() {
        int count = batch.size();
        vector<dtype*> losses(count), in_losses(count);
        int i = 0;
        for (Node *n : batch) {
            DropoutNode *dropout_node = dynamic_cast<DropoutNode*>(n);
#if TEST_CUDA
            dropout_node->grad().copyFromHostToDevice();
            dropout_node->inputGrad().copyFromHostToDevice();
#endif
            losses.at(i) = dropout_node->grad().value;
            in_losses.at(i++) = dropout_node->inputGrad().value;
        }
        cuda::DropoutBackward(losses, count, dims_, max_dim_, offsets_, isTraining(),
                drop_mask.value, dropoutValue(), in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            DropoutNode *dropout_node = dynamic_cast<DropoutNode*>(n);
            cuda::Assert(dropout_node->inputGrad().verify("DropoutExecutor backward"));
        }
#endif
    }
#else
    int calculateFLOPs() override {
        return defaultFLOPs();
    }
#endif

private:
    Tensor1D drop_mask;
    vector<int> dims_, offsets_;
    int dim_sum_ = 0;
    int max_dim_;
};

Executor *DropoutNode::generate() {
    return new DropoutExecutor();
}

class MaxScalarNode : public UniInputNode, public Poolable<MaxScalarNode> {
public:
    MaxScalarNode() : UniInputNode("max_scalar_node") {}

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
        int input_row = inputDim() / size();
        for (int i = 0; i < size(); ++i) {
            float max = inputVal()[input_row * i];
            int max_i = 0;
            for (int j = 1; j < input_row; ++j) {
                if (inputVal()[input_row * i + j] > max) {
                    max = inputVal()[input_row * i + j];
                    max_i = j;
                }
            }
            max_indexes_.push_back(max_i);
            val()[i] = max;
        }
    }

    void backward() override {
        int input_row = inputGrad().dim / size();
        for (int i = 0; i < size(); ++i) {
            inputGrad()[i * input_row + max_indexes_.at(i)] += getGrad()[i];
        }
    }

    Executor* generate() override;

protected:
    bool isDimLegal(const Node &input) const override {
        return input.size() % size() == 0;
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    vector<int> max_indexes_;
    friend class MaxScalarExecutor;
};

#if USE_GPU
class MaxScalarExecutor : public Executor {
public:
    void forward() override {
#if TEST_CUDA
        testForwardInpputs();
        for (Node *node : batch) {
            MaxScalarNode *m = dynamic_cast<MaxScalarNode *>(node);
            m->inputVal().copyFromHostToDevice();
        }
#endif
        vector<dtype*> inputs(batch.size());
        vector<dtype*> results(batch.size());
        max_indexes.resize(batch.size() * batch.front()->size());
        vector<int> head_dims(batch.size());
        int dim = Executor::size();
        for (int i = 0; i < batch.size(); ++i) {
            MaxScalarNode *node = dynamic_cast<MaxScalarNode*>(batch.at(i));
            inputs.at(i) = node->inputVal().value;
            results.at(i) = node->getVal().value;
            int head_dim = node->inputDim() / dim;
            if (head_dim * dim != node->inputDim()) {
                cerr << fmt::format("MaxScalarExecutor forward head_dim:{} dim:{} input dim:{}\n",
                        head_dim, dim, node->inputDim());
                abort();
            }
            head_dims.at(i) = head_dim;
        }
        cuda::MaxScalarForward(inputs, batch.size(), dim, head_dims, results, &max_indexes);

#if TEST_CUDA
        testForward();
        cout << "max scalar forward tested:" << endl;
#endif
    }

    void backward() override {
        vector<dtype*> losses(batch.size());
        vector<dtype *> input_losses(batch.size());

        int i = 0;
        for (Node *node : batch) {
            MaxScalarNode *max_scalar = dynamic_cast<MaxScalarNode*>(node);
            losses.at(i) = max_scalar->getGrad().value;
            input_losses.at(i++) = max_scalar->inputGrad().value;
        }

        cuda::MaxScalarBackward(losses, max_indexes, batch.size(), input_losses);
#if TEST_CUDA
        Executor::testBackward();
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

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    string typeSignature() const override {
        return getNodeType() + to_string(getInputVal().dim);
    }

    void compute() override {
        for (int i = 0; i < getColumn(); ++i) {
            for (int j = 0; j < getRow(); ++j) {
                val()[i * getRow() + j] = inputVal()[i];
            }
        }
    }

    void backward() override {
        int row = getRow();
        for (int i = 0; i < getColumn(); ++i) {
            dtype sum = 0;
            for (int j = 0; j < row; ++j) {
                sum += getGrad()[i * row + j];
            }
            inputGrad()[i] += sum;
        }
    }

    Executor* generate() override;

protected:
    bool isDimLegal(const Node &input) const override {
        return true;
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    friend class ScalarToVectorExecutor;
};

class BatchedScalarToVectorNode : public BatchedNodeImpl<ScalarToVectorNode> {
public:
    void init(BatchedNode &input, int row) {
        allocateBatch(input.size() * row, input.batch().size());
        setInputsPerNode({&input});
        afterInit({&input});
    }

    void init(BatchedNode &input, const vector<int> &rows) {
        vector<int> dims(rows.size());
        int i = 0;
        for (int row : rows) {
            dims.at(i++) = row * input.size();
        }
        allocateBatch(dims);
        setInputsPerNode({&input});
        afterInit({&input});
    }
};

#if USE_GPU
class ScalarToVectorExecutor : public Executor {
public:
    void forward() override {
#if TEST_CUDA
        Executor::testForwardInpputs();
#endif
        vector<dtype*> inputs(batch.size());
        vector<dtype*> results(batch.size());
        dims_.reserve(batch.size());
        int i = 0;
        int dim;
        for (Node *node : batch) {
            ScalarToVectorNode *n = dynamic_cast<ScalarToVectorNode*>(node);
            inputs.at(i) = n->inputVal().value;
            results.at(i++) = n->getVal().value;
            dims_.push_back(n->size() / n->getInputVal().dim);
            dim = n->getInputVal().dim;
        }
        cuda::ScalarToVectorForward(inputs, batch.size(), dim, dims_, results);
#if TEST_CUDA
        Executor::testForward();
        cout << "scalarToVector tested" << endl;
#endif
    }

    void backward() override {
#if TEST_CUDA
        cout << "scalarToVector test before backward..." << endl;
        Executor::testBeforeBackward();
        for (Node *node : batch) {
            ScalarToVectorNode * n = dynamic_cast<ScalarToVectorNode*>(node);
            n->grad().copyFromHostToDevice();
            n->inputGrad().copyFromHostToDevice();
        }
#endif
        vector<dtype*> losses(batch.size());
        vector<dtype*> input_losses(batch.size());
        int i = 0;
        int dim;
        for (Node *node : batch) {
            ScalarToVectorNode * n = dynamic_cast<ScalarToVectorNode*>(node);
            losses.at(i) = n->getGrad().value;
            input_losses.at(i++) = n->inputGrad().value;
            dim = n->inputGrad().dim;
        }
        cuda::ScalarToVectorBackward(losses, batch.size(), dim, dims_, input_losses);
#if TEST_CUDA
        cout << "scalarToVector test backward..." << endl;
        Executor::testBackward();
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
        val().vec() = getInputVal().vec().exp();
    }

    void backward() override {
        inputGrad().vec() += getGrad().vec() * getVal().vec();
    }

protected:
    bool isDimLegal(const Node &input) const override {
        return input.size() == size();
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return false;
    }
};

class BatchedExpNode : public BatchedNodeImpl<ExpNode> {
public:
    void init(BatchedNode &input) {
        allocateBatch(input.sizes());
        setInputsPerNode({&input});
        afterInit({&input});
    }
};

class SumNode : public UniInputNode, public Poolable<SumNode> {
public:
    SumNode(): UniInputNode("sum") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    Executor* generate() override;

    virtual string typeSignature() const override {
        return Node::typeSignature();
    }

    void compute() override {
        for (int i = 0; i < size(); ++i) {
            dtype sum = 0;
            int input_row = getInputVal().dim / size();
            for (int j = 0; j < input_row; ++j) {
                sum += getInputVal()[i * input_row + j];
            }
            val()[i] = sum;
        }
    }

    void backward() override {
        for (int i = 0; i < size(); ++i) {
            int input_row = inputGrad().dim / size();
            for (int j = 0; j < input_row; ++j) {
                inputGrad()[i * input_row + j] += getGrad()[i];
            }
        }
    }

protected:
    bool isDimLegal(const Node &input) const override {
        return input.size() % size() == 0;
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    friend class SumExecutor;
};

class BatchedSumNode : public BatchedNodeImpl<SumNode> {
public:
    void init(BatchedNode &input, dtype dim) {
        allocateBatch(dim, input.batch().size());
        setInputsPerNode({&input});
        afterInit({&input});
    }
};

#if USE_GPU
class SumExecutor : public Executor {
    void forward() override {
        vector<dtype*> inputs(batch.size());
        vector<dtype*> results(batch.size());
        dims_.reserve(batch.size());
        int i = 0;
        for (Node *node : batch) {
            SumNode *sum = dynamic_cast<SumNode*>(node);
            inputs.at(i) = sum->inputVal().value;
            results.at(i++) = sum->getVal().value;
            int row = sum->getInputVal().dim / size();
            if (row * size() != sum->getInputVal().dim) {
                cerr << fmt::format("SumExecutor forward row:{} dim:{} input dim:{}\n", row,
                        size(), sum->getInputVal().dim);
                abort();
            }
            dims_.push_back(row);
        }
        cuda::VectorSumForward(inputs, batch.size(), size(), dims_, results);
#if TEST_CUDA
        Executor::testForward();
        cout << "sum tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype*> losses(batch.size());
        vector<dtype*> input_losses(batch.size());
        int i = 0;
        for (Node *node : batch) {
#if TEST_CUDA
            cuda::Assert(node->grad().verify("input loss"));
            node->grad().copyFromDeviceToHost();
#endif
            losses.at(i) = node->getGrad().value;
            SumNode *sum = dynamic_cast<SumNode*>(node);
            input_losses.at(i++) = sum->inputGrad().value;
        }

        cuda::VectorSumBackward(losses, batch.size(), size(), dims_, input_losses);
#if TEST_CUDA
        Executor::testBackward();
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

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void setFactor(dtype factor) {
        factor_ = factor;
    }

    void compute() override {
        val().vec() = factor_ * inputVal().vec();
    }

    void backward() override {
        inputGrad().vec() += factor_ * getGrad().vec();
    }

    Executor* generate() override;

    virtual string typeSignature() const override {
        return getNodeType();
    }

protected:
    bool isDimLegal(const Node &input) const override {
        return input.size() == size();
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    dtype factor_ = 1;
    friend class ScaledExecutor;
};

class BatchedScaledNode : public BatchedNodeImpl<ScaledNode> {
public:
    void init(BatchedNode &input, const vector<dtype> &factors) {
        const auto &dims = input.sizes();
        allocateBatch(dims);

        int i = 0;
        for (Node *node : batch()) {
            ScaledNode *s = dynamic_cast<ScaledNode *>(node);
            s->setFactor(factors.at(i++));
        }

        setInputsPerNode({&input});
        afterInit({&input});
    }
};

#if USE_GPU
class ScaledExecutor : public Executor {
public:
    void forward() override {
        vector<dtype *> in_vals(batch.size());
        dims.reserve(batch.size());
        factors.reserve(batch.size());
        int i = 0;
        for (Node *node : batch) {
            ScaledNode *scaled = dynamic_cast<ScaledNode *>(node);
            in_vals.at(i++) = scaled->inputVal().value;
            dims.push_back(scaled->size());
            factors.push_back(scaled->factor_);
        }
        auto vals = getVals();
        cuda::ScaledForward(in_vals, batch.size(), dims, factors, vals);
#if TEST_CUDA
        testForward();
        cout << "ScaledExecutor forward tested" << endl;
#endif
    }

    void backward() override {
        vector<dtype *> in_grads(batch.size());
        int i = 0;
        for (Node *node : batch) {
            ScaledNode *scaled = dynamic_cast<ScaledNode *>(node);
            in_grads.at(i++) = scaled->inputGrad().value;
        }
        auto grads = getGrads();
        cuda::ScaledBackward(grads, batch.size(), dims, factors, in_grads);
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

Node *max(Node &input, int input_row) {
    int input_col = input.size() / input_row;
    if (input_col * input_row != input.size()) {
        cerr << fmt::format("input_col:{} input_row:{} dim:{}", input_col, input_row,
                input.size()) << endl;
        abort();
    }
    MaxScalarNode *node = MaxScalarNode::newNode(input_col);
    node->connect(input);
    return node;
}

Node *tanh(Node &input) {
    TanhNode *result = TanhNode::newNode(input.size());
    result->connect(input);
    return result;
}

Node *sigmoid(Node &input) {
    SigmoidNode *result = SigmoidNode::newNode(input.size());
    result->connect(input);
    return result;
}

Node *relu(Node &input) {
    ReluNode *result = ReluNode::newNode(input.size());
    result->connect(input);
    return result;
}

Node *sqrt(Node &input) {
    SqrtNode *result = SqrtNode::newNode(input.size());
    result->connect(input);
    return result;
}

BatchedNode *sqrt(BatchedNode &input) {
    BatchedSqrtNode *node = new BatchedSqrtNode;
    node->init(input);
    return node;
}

Node *scalarToVector(Node &input, int row) {
    ScalarToVectorNode *node = ScalarToVectorNode::newNode(row * input.size());
    node->setColumn(input.size());
    node->connect(input);
    return node;
}

BatchedNode *scalarToVector(BatchedNode &input, int row) {
    BatchedScalarToVectorNode *node = new BatchedScalarToVectorNode;
    node->init(input, row);
    return node;
}

BatchedNode *scalarToVector(BatchedNode &input, const vector<int> &rows) {
    BatchedScalarToVectorNode *node = new BatchedScalarToVectorNode;
    node->init(input, rows);
    return node;
}

Node *vectorSum(Node &input,  int input_col) {
    SumNode *sum = SumNode::newNode(input_col);
    sum->connect(input);
    return sum;
}

BatchedNode *vectorSum(BatchedNode &input,  int input_col) {
    BatchedSumNode *node = new BatchedSumNode;
    node->init(input, input_col);
    return node;
}

Node *exp(Node &input) {
    ExpNode *node = ExpNode::newNode(input.size());
    node->connect(input);
    return node;
}

BatchedNode *exp(BatchedNode &input) {
    BatchedExpNode *node = new BatchedExpNode;
    node->init(input);
    return node;
}

Node *dropout(Node &input, dtype dropout) {
    DropoutNode *node = DropoutNode::newNode(input.size());
    node->init(input.size());
    node->setDropValue(dropout);
    node->connect(input);
    return node;
}

BatchedNode *dropout(BatchedNode &input, dtype dropout) {
    BatchedDropoutNode *node = new BatchedDropoutNode;
    node->init(input, dropout);
    return node;
}

Node *scaled(Node &input, dtype factor) {
    ScaledNode *node = ScaledNode::newNode(input.size());
    node->setFactor(factor);
    node->connect(input);
    return node;
}

BatchedNode *scaled(BatchedNode &input, const vector<dtype> &factors) {
    BatchedScaledNode *node = new BatchedScaledNode;
    node->init(input, factors);
    return node;
}

BatchedNode *scaled(BatchedNode &input, dtype factor) {
    vector<dtype> factors(input.batch().size());
    for (int i = 0; i < input.batch().size(); ++i) {
        factors.at(i) = factor;
    }
    return scaled(input, factors);
}

}
