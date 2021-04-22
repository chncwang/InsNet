#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/def.h"

using std::ptr_fun;
using std::string;
using std::to_string;
using std::vector;
using std::cerr;
using std::cout;

namespace n3ldg_plus {

#if USE_GPU
template<ActivatedEnum activation>
class ActivationExecutor : public UniInputExecutor {
public:
    vector<dtype*> vals;

    void forward() override {
        vals.reserve(batch.size());
        dims_.reserve(batch.size());
        vector<dtype*> inputs(batch.size());
        int i = 0;
        for (Node *node : batch) {
            UniInputNode *expnode = dynamic_cast<UniInputNode*>(node);
            inputs.at(i++) = expnode->getInput().getVal().value;
            vals.push_back(expnode->getVal().value);
            dims_.push_back(node->getDim());
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
        UniInputExecutor::testBeforeBackward();
        for (Node *node : batch) {
            node->loss().copyFromHostToDevice();
            UniInputNode *i = dynamic_cast<UniInputNode *>(node);
            i->getInput().loss().copyFromHostToDevice();
            i->val().copyFromHostToDevice();
        }
#endif

        int i = 0;
        for (Node *node : batch) {
            UniInputNode *exp = dynamic_cast<UniInputNode*>(node);
            losses.at(i) = exp->getLoss().value;
            input_losses.at(i++) = exp->getInput().getLoss().value;
        }

        cuda::ActivationBackward(activation, losses, vals, batch.size(), dims_,
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
        val().vec() = getInput().val().vec().unaryExpr(ptr_fun(ftanh));
    }

    void backward() override {
        getInput().loss().vec() += loss().vec() * getInput().val().vec().binaryExpr(val().vec(),
                ptr_fun(dtanh));
    }

    Executor *generate() override;

    string typeSignature() const override {
        return Node::getNodeType();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }
};

Executor *TanhNode::generate() {
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
        val().vec() = getInput().val().vec().unaryExpr(ptr_fun(fsigmoid));
    }

    void backward() override {
        getInput().loss().vec() += loss().vec() * getInput().val().vec().binaryExpr(val().vec(),
                ptr_fun(dsigmoid));
    }

    Executor *generate() override {
        return new ActivationExecutor<ActivatedEnum::SIGMOID>;
    }

    string typeSignature() const override {
        return Node::getNodeType();
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
        val().vec() = getInput().val().vec().unaryExpr(ptr_fun(frelu));
    }

    void backward() override {
        getInput().loss().vec() += loss().vec() * getInput().val().vec().binaryExpr(val().vec(),
                ptr_fun(drelu));
    }

    Executor *generate() override {
        return new ActivationExecutor<ActivatedEnum::RELU>;
    }

    string typeSignature() const override {
        return Node::getNodeType();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }
};

class BatchedReluNode : public BatchedNodeImpl<ReluNode> {
public:
    void init(BatchedNode &input) {
        allocateBatch(input.getDim(), input.batch().size());
        setInputsPerNode({&input});
        afterInit({&input});
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
        val().vec() = getInput().val().vec().unaryExpr(ptr_fun(fsqrt));
    }

    void backward() override {
        getInput().loss().vec() += loss().vec() * val().vec().unaryExpr(ptr_fun(dsqrt));
    }

    string typeSignature() const override {
        return Node::getNodeType();
    }

    Executor *generate() override {
        return new ActivationExecutor<ActivatedEnum::SQRT>;
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }
};

class BatchedSqrtNode : public BatchedNodeImpl<SqrtNode> {
public:
    void init(BatchedNode &input) {
        allocateBatch(input.getDim(), input.batch().size());
        setInputsPerNode({&input});
        afterInit({&input});
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
#if !USE_GPU || TEST_CUDA
        drop_mask_.init(dimm);
#endif
    }

#if !USE_GPU || TEST_CUDA
    virtual void generate_dropmask() {
        int dropNum = (int)(getDim() * drop_value_);
        vector<int> tmp_masks(getDim());
        for (int idx = 0; idx < getDim(); idx++) {
            tmp_masks[idx] = idx < dropNum ? 0 : 1;
        }
        random_shuffle(tmp_masks.begin(), tmp_masks.end());
        for (int idx = 0; idx < getDim(); idx++) {
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
        val().vec() = getInput().val().vec() * drop_mask_.vec();
    }

    void backward() override {
        getInput().loss().vec() += loss().vec() * drop_mask_.vec();
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

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();;
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
        allocateBatch(input.getDim(), input.batch().size());
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
            dim_sum_ += dropout_node.getDim();
            dims_.push_back(dropout_node.getDim());
        }

        drop_mask.init(dim_sum_);
        int i = 0;
        for (Node *n : batch) {
            DropoutNode *dropout_node = dynamic_cast<DropoutNode*>(n);
#if TEST_CUDA
            dropout_node->getInput().val().copyFromHostToDevice();
#endif
            xs.at(i) = dropout_node->getInput().getVal().value;
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
                for (int j = 0; j < batch.at(i)->getDim(); ++j) {
                    dtype v = drop_mask[offset + j];
                    dynamic_cast<DropoutNode*>(batch.at(i))->dropMask()[j] = v <= dropoutValue() ?
                        0 : 1;
                }
                offset += batch.at(i)->getDim();
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            cuda::Assert(batch.at(idx)->val().verify("Dropout forward"));
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
            dropout_node->loss().copyFromHostToDevice();
            dropout_node->getInput().loss().copyFromHostToDevice();
#endif
            losses.at(i) = dropout_node->loss().value;
            in_losses.at(i++) = dropout_node->getInput().loss().value;
        }
        cuda::DropoutBackward(losses, count, dims_, max_dim_, offsets_, isTraining(),
                drop_mask.value, dropoutValue(), in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            DropoutNode *dropout_node = dynamic_cast<DropoutNode*>(n);
            cuda::Assert(dropout_node->getInput().loss().verify("DropoutExecutor backward"));
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
        int input_row = getInput().getDim() / getDim();
        for (int i = 0; i < getDim(); ++i) {
            float max = getInput().getVal()[input_row * i];
            int max_i = 0;
            for (int j = 1; j < input_row; ++j) {
                if (getInput().getVal()[input_row * i + j] > max) {
                    max = getInput().getVal()[input_row * i + j];
                    max_i = j;
                }
            }
            max_indexes_.push_back(max_i);
            val()[i] = max;
        }
    }

    void backward() override {
        int input_row = getInput().getDim() / getDim();
        for (int i = 0; i < getDim(); ++i) {
            getInput().loss()[i * input_row + max_indexes_.at(i)] += getLoss()[i];
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
    void init(BatchedNode &input, int input_col) {
        allocateBatch(input_col, input.batch().size());
        setInputsPerNode({&input});
        afterInit({&input});
    }
};

#if USE_GPU
class MaxScalarExecutor : public UniInputExecutor {
public:
    void forward() override {
#if TEST_CUDA
        testForwardInpputs();
        for (Node *node : batch) {
            MaxScalarNode *m = dynamic_cast<MaxScalarNode *>(node);
            m->getInput().val().copyFromHostToDevice();
        }
#endif
        vector<dtype*> inputs(batch.size());
        vector<dtype*> results(batch.size());
        max_indexes.resize(batch.size() * batch.front()->getDim());
        vector<int> head_dims(batch.size());
        int dim = Executor::getDim();
        for (int i = 0; i < batch.size(); ++i) {
            MaxScalarNode *node = dynamic_cast<MaxScalarNode*>(batch.at(i));
            inputs.at(i) = node->getInput().getVal().value;
            results.at(i) = node->getVal().value;
            int head_dim = node->getInput().getDim() / dim;
            if (head_dim * dim != node->getInput().getDim()) {
                cerr << fmt::format("MaxScalarExecutor forward head_dim:{} dim:{} input dim:{}\n",
                        head_dim, dim, node->getInput().getDim());
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
            losses.at(i) = max_scalar->getLoss().value;
            input_losses.at(i++) = max_scalar->getInput().getLoss().value;
        }

        cuda::MaxScalarBackward(losses, max_indexes, batch.size(), input_losses);
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
        return getNodeType() + to_string(getInput().getDim());
    }

    void compute() override {
        for (int i = 0; i < getColumn(); ++i) {
            for (int j = 0; j < getRow(); ++j) {
                val()[i * getRow() + j] = getInput().getVal()[i];
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
            getInput().loss()[i] += sum;
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
    void init(BatchedNode &input, int row) {
        allocateBatch(input.getDim() * row, input.batch().size());
        setInputsPerNode({&input});
        afterInit({&input});
    }

    void init(BatchedNode &input, const vector<int> &rows) {
        vector<int> dims(rows.size());
        int i = 0;
        for (int row : rows) {
            dims.at(i++) = row * input.getDim();
        }
        allocateBatch(dims);
        setInputsPerNode({&input});
        afterInit({&input});
    }
};

#if USE_GPU
class ScalarToVectorExecutor : public UniInputExecutor {
public:
    void forward() override {
#if TEST_CUDA
        UniInputExecutor::testForwardInpputs();
#endif
        vector<dtype*> inputs(batch.size());
        vector<dtype*> results(batch.size());
        dims_.reserve(batch.size());
        int i = 0;
        for (Node *node : batch) {
            ScalarToVectorNode *n = dynamic_cast<ScalarToVectorNode*>(node);
            inputs.at(i) = n->getInput().getVal().value;
            results.at(i++) = n->getVal().value;
            dims_.push_back(n->getDim() / n->getInput().getDim());
        }
        int dim = dynamic_cast<ScalarToVectorNode *>(batch.front())->getInput().getDim();
        cuda::ScalarToVectorForward(inputs, batch.size(), dim, dims_, results);
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
            ScalarToVectorNode * n = dynamic_cast<ScalarToVectorNode*>(node);
            n->loss().copyFromHostToDevice();
            n->getInput().loss().copyFromHostToDevice();
        }
#endif
        vector<dtype*> losses(batch.size());
        vector<dtype*> input_losses(batch.size());
        int i = 0;
        for (Node *node : batch) {
            ScalarToVectorNode * n = dynamic_cast<ScalarToVectorNode*>(node);
            losses.at(i) = n->getLoss().value;
            input_losses.at(i++) = n->getInput().getLoss().value;
        }
        int dim = dynamic_cast<ScalarToVectorNode *>(batch.front())->getInput().getDim();
        cuda::ScalarToVectorBackward(losses, batch.size(), dim, dims_, input_losses);
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
        val().vec() = getInput().getVal().vec().exp();
    }

    void backward() override {
        getInput().loss().vec() += getLoss().vec() * getVal().vec();
    }

protected:
    bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }
};

class BatchedExpNode : public BatchedNodeImpl<ExpNode> {
public:
    void init(BatchedNode &input) {
        allocateBatch(input.getDims());
        setInputsPerNode({&input});
        afterInit({&input});
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
            int input_row = getInput().getDim() / getDim();
            for (int j = 0; j < input_row; ++j) {
                sum += getInput().getVal()[i * input_row + j];
            }
            val()[i] = sum;
        }
    }

    void backward() override {
        for (int i = 0; i < getDim(); ++i) {
            int input_row = getInput().getDim() / getDim();
            for (int j = 0; j < input_row; ++j) {
                getInput().loss()[i * input_row + j] += getLoss()[i];
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
    void init(BatchedNode &input, dtype dim) {
        allocateBatch(dim, input.batch().size());
        setInputsPerNode({&input});
        afterInit({&input});
    }
};

#if USE_GPU
class SumExecutor : public UniInputExecutor {
    void forward() override {
        vector<dtype*> inputs(batch.size());
        vector<dtype*> results(batch.size());
        dims_.reserve(batch.size());
        int i = 0;
        for (Node *node : batch) {
            SumNode *sum = dynamic_cast<SumNode*>(node);
            inputs.at(i) = sum->getInput().getVal().value;
            results.at(i++) = sum->getVal().value;
            int row = sum->getInput().getDim()/ getDim();
            if (row * getDim() != sum->getInput().getDim()) {
                cerr << fmt::format("SumExecutor forward row:{} dim:{} input dim:{}\n", row,
                        getDim(), sum->getInput().getDim());
                abort();
            }
            dims_.push_back(row);
        }
        cuda::VectorSumForward(inputs, batch.size(), getDim(), dims_, results);
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
            cuda::Assert(node->loss().verify("input loss"));
            node->loss().copyFromDeviceToHost();
#endif
            losses.at(i) = node->getLoss().value;
            SumNode *sum = dynamic_cast<SumNode*>(node);
            input_losses.at(i++) = sum->getInput().getLoss().value;
        }

        cuda::VectorSumBackward(losses, batch.size(), getDim(), dims_, input_losses);
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
        val().vec() = factor_ * getInput().getVal().vec();
    }

    void backward() override {
        getInput().loss().vec() += factor_ * getLoss().vec();
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
    void init(BatchedNode &input, const vector<dtype> &factors) {
        const auto &dims = input.getDims();
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
class ScaledExecutor : public UniInputExecutor {
public:
    void forward() override {
        vector<dtype *> in_vals(batch.size());
        dims.reserve(batch.size());
        factors.reserve(batch.size());
        int i = 0;
        for (Node *node : batch) {
            ScaledNode *scaled = dynamic_cast<ScaledNode *>(node);
            in_vals.at(i++) = scaled->getInput().getVal().value;
            dims.push_back(scaled->getDim());
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
            in_grads.at(i++) = scaled->getInput().getLoss().value;
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

Node *maxScalar(Node &input, int input_col) {
    MaxScalarNode *node = MaxScalarNode::newNode(input_col);
    node->connect(input);
    return node;
}

BatchedNode *maxScalar(BatchedNode &input, int input_col) {
    BatchedMaxScalarNode *node = new BatchedMaxScalarNode;
    node->init(input, input_col);
    return node;
};

Node *tanh(Node &input) {
    TanhNode *result = TanhNode::newNode(input.getDim());
    result->connect(input);
    return result;
}

Node *sigmoid(Node &input) {
    SigmoidNode *result = SigmoidNode::newNode(input.getDim());
    result->connect(input);
    return result;
}

Node *relu(Node &input) {
    ReluNode *result = ReluNode::newNode(input.getDim());
    result->connect(input);
    return result;
}

BatchedNode *relu(BatchedNode &input) {
    BatchedReluNode *node = new BatchedReluNode;
    node->init(input);
    return node;
}

Node *sqrt(Node &input) {
    SqrtNode *result = SqrtNode::newNode(input.getDim());
    result->connect(input);
    return result;
}

BatchedNode *sqrt(BatchedNode &input) {
    BatchedSqrtNode *node = new BatchedSqrtNode;
    node->init(input);
    return node;
}

Node *scalarToVector(Node &input, int row) {
    ScalarToVectorNode *node = ScalarToVectorNode::newNode(row * input.getDim());
    node->setColumn(input.getDim());
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
    ExpNode *node = ExpNode::newNode(input.getDim());
    node->connect(input);
    return node;
}

BatchedNode *exp(BatchedNode &input) {
    BatchedExpNode *node = new BatchedExpNode;
    node->init(input);
    return node;
}

Node *dropout(Node &input, dtype dropout) {
    DropoutNode *node = DropoutNode::newNode(input.getDim());
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
    ScaledNode *node = ScaledNode::newNode(input.getDim());
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
