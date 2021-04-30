#include "n3ldg-plus/operator/layer_normalization.h"

using std::string;
using std::to_string;
using std::vector;
using std::cout;
using std::endl;

namespace n3ldg_plus {

class StandardLayerNormNode : public UniInputNode, public Poolable<StandardLayerNormNode> {
public:
    StandardLayerNormNode() : UniInputNode("standard-layernorm") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        Node::setDim(dim);
    }

    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }

    Executor *generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + to_string(getDim() / getColumn());
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return false;
    }

private:
    friend class LayerNormExecutor;
};

class BatchedStandardLayerNormNode : public BatchedNodeImpl<StandardLayerNormNode> {
public:
    void init(BatchedNode &input) {
        allocateBatch(input.getDim(), input.batch().size());
        setInputsPerNode({&input});
        afterInit({&input});
    }
};

#if USE_GPU
class LayerNormExecutor : public Executor {
public:
    void forward() override {
        int count = batch.size();
        vector<dtype *> in_vals(count), vals(count);
        vector<int> cols(count);
        int i = 0;
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            in_vals.at(i) = s.getInputVal().value;
            cols.at(i) = s.getColumn();
            vals.at(i++) = s.getVal().value;
        }
        val_arr_.init(vals.data(), count);

        cuda::NumberPointerArray in_val_arr;
        in_val_arr.init(in_vals.data(), count);
        col_arr_.init(cols.data(), count);
        max_col_ = *max_element(cols.begin(), cols.end());
        sds_.init(count * max_col_);

        cuda::StandardLayerNormForward(in_val_arr.value, count, getRow(), col_arr_.value,
                max_col_, val_arr_.value, sds_.value);
#if TEST_CUDA
        i = 0;
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            auto &input = s.inputVal();
            for (int j = 0; j < s.getColumn(); ++j) {
                int row = getRow();
                dtype mean = Mat(input.v + row * j, row, 1).sum() / row;
                Tensor1D x;
                x.init(row);
                x.vec() = (Vec(input.v + row * j, row) - mean).square();
                dtype sd = sqrt(x.mat().sum() / row);
                sds_[i++] = sd;
                Vec(s.val().v + row * j, row) = (Vec(input.v + row * j, row) - mean) / sd;
            }
        }
        verifyForward();
#endif
    }

    void backward() override {
#if TEST_CUDA
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            s.grad().verify("standard layernorm before backward grad");
            s.grad().copyFromHostToDevice();
            s.val().verify("standard layernorm before backward val");
            s.val().copyFromHostToDevice();
            s.inputGrad().verify("standard layernorm before backward input grad");
            s.inputGrad().copyFromHostToDevice();
        }
#endif
        int count = batch.size();
        vector<dtype *> in_grads(count), grads(count);
        int i = 0;
        int col_sum = 0;
        vector<int> col_offsets(count), dims(count), dim_offsets(count);
        int row = getRow();
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            col_offsets.at(i) = col_sum;
            dim_offsets.at(i) = col_sum * row;
            dims.at(i) = s.getDim();
            in_grads.at(i) = s.inputGrad().value;
            grads.at(i++) = s.getGrad().value;
            col_sum += s.getColumn();
        }
        cuda::NumberPointerArray grad_arr, in_grad_arr;
        grad_arr.init(grads.data(), count);
        in_grad_arr.init(in_grads.data(), count);
        cuda::IntArray col_offset_arr, dim_arr, dim_offset_arr;
        col_offset_arr.init(col_offsets.data(), count);
        dim_arr.init(dims.data(), count);
        dim_offset_arr.init(dim_offsets.data(), count);
        cuda::StandardLayerNormBackward(grad_arr.value, count, row, col_arr_.value, col_sum,
                max_col_, col_offset_arr.value, dim_arr.value, dim_offset_arr.value,
                val_arr_.value, sds_.value, in_grad_arr.value);
#if TEST_CUDA
        i = 0;
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            int n = getRow();
            for (int j = 0; j < s.getColumn(); ++j) {
                dtype c = 1.0 / (n * sds_[i]);
                Tensor1D y2;
                y2.init(n);
                y2.vec() = Vec(s.getVal().v + j * n, n).square();
                Tensor1D m;
                m.init(n);
                m.vec() = Vec(s.getGrad().v + j * n, n) * Vec(s.getVal().v + j * n, n);
                Tensor1D x;
                x.init(n);
                x.vec() = c * ((-y2.vec() +
                            static_cast<dtype>(n -1)) * Vec(s.getGrad().v + j * n, n) -
                        ((m.mat().sum() - m.vec()) * Vec(s.getVal().v + j * n, n) +
                         Mat(s.getGrad().v + j * n, n, 1).sum() - Vec(s.getGrad().v + j * n, n)));
                Vec(s.inputGrad().v + j * n, n) += x.vec();
                ++i;
            }
        }
        verifyBackward();
#endif
    }

private:
    Tensor1D sds_;
    cuda::NumberPointerArray val_arr_;
    cuda::IntArray col_arr_;
    int max_col_;
};
#else
class LayerNormExecutor : public Executor {
public:
    void forward() override {
        for (Node *node : batch) {
            col_sum_ += node->getColumn();
        }
        sds_.init(col_sum_);
        int i = 0;
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            auto &input = s.inputVal();
            for (int j = 0; j < s.getColumn(); ++j) {
                int row = getRow();
                dtype mean = Mat(input.v + row * j, row, 1).sum() / row;
                Tensor1D x;
                x.init(row);
                x.vec() = (Vec(input.v + row * j, row) - mean).square();
                dtype sd = sqrt(x.mat().sum() / row);
                sds_[i++] = sd;
                Vec(s.val().v + row * j, row) = (Vec(input.v + row * j, row) - mean) / sd;
            }
        }
    }

    void backward() override {
        int i = 0;
        for (Node *node : batch) {
            StandardLayerNormNode &s = dynamic_cast<StandardLayerNormNode &>(*node);
            int n = getRow();
            for (int j = 0; j < s.getColumn(); ++j) {
                dtype c = 1.0 / (n * sds_[i]);
                Tensor1D y2;
                y2.init(n);
                y2.vec() = Vec(s.getVal().v + j * n, n).square();
                Tensor1D m;
                m.init(n);
                m.vec() = Vec(s.getGrad().v + j * n, n) * Vec(s.getVal().v + j * n, n);
                Tensor1D x;
                x.init(n);
                x.vec() = c * ((-y2.vec() +
                            static_cast<dtype>(n -1)) * Vec(s.getGrad().v + j * n, n) -
                        ((m.mat().sum() - m.vec()) * Vec(s.getVal().v + j * n, n) +
                         Mat(s.getGrad().v + j * n, n, 1).sum() - Vec(s.getGrad().v + j * n, n)));
                Vec(s.inputGrad().v + j * n, n) += x.vec();
                ++i;
            }
        }
    }

    int calculateFLOPs() override {
        return 0; // TODO
    }

private:
    Tensor1D sds_;
    int col_sum_ = 0;
};
#endif

Executor *StandardLayerNormNode::generate() {
    return new LayerNormExecutor;
}

class PointwiseLinearNode : public UniInputNode, public Poolable<PointwiseLinearNode> {
public:
    PointwiseLinearNode() : UniInputNode("pointise-linear") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        Node::setDim(dim);
    }

    void compute() override {
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            Vec(val().v + i * row, row) = Vec(getInputVal().v + i * row, row) *
                params_->g().val().vec() + params_->b().val().vec();
        }
    }

    void backward() override {
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            Vec(inputGrad().v + i * row, row) += Vec(getGrad().v + i * row, row) *
                params_->g().val().vec();
            params_->g().grad().vec() += Vec(getGrad().v + i * row, row) *
                Vec(getInputVal().v + i * row, row);
            params_->b().grad().vec() += Vec(getGrad().v + i * row, row);
        }
    }

    Executor *generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + "-" + addressToString(params_);
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();;
    }

    bool isInputValForwardOnly() const override {
        return false;
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    LayerNormalizationParams *params_;

    friend class BatchedPointwiseLinearNode;
    friend class PointwiseLinearExecutor;
    friend Node *layerNormalization(LayerNormalizationParams &params,
            Node &input_layer, int col);
};

class BatchedPointwiseLinearNode : public BatchedNodeImpl<PointwiseLinearNode> {
public:
    void init(BatchedNode &input, LayerNormalizationParams &params) {
        allocateBatch(input.getDim(), input.batch().size());
        setInputsPerNode({&input});
        for (Node *node : batch()) {
            PointwiseLinearNode &p = dynamic_cast<PointwiseLinearNode &>(*node);
            p.params_ = &params;
        }
        afterInit({&input});
    }
};

#if USE_GPU
class PointwiseLinearExecutor : public Executor {
public:
    void forward() override {
#if TEST_CUDA
        testForwardInpputs();
#endif
        int count = batch.size();
        in_vals_.reserve(count);
        vector<dtype *> vals(count);
        vector<int> cols;
        cols.reserve(count);
        int i = 0;
        for (Node *node : batch)  {
            PointwiseLinearNode &p = dynamic_cast<PointwiseLinearNode &>(*node);
            in_vals_.push_back(p.getInputVal().value);
            vals.at(i++) = p.getVal().value;
            cols.push_back(p.getColumn());
        }
        in_val_arr_.init(in_vals_.data(), count);
        col_arr_.init(cols.data(), count);
        int row = getRow();
        max_col_ = *max_element(cols.begin(), cols.end());
        cuda::NumberPointerArray val_arr;
        val_arr.init(vals.data(), count);

        cuda::PointwiseLinearForward(in_val_arr_.value, count, row, col_arr_.value, max_col_,
                params().g().val().value, params().b().val().value, val_arr.value);

#if TEST_CUDA
        testForward();
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype *> grads(count), in_grads(count);
        int i = 0;
        int col_sum = 0;
        vector<int> dims(count), dim_offsets(count);
        int row = getRow();
        for (Node *node : batch)  {
            PointwiseLinearNode &p = dynamic_cast<PointwiseLinearNode &>(*node);
            grads.at(i) = p.getGrad().value;
            dims.at(i) = p.getDim();
            dim_offsets.at(i) = col_sum * row;
            in_grads.at(i++) = p.inputGrad().value;
            col_sum += p.getColumn();
        }
        cuda::NumberPointerArray grad_arr, in_grad_arr;
        grad_arr.init(grads.data(), count);
        in_grad_arr.init(in_grads.data(), count);
        cuda::IntArray dim_arr, dim_offset_arr;
        dim_arr.init(dims.data(), count);
        dim_offset_arr.init(dim_offsets.data(), count);

        cuda::PointwiseLinearBackward(grad_arr.value, in_val_arr_.value,
                params().g().val().value, count, row, col_arr_.value, max_col_, col_sum,
                dim_arr.value, dim_offset_arr.value, in_grad_arr.value, params().g().grad().value,
                params().b().grad().value);

#if TEST_CUDA
        testBackward();
        params().g().grad().verify("PointwiseLinearExecutor backward g");
        params().b().grad().verify("PointwiseLinearExecutor backward bias");
#endif
    }

private:
    vector<dtype *> in_vals_;
    cuda::NumberPointerArray in_val_arr_;
    cuda::IntArray col_arr_;
    int max_col_;

    LayerNormalizationParams &params() {
        return *dynamic_cast<PointwiseLinearNode *>(batch.front())->params_;
    }
};
#else
class PointwiseLinearExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0; // TODO
    }
};
#endif

Executor *PointwiseLinearNode::generate() {
    return new PointwiseLinearExecutor;
}

Node *layerNormalization(LayerNormalizationParams &params, Node &input_layer,
        int col) {
    using namespace n3ldg_plus;
    bool pool = col == 1;
    StandardLayerNormNode *a = StandardLayerNormNode::newNode(input_layer.getDim(), pool);
    a->setColumn(col);
    a->connect(input_layer);
    PointwiseLinearNode *b = PointwiseLinearNode::newNode(input_layer.getDim());
    b->setColumn(col);
    b->params_ = &params;
    b->connect(*a);
    return b;
}

BatchedNode *layerNormalization(LayerNormalizationParams &params,
        BatchedNode &input_layer) {
    using namespace n3ldg_plus;
    BatchedStandardLayerNormNode *a = new BatchedStandardLayerNormNode;
    a->init(input_layer);
    BatchedPointwiseLinearNode *b = new BatchedPointwiseLinearNode;
    b->init(*a, params);
    return b;
}

vector<Node *> layerNormalization(LayerNormalizationParams &params,
        const vector<Node *> &input_layer) {
    vector<Node *> results;
    results.reserve(input_layer.size());
    for (Node *x : input_layer) {
        results.push_back(layerNormalization(params, *x));
    }
    return results;
}

}
