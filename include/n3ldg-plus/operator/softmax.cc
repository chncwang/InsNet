#include "n3ldg-plus/operator/softmax.h"

using std::string;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;

namespace n3ldg_plus {

class SoftmaxNode : public UniInputNode, public Poolable<SoftmaxNode> {
public:
    SoftmaxNode() : UniInputNode("softmax") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    Executor *generate() override;

    void compute() override {
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            dtype max = Mat(inputVal().v + row * i, row, 1).maxCoeff();
            Tensor1D x_exp, x;
            x.init(row);
            x_exp.init(row);
            x.vec() = Vec(inputVal().v + row * i, row) - max;
            x_exp.vec() = x.vec().exp();
            dtype sum = x_exp.mat().sum();
            Vec(val().v + row * i, row) = x_exp.vec() / sum;
        }
    }

    void backward() override {
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            Tensor1D a;
            a.init(row);
            a.vec() = Vec(getGrad().v + i * row, row) * Vec(getVal().v + i * row, row);
            dtype z = a.mat().sum();
            Tensor1D b;
            b.init(row);
            b.vec() = z - a.vec();
            Vec(inputGrad().v + i * row, row) += Vec(getVal().v + i * row, row) *
                ((1 - Vec(getVal().v + i * row, row)) * Vec(getGrad().v + i * row, row) - b.vec());
        }
    }

    string typeSignature() const override {
        return Node::getNodeType();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return true;
    }

    bool isInputValForwardOnly() const override {
        return true;
    }

    bool isValForwardOnly() const override {
        return false;
    }
};

class BatchedSoftmaxNode : public BatchedNodeImpl<SoftmaxNode> {
public:
    void init(BatchedNode &input, int col) {
        allocateBatch(input.getDims());
        setInputsPerNode({&input});
        for (Node *node : batch()) {
            node->setColumn(col);
        }
        afterInit({&input});
    }
};

#if USE_GPU
class SoftmaxExecutor : public Executor {
public:
    void forward() override {
#if TEST_CUDA
        Executor::testForwardInpputs();
#endif
        int count = batch.size();
        vector<dtype *> in_vals(count);
        vals_.reserve(count);
        rows_.reserve(count);
        cols_.reserve(count);
        int i = 0;
        for (Node *node : batch) {
            SoftmaxNode &s = dynamic_cast<SoftmaxNode &>(*node);
            vals_.push_back(s.getVal().value);
            rows_.push_back(s.getDim() / s.getColumn());
            cols_.push_back(s.getColumn());
            in_vals.at(i++) = s.inputVal().value;
        }
        row_arr_.init(rows_.data(), count);
        col_arr_.init(cols_.data(), count);
        val_arr_.init(vals_.data(), count);
        max_col_ = *max_element(cols_.begin(), cols_.end());
        max_row_ = *max_element(rows_.begin(), rows_.end());
        cuda::SoftmaxForward(in_vals, count, row_arr_.value, max_row_, col_arr_.value,
                max_col_, val_arr_.value);
#if TEST_CUDA
        try {
            Executor::testForward();
        } catch (cuda::CudaVerificationException &e) {
            cerr << "softmax forward verification failed" << endl;
            SoftmaxNode &s = dynamic_cast<SoftmaxNode &>(*batch.at(e.getIndex()));
            cerr << "input val:" << s.inputVal().toString() << endl;
            cerr << "gpu:" << endl;
            s.inputVal().print();
            cout << fmt::format("count:{} dim:{} col:{}\n", count, s.getDim(), s.getColumn());
            abort();
        }
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype *> grads(count), in_grads(count);
        int i = 0;
        vector<int> offsets(count);
        int dim_sum = 0;
        for (Node *node : batch) {
            SoftmaxNode &s = dynamic_cast<SoftmaxNode &>(*node);
            offsets.at(i) = dim_sum;
            dim_sum += s.getDim();
            grads.at(i) = s.getGrad().value;
            in_grads.at(i++) = s.inputGrad().value;
        }
        cuda::IntArray offset_arr;
        offset_arr.init(offsets.data(), count);
        cuda::SoftmaxBackward(grads, val_arr_.value, count, row_arr_.value, max_row_,
                col_arr_.value, max_col_, offset_arr.value, dim_sum, in_grads);
#if TEST_CUDA
        Executor::testBackward();
#endif
    }

private:
    vector<dtype *> vals_;
    vector<int> rows_, cols_;
    cuda::IntArray row_arr_, col_arr_;
    cuda::NumberPointerArray val_arr_;
    int max_row_, max_col_;
};
#else
class SoftmaxExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0; // TODO
    }
};
#endif

Executor *SoftmaxNode::generate() {
    return new SoftmaxExecutor;
}

Node* softmax(Node &input, int row) {
    using namespace n3ldg_plus;
    SoftmaxNode *node = SoftmaxNode::newNode(input.getDim());
    int col = input.getDim() / row;
    if (col * row != input.getDim()) {
        cerr << fmt::format("softmax - col:{} row:{} input dim:{}", col, row, input.getDim()) <<
            endl;
        abort();
    }
    node->setColumn(col);
    node->connect(input);
    return node;
}

BatchedNode* softmax(BatchedNode &input, int row) {
    BatchedSoftmaxNode *ret = new BatchedSoftmaxNode;
    int col = input.getDim() / row;
    if (col * row != input.getDim()) {
        cerr << fmt::format("softmax - col:{} row:{} input dim:{}", col, row, input.getDim()) <<
            endl;
        abort();
    }
    ret->init(input, col);
    return ret;
}

}
