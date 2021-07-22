#include "insnet/operator/log_softmax.h"

using std::string;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;

namespace insnet {

class LogSoftmaxNode : public UniInputNode, public Poolable<LogSoftmaxNode> {
public:
    LogSoftmaxNode() : UniInputNode("logsoftmax") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    Executor *generate() override;

    void compute() override {
        int row = size() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            dtype max = Mat(inputVal().v + row * i, row, 1).maxCoeff();
            Tensor1D x_exp, x;
            x.init(row);
            x_exp.init(row);
            x.vec() = Vec(inputVal().v + row * i, row) - max;
            x_exp.vec() = x.vec().exp();
            dtype log_sum = std::log(x_exp.mat().sum());
            Vec(val().v + row * i, row) = x.vec() - log_sum;
        }
    }

    void backward() override {
        int row = size() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            dtype z = Mat(getGrad().v + row * i, row, 1).sum();
            Vec(inputGrad().v + i * row, row) += Vec(getGrad().v + i * row, row) -
                z * Vec(getVal().v + row * i, row).exp();
        }
    }

    string typeSignature() const override {
        return Node::getNodeType() + "-" + std::to_string(size() / getColumn());
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

#if USE_GPU
class LogSoftmaxExecutor : public Executor {
public:
    void forward() override {
#if TEST_CUDA
        Executor::testForwardInpputs();
#endif
        int count = batch.size();
        vector<dtype *> in_vals(count);
        vector<dtype *> vals;
        vector<int> cols;
        vals.reserve(count);
        cols.reserve(count);
        int i = 0;
        int row;
        for (Node *node : batch) {
            LogSoftmaxNode &s = dynamic_cast<LogSoftmaxNode &>(*node);
            vals.push_back(s.getVal().value);
            cols.push_back(s.getColumn());
            row = s.size() / s.getColumn();
            in_vals.at(i++) = s.inputVal().value;
        }
        col_arr_.init(cols.data(), count);
        val_arr_.init(vals.data(), count);
        cuda::NumberPointerArray in_val_arr;
        in_val_arr.init(in_vals.data(), count);
        max_col_ = *max_element(cols.begin(), cols.end());
        cuda::LogSoftmaxForward(in_val_arr.value, count, row, col_arr_.value, max_col_,
                val_arr_.value);
#if TEST_CUDA
        try {
            Executor::testForward();
        } catch (cuda::CudaVerificationException &e) {
            cerr << "logsoftmax forward verification failed" << endl;
            LogSoftmaxNode &s = dynamic_cast<LogSoftmaxNode &>(*batch.at(e.getIndex()));
            cerr << "input val:" << s.inputVal().toString() << endl;
            cerr << "gpu:" << endl;
            s.inputVal().print();
            cout << fmt::format("count:{} dim:{} col:{}\n", count, s.size(), s.getColumn());
            abort();
        }
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype *> grads(count), in_grads(count);
        int i = 0;
        int dim_sum = 0;
        int row;
        for (Node *node : batch) {
            LogSoftmaxNode &s = dynamic_cast<LogSoftmaxNode &>(*node);
            row = s.size() / s.getColumn();
            dim_sum += s.size();
            grads.at(i) = s.getGrad().value;
            in_grads.at(i++) = s.inputGrad().value;
        }
        cuda::NumberPointerArray grad_arr, in_grad_arr;
        grad_arr.init(grads.data(), count);
        in_grad_arr.init(in_grads.data(), count);
        cuda::LogSoftmaxBackward(grad_arr.value, val_arr_.value, count, row, col_arr_.value,
                max_col_, in_grad_arr.value);
#if TEST_CUDA
        Executor::testBackward();
#endif
    }

private:
    cuda::IntArray col_arr_;
    cuda::NumberPointerArray val_arr_;
    int max_col_;
};
#else
class LogSoftmaxExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0; // TODO
    }
};
#endif

Executor *LogSoftmaxNode::generate() {
    return new LogSoftmaxExecutor;
}

Node* logSoftmax(Node &input, int row) {
    LogSoftmaxNode *node = LogSoftmaxNode::newNode(input.size());
    int col = input.size() / row;
    if (col * row != input.size()) {
        cerr << fmt::format("logSoftmax - col:{} row:{} input dim:{}", col, row, input.size()) <<
            endl;
        abort();
    }
    node->setColumn(col);
    node->connect(input);
    return node;
}

}
