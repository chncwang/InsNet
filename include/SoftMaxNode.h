#ifndef N3LDG_PLUS_SOFTMAX_NODE
#define N3LDG_PLUS_SOFTMAX_NODE

#include "AtomicOP.h"
#include "Sub.h"
#include "AtomicOP.h"
#include "Div.h"
#include "Split.h"

#include <boost/format.hpp>

class SoftmaxNode : public UniInputNode, public Poolable<SoftmaxNode> {
public:
    SoftmaxNode() : UniInputNode("softmax") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    Executor *generate() override;

    void compute() override {
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            dtype max = Mat(getInput().getVal().v + row * i, row, 1).maxCoeff();
            Tensor1D x_exp, x;
            x.init(row);
            x_exp.init(row);
            x.vec() = Vec(getInput().getVal().v + row * i, row) - max;
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
            a.vec() = Vec(getLoss().v + i * row, row) * Vec(getVal().v + i * row, row);
            dtype z = a.mat().sum();
            Tensor1D b;
            b.init(row);
            b.vec() = z - a.vec();
            Vec(getInput().loss().v + i * row, row) += Vec(getVal().v + i * row, row) *
                ((1 - Vec(getVal().v + i * row, row)) * Vec(getLoss().v + i * row, row) - b.vec());
        }
    }

    string typeSignature() const override {
        return Node::getNodeType();
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return true;
    }
};

class BatchedSoftmaxNode : public BatchedNodeImpl<SoftmaxNode> {
public:
    void init(Graph &graph, BatchedNode &input, int col) {
        allocateBatch(input.getDims());
        setInputsPerNode({&input});
        for (Node *node : batch()) {
            node->setColumn(col);
        }
        afterInit(graph, {&input});
    }
};

#if USE_GPU
class SoftmaxExecutor : public UniInputExecutor {
public:
    void forward() override {
#if TEST_CUDA
        UniInputExecutor::testForwardInpputs();
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
            in_vals.at(i++) = s.getInput().getVal().value;
        }
        row_arr_.init(rows_.data(), count);
        col_arr_.init(cols_.data(), count);
        val_arr_.init(vals_.data(), count);
        max_col_ = *max_element(cols_.begin(), cols_.end());
        max_row_ = *max_element(rows_.begin(), rows_.end());
        n3ldg_cuda::SoftmaxForward(in_vals, count, row_arr_.value, max_row_, col_arr_.value,
                max_col_, val_arr_.value);
#if TEST_CUDA
        try {
            UniInputExecutor::testForward();
        } catch (n3ldg_cuda::CudaVerificationException &e) {
            cerr << "softmax forward verification failed" << endl;
            SoftmaxNode &s = dynamic_cast<SoftmaxNode &>(*batch.at(e.getIndex()));
            cerr << "input val:" << s.getInput().getVal().toString() << endl;
            cerr << "gpu:" << endl;
            s.getInput().getVal().print();
            cout << boost::format("count:%1% dim:%2% col:%3%") % count % s.getDim() %
                s.getColumn() << endl;
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
            grads.at(i) = s.getLoss().value;
            in_grads.at(i++) = s.getInput().getLoss().value;
        }
        n3ldg_cuda::IntArray offset_arr;
        offset_arr.init(offsets.data(), count);
        n3ldg_cuda::SoftmaxBackward(grads, val_arr_.value, count, row_arr_.value, max_row_,
                col_arr_.value, max_col_, offset_arr.value, in_grads);
#if TEST_CUDA
        UniInputExecutor::testBackward();
#endif
    }

private:
    vector<dtype *> vals_;
    vector<int> rows_, cols_;
    n3ldg_cuda::IntArray row_arr_, col_arr_;
    n3ldg_cuda::NumberPointerArray val_arr_;
    int max_row_, max_col_;
};
#else
class SoftmaxExecutor : public UniInputExecutor {
public:
    int calculateFLOPs() override {
        return 0; // TODO
    }
};
#endif

Executor *SoftmaxNode::generate() {
    return new SoftmaxExecutor;
}

namespace n3ldg_plus {

Node *minusMaxScalar(Graph &graph, Node &input, int input_col) {
    using namespace n3ldg_plus;

    Node *max_scalar = maxScalar(graph, input, input_col);
    int input_row = input.getDim() / input_col;
    Node *scalar_to_vector = scalarToVector(graph, *max_scalar, input_row);
    Node *subtracted = sub(graph, input, *scalar_to_vector);
    return subtracted;
}

BatchedNode *minusMaxScalar(Graph &graph, BatchedNode &input, int input_col) {
    using namespace n3ldg_plus;

    BatchedNode *max_scalar = maxScalar(graph, input, input_col);
    vector<int> input_rows;
    for (int dim : input.getDims()) {
        input_rows.push_back(dim / input_col);
    }
    BatchedNode *scalar_to_vector = scalarToVector(graph, *max_scalar, input_rows);
    BatchedNode *subtracted = sub(graph, input, *scalar_to_vector);
    return subtracted;
}

Node* softmax(Graph &graph, Node &input, int input_col) {
    using namespace n3ldg_plus;
    Node *subtracted = minusMaxScalar(graph, input, input_col);
    Node *exp = n3ldg_plus::exp(graph, *subtracted);
    Node *sum = vectorSum(graph, *exp, input_col);
    int input_row = input.getDim() / input_col;
    sum = scalarToVector(graph, *sum, input_row);
    Node *div = n3ldg_plus::fullDiv(graph, *exp, *sum);
    return div;
}

BatchedNode* softmax(Graph &graph, BatchedNode &input, int col = 1) {
    using namespace n3ldg_plus;
    BatchedSoftmaxNode *node = new BatchedSoftmaxNode;
    node->init(graph, input, col);
    return node;
}

};

#endif
