#include "n3ldg-plus/operator/pooling.h"
#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/split.h"
#include "n3ldg-plus/util/util.h"

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::function;

namespace n3ldg_plus {

class PoolNode : public Node {
public:
    vector<int> masks;

    PoolNode(const string &node_type) : Node(node_type) {}

    void clear() override {
        Node::clear();
    }

    void init(int ndim) {
        masks.resize(ndim);
        for(int idx = 0; idx < ndim; idx++) {
            masks[idx] = -1;
        }
    }

    void connect(const vector<Node *> &x) {
        if (x.size() == 0) {
            cerr << "empty inputs for max|min|sum|avg pooling" << endl;
            abort();
        }
        int nSize = x.size();
        for (int i = 0; i < nSize; i++) {
            if (x[i]->val().dim != size()) {
                cerr << "input matrixes are not matched" << endl;
                abort();
            }
        }

        setInputs(x);
        afterConnect(x);
    }

    Executor *generate() override;

    virtual void setMask() = 0;

    void compute() override {
        setMask();
        for(int i = 0; i < size(); i++) {
            int mask_i = masks.at(i);
            val()[i] = (*input_vals_.at(mask_i))[i];
        }
    }

    void backward() override {
        for(int i = 0; i < size(); i++) {
            int mask_i = masks.at(i);
            (*input_grads_.at(mask_i))[i] += grad()[i];
        }
    }
};

#if USE_GPU
class MaxPoolNode : public PoolNode, public Poolable<MaxPoolNode> {
public:
    MaxPoolNode() : PoolNode("max_pool") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

#if TEST_CUDA
    void setMask() override {
        int nSize = inputSize();
        int thread_count = 8;
        while (thread_count < nSize) {
            thread_count <<= 1;
        }

        for (int dim_i = 0; dim_i < size(); ++dim_i) {
            dtype shared_arr[1024];
            dtype shared_indexers[1024];
            for (int i = 0; i < 1024; ++i) {
                shared_arr[i] = i < nSize ? (*input_vals_.at(i))[dim_i] : -INFINITY;
                shared_indexers[i] = i;
            }
            for (int i = (thread_count >> 1); i > 0; i >>= 1) {
                for (int j = 0; j < i; ++j) {
                    int plus_i = j + i;
                    if (shared_arr[j] < shared_arr[plus_i]) {
                        shared_arr[j] = shared_arr[plus_i];
                        shared_indexers[j] = shared_indexers[plus_i];
                    }
                }

                masks[dim_i] = shared_indexers[0];
            }
        }
    }
#else
    void setMask() override {
        cerr << "unsupported op" << endl;
        abort();
    }

    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }
#endif

    Executor * generate() override;

protected:
    int forwardOnlyInputValSize() override {
        return inputSize();
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    friend class MaxPoolExecutor;
};
#else
class MaxPoolNode : public PoolNode, public Poolable<MaxPoolNode> {
public:
    MaxPoolNode() : PoolNode("max-pooling") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void setMask() override {
        int size = inputSize();

        for (int idx = 0; idx < size(); idx++) {
            int max_i = -1;
            for (int i = 0; i < size; ++i) {
                if (max_i == -1 || (*input_vals_.at(i))[idx] > (*input_vals_.at(i))[idx]) {
                    max_i = i;
                }
            }
            masks[idx] = max_i;
        }
    }

protected:
    int forwardOnlyInputValSize() override {
        return inputSize();
    }

    bool isValForwardOnly() const override {
        return true;
    }
};
#endif

#if USE_GPU
class MaxPoolExecutor : public Executor
{
public:
    void forward() override {
        int count = batch.size();
        hit_inputs.init(count * size());
        in_counts.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = dynamic_cast<MaxPoolNode*>(n);
            in_counts.push_back(m->inputSize());
        }
        max_in_count = *max_element(in_counts.begin(), in_counts.end());
        vector<dtype*> in_vals;
        in_vals.reserve(count * max_in_count);
        vector<dtype*> vals;
        vals.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = dynamic_cast<MaxPoolNode*>(n);
            vals.push_back(m->val().value);
            for (auto &in : m->input_vals_) {
                in_vals.push_back(in->value);
            }
            for (int i = 0; i < max_in_count - m->inputSize(); ++i) {
                in_vals.push_back(NULL);
            }
        }
        cuda::PoolForward(cuda::PoolingEnum::MAX, in_vals, vals, count, in_counts, size(),
                hit_inputs.value);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            cuda::Assert(batch[idx]->val().verify("max pooling forward"));
            MaxPoolNode *n = dynamic_cast<MaxPoolNode*>(batch[idx]);
            int dim = n->input_dims_.front();
            if (!cuda::Verify(n->masks.data(), hit_inputs.value + idx * dim, dim,
                        "max pooling forward mask")) {
                abort();
            }
        }
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype*> in_grades;
        in_grades.reserve(count * max_in_count);
        vector<dtype*> grades;
        grades.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = dynamic_cast<MaxPoolNode*>(n);
            grades.push_back(m->grad().value);
            for (auto &in : m->input_grads_) {
                in_grades.push_back(in->value);
            }
            for (int i = 0; i < max_in_count - m->inputSize(); ++i) {
                in_grades.push_back(NULL);
            }
        }

        cuda::PoolBackward(grades, in_grades, in_counts, hit_inputs.value, count, size());

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        for (int idx = 0; idx < count; idx++) {
            for (Tensor1D *t : dynamic_cast<MaxPoolNode*>(batch[idx])->input_grads_) {
                cuda::Assert(t->verify("max pooling backward"));
            }
        }
#endif
    }

private:
    cuda::IntArray hit_inputs;
    vector<int> in_counts;
    int max_in_count;
};

Executor * MaxPoolNode::generate() {
    MaxPoolExecutor *exec = new MaxPoolExecutor;
    return exec;
}
#endif

class PoolExecutor : public Executor {
public:
#if !USE_GPU
    int calculateFLOPs() override {
        cerr << "unsupported op" << endl;
        abort();
    }
#endif
};

Executor * PoolNode::generate() {
    PoolExecutor* exec = new PoolExecutor();
    exec->batch.push_back(this);
    return exec;
}



class SumPoolNode : public Node, public Poolable<SumPoolNode> {
public:
    void clear() override {
        Node::clear();
    }

    SumPoolNode() : Node("sum-pool") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void connect(const vector<Node *> &x) {
        if (x.size() == 0) {
            cerr << "empty inputs for add" << endl;
            abort();
        }

        for (int i = 0; i < x.size(); i++) {
            if (x[i]->size() != size()) {
                cerr << "dim does not match" << endl;
                abort();
            }
        }

        setInputs(x);
        afterConnect(x);
    }

    void compute() override {
        val().zero();
        for (int i = 0; i < inputSize(); ++i) {
            val().vec() += input_vals_.at(i)->vec();
        }
    }

    void backward() override {
        for (int i = 0; i < inputSize(); ++i) {
            input_grads_.at(i)->vec() += grad().vec();
        }
    }

    Executor * generate() override;

protected:
    int forwardOnlyInputValSize() override {
        return inputSize();
    }

    virtual bool isValForwardOnly() const override {
        return true;
    }

private:
    friend class SumPoolExecutor;
};

#if USE_GPU
class SumPoolExecutor : public Executor {
public:
    int dim;
    vector<int> in_counts;
    int max_in_count;
    vector<dtype*> in_vals;

    void  forward() {
        int count = batch.size();
        in_counts.reserve(count);
        for (Node *n : batch) {
            SumPoolNode *sum = dynamic_cast<SumPoolNode*>(n);
            in_counts.push_back(sum->inputSize());
        }

        max_in_count = *max_element(in_counts.begin(), in_counts.end());

        for (Node *n : batch) {
            SumPoolNode *sum = dynamic_cast<SumPoolNode*>(n);
            in_counts.push_back(sum->inputSize());
        }

        vector<dtype*> vals;
        in_vals.reserve(count * max_in_count);
        vals.reserve(count);

        for (Node *n : batch) {
            SumPoolNode *sum = dynamic_cast<SumPoolNode*>(n);
            vals.push_back(sum->val().value);
            for (int i = 0; i < sum->inputSize(); ++i) {
                in_vals.push_back(sum->input_vals_.at(i)->value);
            }
            for (int i = 0; i < max_in_count - sum->inputSize(); ++i) {
                in_vals.push_back(NULL);
            }
        }

        cuda::SumPoolForward(cuda::PoolingEnum::SUM, in_vals,
                count, dim, in_counts, vals);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
        }

        for (Node *n : batch) {
            cuda::Assert(n->val().verify("sum pool forward"));
        }
#endif
    }

    void backward() {
        int count = batch.size();
        vector<dtype*> losses;
        losses.reserve(count);
        vector<dtype*> in_losses;
        in_losses.reserve(max_in_count * count);
        for (Node *n : batch) {
            SumPoolNode *sum = dynamic_cast<SumPoolNode*>(n);
            losses.push_back(n->grad().value);
            for (auto &in : sum->input_grads_) {
                in_losses.push_back(in->value);
            }
            for (int i = 0; i < max_in_count - sum->inputSize(); ++i) {
                in_losses.push_back(NULL);
            }
        }
        cuda::SumPoolBackward(cuda::PoolingEnum::SUM, losses,
                in_counts, count, dim, in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            SumPoolNode *sum = dynamic_cast<SumPoolNode*>(n);
            for (Tensor1D *t : sum->input_grads_) {
                cuda::Assert(t->verify("SumPoolExecutor backward"));
            }
        }
#endif
    }
};
#else
class SumPoolExecutor : public Executor {
public:
    int calculateFLOPs() override {
        int sum = 0;
        for (Node *node : batch) {
            SumPoolNode *s = dynamic_cast<SumPoolNode*>(node);
            sum += s->size() * s->inputSize();
        }
        return sum;
    }
};
#endif

Executor * SumPoolNode::generate() {
    SumPoolExecutor* exec = new SumPoolExecutor();
    exec->batch.push_back(this);
#if USE_GPU
    exec->dim = size();
#endif
    return exec;
}

Node *maxPool(vector<Node *> &inputs) {
    int dim = inputs.front()->size();
    for (int i = 1; i < inputs.size(); ++i) {
        if (dim != inputs.at(i)->size()) {
            cerr << "dim not equal" << endl;
            abort();
        }
    }

    MaxPoolNode *pool = MaxPoolNode::newNode(dim);
#if !USE_GPU
    pool->init(dim);
#endif
    pool->connect(inputs);
    return pool;
}

Node *minPool(vector<Node *> &inputs) {
    vector<Node *> negative;
    negative.reserve(inputs.size());
    for (Node *input : inputs) {
        negative.push_back(scaled(*input, -1));
    }
    return scaled(*maxPool(negative), -1);
}

Node *sumPool(vector<Node *> &inputs) {
    int dim = inputs.front()->size();
    SumPoolNode *pool = SumPoolNode::newNode(dim);
    pool->connect(inputs);
    return pool;
}

Node *avgPool(vector<Node *> &inputs) {
    Node *sum = sumPool(inputs);
    return scaled(*sum, 1.0 / inputs.size());
}

Node *avgPool(Node &input, int row) {
    int col = input.size() / row;
    if (col * row != input.size()) {
        cerr << fmt::format("avgPool col:{} row:{} input dim:{}", col, row, input.size()) <<
            endl;
        abort();
    }
    int offset = 0;
    vector<Node *> inputs;
    inputs.reserve(col);
    for (int i = 0; i < col; ++i) {
        Node *in = n3ldg_plus::split(input, row, offset);
        inputs.push_back(in);
        offset += row;
    }
    return avgPool(inputs);
}

}
