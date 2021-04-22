#include "n3ldg-plus/operator/pooling.h"
#include "n3ldg-plus/operator/atomic.h"

using std::string;
using std::vector;
using std::cerr;
using std::endl;

namespace n3ldg_plus {

class PoolNode : public Node {
public:
    vector<int> masks;
    vector<Node *> ins;

    PoolNode(const string &node_type) : Node(node_type) {}

    void clear() override {
        ins.clear();
        Node::clear();
    }

    void init(int ndim) override {
        Node::init(ndim);
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
            if (x[i]->val().dim != getDim()) {
                cerr << "input matrixes are not matched" << endl;
                abort();
            }
            ins.push_back(x[i]);
        }

        afterConnect(x);
    }

    Executor *generate() override;

    virtual void setMask() = 0;

    void compute() override {
        setMask();
        for(int i = 0; i < getDim(); i++) {
            int mask_i = masks.at(i);
            val()[i] = ins.at(mask_i)->val()[i];
        }
    }

    void backward() override {
        for(int i = 0; i < getDim(); i++) {
            ins[masks[i]]->loss()[i] += loss()[i];
        }
    }
};

#if USE_GPU
class MaxPoolNode : public PoolNode, public Poolable<MaxPoolNode>
{
public:
    MaxPoolNode() : PoolNode("max_pool") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

#if TEST_CUDA
    void setMask() override {
        int nSize = ins.size();
        int thread_count = 8;
        while (thread_count < nSize) {
            thread_count <<= 1;
        }

        for (int dim_i = 0; dim_i < getDim(); ++dim_i) {
            dtype shared_arr[1024];
            dtype shared_indexers[1024];
            for (int i = 0; i < 1024; ++i) {
                shared_arr[i] = i < nSize ? ins[i]->val()[dim_i] : -INFINITY;
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
};
#else
class MaxPoolNode : public PoolNode, public Poolable<MaxPoolNode> {
public:
    MaxPoolNode() : PoolNode("max-pooling") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void setMask() override {
        int nSize = ins.size();

        for (int idx = 0; idx < getDim(); idx++) {
            int maxIndex = -1;
            for (int i = 0; i < nSize; ++i) {
                if (maxIndex == -1 || ins[i]->val()[idx] > ins[maxIndex]->val()[idx]) {
                    maxIndex = i;
                }
            }
            masks[idx] = maxIndex;
        }
    }

};
#endif

#if USE_GPU
class MaxPoolExecutor : public Executor
{
public:
    int dim;
    cuda::IntArray hit_inputs;
    vector<int> in_counts;
    int max_in_count;

    void forward() override {
        int count = batch.size();
        hit_inputs.init(count * dim);
        in_counts.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = dynamic_cast<MaxPoolNode*>(n);
            in_counts.push_back(m->ins.size());
        }
        max_in_count = *max_element(in_counts.begin(), in_counts.end());
        vector<dtype*> in_vals;
        in_vals.reserve(count * max_in_count);
        vector<dtype*> vals;
        vals.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = dynamic_cast<MaxPoolNode*>(n);
            vals.push_back(m->val().value);
            for (Node *in : m->ins) {
                in_vals.push_back(in->val().value);
            }
            for (int i = 0; i < max_in_count - m->ins.size(); ++i) {
                in_vals.push_back(NULL);
            }
        }
        cuda::PoolForward(cuda::PoolingEnum::MAX, in_vals, vals,
                count, in_counts, dim, hit_inputs.value);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            cuda::Assert(batch[idx]->val().verify("max pooling forward"));
            MaxPoolNode *n = dynamic_cast<MaxPoolNode*>(batch[idx]);
            if (!cuda::Verify(n->masks.data(),
                        hit_inputs.value + idx * dim, dim,
                        "max pooling forward mask")) {
                abort();
            }
        }
#endif
    }

    void backward() override {
        int count = batch.size();
        vector<dtype*> in_losses;
        in_losses.reserve(count * max_in_count);
        vector<dtype*> losses;
        losses.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = dynamic_cast<MaxPoolNode*>(n);
            losses.push_back(m->loss().value);
            for (Node *in : m->ins) {
                in_losses.push_back(in->loss().value);
            }
            for (int i = 0; i < max_in_count - m->ins.size(); ++i) {
                in_losses.push_back(NULL);
            }
        }

        cuda::PoolBackward(losses, in_losses, in_counts,
                hit_inputs.value, count, dim);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        for (int idx = 0; idx < count; idx++) {
            for (Node *n : dynamic_cast<MaxPoolNode*>(batch[idx])->ins) {
                cuda::Assert(n->loss().verify("max pooling backward"));
            }
        }
#endif
    }
};

Executor * MaxPoolNode::generate() {
    MaxPoolExecutor *exec = new MaxPoolExecutor;
    exec->batch.push_back(this);
    exec->dim = getDim();
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
    vector<Node *> ins;

    void clear() override {
        ins.clear();
        Node::clear();
    }

    SumPoolNode() : Node("sum-pool") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void connect(const vector<Node *>& x) {
        if (x.size() == 0) {
            cerr << "empty inputs for add" << endl;
            abort();
        }

        for (int i = 0; i < x.size(); i++) {
            if (x[i]->getDim() == getDim()) {
                ins.push_back(x[i]);
            } else {
                cerr << "dim does not match" << endl;
                abort();
            }
        }

        afterConnect(x);
    }

    void compute() override {
        int nSize = ins.size();
        val().zero();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < getDim(); idx++) {
                val()[idx] += ins[i]->val()[idx];
            }
        }
    }

    void backward() override {
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < getDim(); idx++) {
                ins[i]->loss()[idx] += loss()[idx];
            }
        }
    }

    Executor * generate() override;
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
            in_counts.push_back(sum->ins.size());
        }

        max_in_count = *max_element(in_counts.begin(), in_counts.end());

        for (Node *n : batch) {
            SumPoolNode *sum = dynamic_cast<SumPoolNode*>(n);
            in_counts.push_back(sum->ins.size());
        }

        vector<dtype*> vals;
        in_vals.reserve(count * max_in_count);
        vals.reserve(count);

        for (Node *n : batch) {
            SumPoolNode *sum = dynamic_cast<SumPoolNode*>(n);
            vals.push_back(sum->val().value);
            for (int i = 0; i < sum->ins.size(); ++i) {
                in_vals.push_back(sum->ins.at(i)->val().value);
            }
            for (int i = 0; i < max_in_count - sum->ins.size(); ++i) {
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
            losses.push_back(n->loss().value);
            for (Node *in : sum->ins) {
                in_losses.push_back(in->loss().value);
            }
            for (int i = 0; i < max_in_count - sum->ins.size(); ++i) {
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
            for (Node *in : sum->ins) {
                cuda::Assert(in->loss().verify("SumPoolExecutor backward"));
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
            sum += s->getDim() * s->ins.size();
        }
        return sum;
    }
};
#endif

Executor * SumPoolNode::generate() {
    SumPoolExecutor* exec = new SumPoolExecutor();
    exec->batch.push_back(this);
#if USE_GPU
    exec->dim = getDim();
#endif
    return exec;
}

Node *maxPool(vector<Node *> &inputs) {
    int dim = inputs.front()->getDim();
    for (int i = 1; i < inputs.size(); ++i) {
        if (dim != inputs.at(i)->getDim()) {
            cerr << "dim not equal" << endl;
            abort();
        }
    }

    MaxPoolNode *pool = MaxPoolNode::newNode(dim);
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
    int dim = inputs.front()->getDim();
    SumPoolNode *pool = SumPoolNode::newNode(dim);
    pool->connect(inputs);
    return pool;
}

Node *averagePool(vector<Node *> &inputs) {
    Node *sum = sumPool(inputs);
    return scaled(*sum, 1.0 / inputs.size());
}

}
