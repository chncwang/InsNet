#ifndef POOLING
#define POOLING

/*
*  Pooling.h:
*  pool operation, max, min, average and sum pooling
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#if USE_GPU
#include "N3LDG_cuda.h"
#endif
#include "profiler.h"

class PoolNode : public Node {
  public:
    vector<int> masks;
    vector<PNode> ins;

  public:
    PoolNode() : Node() {
        ins.clear();
        masks.clear();
    }

    ~PoolNode() {
        masks.clear();
        ins.clear();
    }

    void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
        masks.resize(ndim);
        for(int idx = 0; idx < ndim; idx++) {
            masks[idx] = -1;
        }
    }

  public:
    void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for max|min|sum|avg pooling" << std::endl;
            return;
        }
        int nSize = x.size();
        ins.clear();
        for (int i = 0; i < nSize; i++) {
            if (x[i]->val.dim != dim) {
                std::cout << "input matrixes are not matched" << std::endl;
                abort();
            }
            ins.push_back(x[i]);
        }

        degree = 0;
        for (int i = 0; i < nSize; i++) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }


  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

  public:
    virtual void setMask() = 0;

    void compute() {
        int nSize = ins.size();
        setMask();
        for(int i = 0; i < dim; i++) {
            int mask_i = masks.at(i);
            val[i] = ins.at(mask_i)->val[i];
        }
    }

    void backward() {
        for(int i = 0; i < dim; i++) {
            ins[masks[i]]->loss[i] += loss[i];
        }
    }

};

#if USE_GPU
class MaxPoolNode : public
#if TEST_CUDA
                    PoolNode
#else
                    Node
#endif
{
public:
#if !TEST_CUDA
    vector<PNode> ins;
#endif
    MaxPoolNode() {
        node_type = "max-pooling";
    }

#if TEST_CUDA
    void setMask() {
        int nSize = ins.size();
        int thread_count = 8;
        while (thread_count < nSize) {
            thread_count <<= 1;
        }

        for (int dim_i = 0; dim_i < dim; ++dim_i) {
            dtype shared_arr[1024];
            dtype shared_indexers[1024];
            for (int i = 0; i < 1024; ++i) {
                shared_arr[i] = i < nSize ? ins[i]->val[dim_i] : -INFINITY;
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
    void compute() {
        abort();
    }

    void backward() {
        abort();
    }
#endif

    void forward(Graph *cg, const vector<PNode>& x) {
        assert(!x.empty());
        int nSize = x.size();
        ins.clear();
        for (int i = 0; i < nSize; i++) {
            assert(x[i]->val.dim == dim);
            ins.push_back(x[i]);
        }

        degree = 0;
        for (int i = 0; i < nSize; i++) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    PExecute generate(bool bTrain, dtype cur_drop_factor) override;
};
#else
class MaxPoolNode : public PoolNode {
  public:
    MaxPoolNode() : PoolNode() {
        node_type = "max-pooling";
    }

    void setMask() {
        int nSize = ins.size();

        for (int idx = 0; idx < dim; idx++) {
            int maxIndex = -1;
            for (int i = 0; i < nSize; ++i) {
                if (maxIndex == -1 || ins[i]->val[idx] > ins[maxIndex]->val[idx]) {
                    maxIndex = i;
                }
            }
            masks[idx] = maxIndex;
        }
    }

};
#endif

#if USE_GPU
class MinPoolNode : public
#if TEST_CUDA
                    PoolNode
#else
                    Node
#endif
{
public:
#if !TEST_CUDA
    vector<PNode> ins;
#endif
    MinPoolNode() {
        node_type = "max-pooling";
    }

#if TEST_CUDA
    void setMask() {
        int nSize = ins.size();
        int thread_count = 8;
        while (thread_count < nSize) {
            thread_count <<= 1;
        }

        for (int dim_i = 0; dim_i < dim; ++dim_i) {
            dtype shared_arr[1024];
            dtype shared_indexers[1024];
            for (int i = 0; i < 1024; ++i) {
                shared_arr[i] = i < nSize ? ins[i]->val[dim_i] : INFINITY;
                shared_indexers[i] = i;
            }
            for (int i = (thread_count >> 1); i > 0; i >>= 1) {
                for (int j = 0; j < i; ++j) {
                    int plus_i = j + i;
                    if (shared_arr[j] > shared_arr[plus_i]) {
                        shared_arr[j] = shared_arr[plus_i];
                        shared_indexers[j] = shared_indexers[plus_i];
                    }
                }

                masks[dim_i] = shared_indexers[0];
            }
        }
    }
#else
    void compute() {
        abort();
    }

    void backward() {
        abort();
    }
#endif

    void forward(Graph *cg, const vector<PNode>& x) {
        assert(!x.empty());
        int nSize = x.size();
        ins.clear();
        for (int i = 0; i < nSize; i++) {
            assert(x[i]->val.dim == dim);
            ins.push_back(x[i]);
        }

        degree = 0;
        for (int i = 0; i < nSize; i++) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    PExecute generate(bool bTrain, dtype cur_drop_factor) override;
};
#else
class MinPoolNode : public PoolNode {
  public:
    MinPoolNode() : PoolNode() {
        node_type = "min-pooling";
    }

  public:
    //Be careful that the row is the dim of input vector, and the col is the number of input vectors
    //Another point is that we change the input vectors directly.
    void setMask() {
        int nSize = ins.size();
        for (int idx = 0; idx < dim; idx++) {
            int minIndex = -1;
            for (int i = 0; i < nSize; ++i) {
                if (minIndex == -1 || ins[i]->val[idx] < ins[minIndex]->val[idx]) {
                    minIndex = i;
                }
            }
            masks[idx] = minIndex;
        }
    }
};
#endif

#if USE_GPU
class MaxPoolExecute : public Execute
{
public:
    int dim;
    n3ldg_cuda::IntArray hit_inputs;
    std::vector<int> in_counts;
    int max_in_count;

    void forward() override {
        int count = batch.size();
        hit_inputs.init(count * dim);
        in_counts.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = static_cast<MaxPoolNode*>(n);
            in_counts.push_back(m->ins.size());
        }
        max_in_count = *std::max_element(in_counts.begin(), in_counts.end());
        std::vector<dtype*> in_vals;
        in_vals.reserve(count * max_in_count);
        std::vector<dtype*> vals;
        vals.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = static_cast<MaxPoolNode*>(n);
            vals.push_back(m->val.value);
            for (Node *in : m->ins) {
                in_vals.push_back(in->val.value);
            }
            for (int i = 0; i < max_in_count - m->ins.size(); ++i) {
                in_vals.push_back(NULL);
            }
        }
        n3ldg_cuda::PoolForward(n3ldg_cuda::PoolingEnum::MAX, in_vals, vals,
                count, in_counts, dim, hit_inputs.value);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch[idx]->val.verify("max pooling forward"));
            MaxPoolNode *n = static_cast<MaxPoolNode*>(batch[idx]);
            if (!n3ldg_cuda::Verify(n->masks.data(),
                        hit_inputs.value + idx * dim, dim,
                        "max pooling forward mask")) {
                abort();
            }
        }
#endif
    }

    void backward() override {
        int count = batch.size();
        std::vector<dtype*> in_losses;
        in_losses.reserve(count * max_in_count);
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = static_cast<MaxPoolNode*>(n);
            losses.push_back(m->loss.value);
            for (Node *in : m->ins) {
                in_losses.push_back(in->loss.value);
            }
            for (int i = 0; i < max_in_count - m->ins.size(); ++i) {
                in_losses.push_back(NULL);
            }
        }

        n3ldg_cuda::PoolBackward(losses, in_losses, in_counts,
                hit_inputs.value, count, dim);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        for (int idx = 0; idx < count; idx++) {
            int in_i = 0;
            for (Node *n : static_cast<MaxPoolNode*>(batch[idx])->ins) {
                n3ldg_cuda::Assert(n->loss.verify("max pooling backward"));
            }
        }
#endif
    }
};

PExecute MaxPoolNode::generate(bool bTrain, dtype cur_drop_factor) {
    MaxPoolExecute *exec = new MaxPoolExecute;
    exec->batch.push_back(this);
    exec->dim = dim;
    return exec;
}
#endif

#if USE_GPU
class MinPoolExecute : public Execute
{
public:
    int dim;
    n3ldg_cuda::IntArray hit_inputs;
    std::vector<int> in_counts;
    int max_in_count;

    void forward() override {
        int count = batch.size();
        hit_inputs.init(count * dim);
        in_counts.reserve(count);
        for (Node *n : batch) {
            MinPoolNode *m = static_cast<MinPoolNode*>(n);
            in_counts.push_back(m->ins.size());
        }
        max_in_count = *std::max_element(in_counts.begin(), in_counts.end());
        std::vector<dtype*> in_vals;
        in_vals.reserve(count * max_in_count);
        std::vector<dtype*> vals;
        vals.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = static_cast<MaxPoolNode*>(n);
            vals.push_back(m->val.value);
            for (Node *in : m->ins) {
                in_vals.push_back(in->val.value);
            }
            for (int i = 0; i < max_in_count - m->ins.size(); ++i) {
                in_vals.push_back(NULL);
            }
        }
        n3ldg_cuda::PoolForward(n3ldg_cuda::PoolingEnum::MIN, in_vals, vals,
                count, in_counts, dim, hit_inputs.value);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch[idx]->val.verify("min pooling forward"));
            MinPoolNode *n = static_cast<MinPoolNode*>(batch[idx]);
            if (!n3ldg_cuda::Verify(n->masks.data(),
                        hit_inputs.value + idx * dim, dim,
                        "min pooling forward mask")) {
            }
        }
#endif
    }

    void backward() override {
        int count = batch.size();
        std::vector<dtype*> in_losses;
        in_losses.reserve(count * max_in_count);
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (Node *n : batch) {
            MaxPoolNode *m = static_cast<MaxPoolNode*>(n);
            losses.push_back(m->loss.value);
            for (Node *in : m->ins) {
                in_losses.push_back(in->loss.value);
            }
            for (int i = 0; i < max_in_count - m->ins.size(); ++i) {
                in_losses.push_back(NULL);
            }
        }

        n3ldg_cuda::PoolBackward(losses, in_losses, in_counts,
                hit_inputs.value, count, dim);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        for (int idx = 0; idx < count; idx++) {
            int in_i = 0;
            for (Node *n : static_cast<MaxPoolNode*>(batch[idx])->ins) {
                n3ldg_cuda::Assert(n->loss.verify("max pooling backward"));
            }
        }
#endif
    }
};

PExecute MinPoolNode::generate(bool bTrain, dtype cur_drop_factor) {
    MinPoolExecute *exec = new MinPoolExecute;
    exec->batch.push_back(this);
    exec->dim = dim;
    return exec;
}
#endif

class PoolExecute : public Execute {
  public:
    virtual void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    virtual void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};

PExecute PoolNode::generate(bool bTrain, dtype cur_drop_factor) {
    PoolExecute* exec = new PoolExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
}



class SumPoolNode : public Node {
  public:
    vector<PNode> ins;

    ~SumPoolNode() {
        ins.clear();
    }

    SumPoolNode() : Node() {
        ins.clear();
        node_type = "sum-pool";
    }

    void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for add" << std::endl;
            return;
        }

        ins.clear();
        for (int i = 0; i < x.size(); i++) {
            if (x[i]->val.dim == dim) {
                ins.push_back(x[i]);
            } else {
                std::cout << "dim does not match" << std::endl;
            }
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x4->dim == dim) {
            ins.push_back(x4);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x4->dim == dim) {
            ins.push_back(x4);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x5->dim == dim) {
            ins.push_back(x5);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x4->dim == dim) {
            ins.push_back(x4);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x5->dim == dim) {
            ins.push_back(x5);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x6->dim == dim) {
            ins.push_back(x6);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

  public:
    void compute() {
        int nSize = ins.size();
        val.zero();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < dim; idx++) {
                val[idx] += ins[i]->val[idx];
            }
        }
    }


    void backward() {
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < dim; idx++) {
                ins[i]->loss[idx] += loss[idx];
            }
        }
    }


  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

};

#if USE_GPU
class SumPoolExecute : public Execute {
public:
    int dim;
    std::vector<int> in_counts;
    int max_in_count;
    std::vector<dtype*> in_vals;

    void  forward() {
        int count = batch.size();
        in_counts.reserve(count);
        for (Node *n : batch) {
            SumPoolNode *sum = static_cast<SumPoolNode*>(n);
            in_counts.push_back(sum->ins.size());
        }

        max_in_count = *std::max_element(in_counts.begin(), in_counts.end());

        for (Node *n : batch) {
            SumPoolNode *sum = static_cast<SumPoolNode*>(n);
            in_counts.push_back(sum->ins.size());
        }

        std::vector<dtype*> vals;
        in_vals.reserve(count * max_in_count);
        vals.reserve(count);

        for (Node *n : batch) {
            SumPoolNode *sum = static_cast<SumPoolNode*>(n);
            vals.push_back(sum->val.value);
            for (int i = 0; i < sum->ins.size(); ++i) {
                in_vals.push_back(sum->ins.at(i)->val.value);
            }
            for (int i = 0; i < max_in_count - sum->ins.size(); ++i) {
                in_vals.push_back(NULL);
            }
        }

        n3ldg_cuda::SumPoolForward(n3ldg_cuda::PoolingEnum::SUM, in_vals,
                count, dim, in_counts, vals);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
        }

        for (Node *n : batch) {
            n3ldg_cuda::Assert(n->val.verify("sum pool forward"));
        }
#endif
    }

    void backward() {
        int count = batch.size();
        std::vector<dtype*> losses;
        losses.reserve(count);
        std::vector<dtype*> in_losses;
        in_losses.reserve(max_in_count * count);
        for (Node *n : batch) {
            SumPoolNode *sum = static_cast<SumPoolNode*>(n);
            losses.push_back(n->loss.value);
            for (Node *in : sum->ins) {
                in_losses.push_back(in->loss.value);
            }
            for (int i = 0; i < max_in_count - sum->ins.size(); ++i) {
                in_losses.push_back(NULL);
            }
        }
        n3ldg_cuda::SumPoolBackward(n3ldg_cuda::PoolingEnum::SUM, losses,
                in_counts, count, dim, in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            SumPoolNode *sum = static_cast<SumPoolNode*>(n);
            for (Node *in : sum->ins) {
                n3ldg_cuda::Assert(in->loss.verify("SumPoolExecute backward"));
            }
        }
#endif
    }
};
#else
class SumPoolExecute : public Execute {
  public:
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};
#endif

PExecute SumPoolNode::generate(bool bTrain, dtype cur_drop_factor) {
    SumPoolExecute* exec = new SumPoolExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
#if USE_GPU
    exec->dim = dim;
#endif
    return exec;
}



class AvgPoolNode : public Node {
  public:
    vector<PNode> ins;

    ~AvgPoolNode() {
        ins.clear();
    }

    AvgPoolNode() : Node() {
        ins.clear();
        node_type = "avg-pool";
    }

  public:
    void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for add" << std::endl;
            return;
        }

        ins.clear();
        for (int i = 0; i < x.size(); i++) {
            if (x[i]->val.dim == dim) {
                ins.push_back(x[i]);
            } else {
                std::cout << "dim does not match" << std::endl;
            }
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x4->dim == dim) {
            ins.push_back(x4);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x4->dim == dim) {
            ins.push_back(x4);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x5->dim == dim) {
            ins.push_back(x5);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6) {
        ins.clear();
        if (x1->dim == dim) {
            ins.push_back(x1);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x2->dim == dim) {
            ins.push_back(x2);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x3->dim == dim) {
            ins.push_back(x3);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x4->dim == dim) {
            ins.push_back(x4);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x5->dim == dim) {
            ins.push_back(x5);
        } else {
            std::cout << "dim does not match" << std::endl;
        }
        if (x6->dim == dim) {
            ins.push_back(x6);
        } else {
            std::cout << "dim does not match" << std::endl;
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }

  public:
    void compute() {
        int nSize = ins.size();
        val.zero();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < dim; idx++) {
                val[idx] += ins[i]->val[idx] * 1.0 / nSize;
            }
        }

    }


    void backward() {
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < dim; idx++) {
                ins[i]->loss[idx] += loss[idx] * 1.0 / nSize;
            }
        }
    }


  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

};

#if USE_GPU
class AvgPoolExecute : public Execute {
public:
    int dim;
    std::vector<int> in_counts;
    int max_in_count;
    std::vector<dtype*> in_vals;

    void  forward() {
        int count = batch.size();
        in_counts.reserve(count);
        for (Node *n : batch) {
            AvgPoolNode *sum = static_cast<AvgPoolNode*>(n);
            in_counts.push_back(sum->ins.size());
        }

        max_in_count = *std::max_element(in_counts.begin(), in_counts.end());

        for (Node *n : batch) {
            AvgPoolNode *sum = static_cast<AvgPoolNode*>(n);
            in_counts.push_back(sum->ins.size());
        }

        std::vector<dtype*> vals;
        in_vals.reserve(count * max_in_count);
        vals.reserve(count);

        for (Node *n : batch) {
            AvgPoolNode *sum = static_cast<AvgPoolNode*>(n);
            vals.push_back(sum->val.value);
            for (int i = 0; i < sum->ins.size(); ++i) {
                in_vals.push_back(sum->ins.at(i)->val.value);
            }
            for (int i = 0; i < max_in_count - sum->ins.size(); ++i) {
                in_vals.push_back(NULL);
            }
        }

        n3ldg_cuda::SumPoolForward(n3ldg_cuda::PoolingEnum::AVG, in_vals,
                count, dim, in_counts, vals);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
        }

        for (Node *n : batch) {
            n3ldg_cuda::Assert(n->val.verify("avg pool forward"));
        }
#endif
    }

    void backward() {
        int count = batch.size();
        std::vector<dtype*> losses;
        losses.reserve(count);
        std::vector<dtype*> in_losses;
        in_losses.reserve(max_in_count * count);
        for (Node *n : batch) {
            AvgPoolNode *sum = static_cast<AvgPoolNode*>(n);
            losses.push_back(n->loss.value);
            for (Node *in : sum->ins) {
                in_losses.push_back(in->loss.value);
            }
            for (int i = 0; i < max_in_count - sum->ins.size(); ++i) {
                in_losses.push_back(NULL);
            }
        }
        n3ldg_cuda::SumPoolBackward(n3ldg_cuda::PoolingEnum::AVG, losses,
                in_counts, count, dim, in_losses);
#if TEST_CUDA
        for (Node *n : batch) {
            n->backward();
        }
        for (Node *n : batch) {
            AvgPoolNode *sum = static_cast<AvgPoolNode*>(n);
            for (Node *in : sum->ins) {
                n3ldg_cuda::Assert(in->loss.verify("AvgPoolExecute backward"));
            }
        }
#endif
    }
};
#else
class AvgPoolExecute : public Execute {
  public:
    void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};
#endif

PExecute AvgPoolNode::generate(bool bTrain, dtype cur_drop_factor) {
    AvgPoolExecute* exec = new AvgPoolExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
#if USE_GPU
    exec->dim = dim;
#endif
    return exec;
}

#endif
