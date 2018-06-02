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

    inline void clearValue() {
        Node::clearValue();
        ins.clear();
        for(int idx = 0; idx < dim; idx++) {
            masks[idx] = -1;
        }
    }


    inline void init(int ndim, dtype dropout) {
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
                clearValue();
                return;
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
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

  public:
    virtual inline void setMask() = 0;

    inline void compute() {
        int nSize = ins.size();
        setMask();
        for(int i = 0; i < dim; i++) {
            val[i] = ins[masks[i]]->val[i];
        }
    }

    void backward() {
        for(int i = 0; i < dim; i++) {
            ins[masks[i]]->loss[i] += loss[i];
        }
    }

};

class MaxPoolNode : public PoolNode {
  public:
    MaxPoolNode() : PoolNode() {
        node_type = "max-pooling";
    }

  public:
    //Be careful that the row is the dim of input vector, and the col is the number of input vectors
    //Another point is that we change the input vectors directly.
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



class PoolExecute : public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};

inline PExecute PoolNode::generate(bool bTrain, dtype cur_drop_factor) {
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
  public:
    SumPoolNode() : Node() {
        ins.clear();
        node_type = "sum-pool";
    }

    inline void clearValue() {
        ins.clear();
        Node::clearValue();
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
    inline void compute() {
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
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

};


class SumPoolExecute : public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};


inline PExecute SumPoolNode::generate(bool bTrain, dtype cur_drop_factor) {
    SumPoolExecute* exec = new SumPoolExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
}



class AvgPoolNode : public Node {
  public:
    vector<PNode> ins;

    ~AvgPoolNode() {
        ins.clear();
    }
  public:
    AvgPoolNode() : Node() {
        ins.clear();
        node_type = "avg-pool";
    }

    inline void clearValue() {
        ins.clear();
        Node::clearValue();
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
    inline void compute() {
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
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

};


class AvgPoolExecute : public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};


inline PExecute AvgPoolNode::generate(bool bTrain, dtype cur_drop_factor) {
    AvgPoolExecute* exec = new AvgPoolExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
}

#endif
