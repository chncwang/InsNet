/*
* SparseCOP.h
*
*  Created on: Jul 20, 2016
*      Author: mszhang
*/

#ifndef SparseCOP_H_
#define SparseCOP_H_

#include "COPUtils.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "SparseParam.h"

class SparseC1Params {
  public:
    SparseParam W;
    int nDim;
    unordered_map<C1Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    SparseC1Params() {
        nDim = 0;
        nVSize = -1;
        bound = 0;
        hash2id.clear();
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        if (nVSize > 0)ada.addParam(&W);
    }

    inline void initialWeights() {
        if (nVSize <= 0) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        W.initial(nDim, nVSize);
    }

    //random initialization
    inline void initial(int outputDim = 1) {
        hash2id.clear();
        bound = 0;
        nVSize = -1;
        nDim = outputDim;
    }

    inline int getFeatureId(const int& id) {
        if (id < 0) {
            return -1;
        }
        C1Feat feat;
        feat.setId(id);
        if (hash2id.find(feat) != hash2id.end()) {
            return hash2id[feat];
        }
        return -1;
    }

    inline void collectFeature(const int& id) {
        if (id < 0) {
            return;
        }
        C1Feat feat;
        feat.setId(id);

        if (hash2id.find(feat) != hash2id.end()) {
            return;
        }

        if (nVSize < 0 && bound < maxCapacity) {
            hash2id[feat] = bound;
            bound++;
            return;
        }

        if (nVSize > 0 && bound < nVSize) {
            hash2id[feat] = bound;
            bound++;
            return;
        }
    }

    inline void setFixed(int base) {
        nVSize = (bound + 1) * base;
        if (nVSize > maxCapacity) nVSize = maxCapacity;
        initialWeights();
    }
};

// a single node;
// input variables are not allocated by current node(s)
class SparseC1Node : public Node {
  public:
    SparseC1Params* param;
    int tx;
    bool executed;

  public:
    SparseC1Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "sparsec1";
    }

    inline void setParam(SparseC1Params* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        tx = -1;
        executed = false;
    }

  public:
    inline void forward(Graph* cg, const int& x1) {
        //assert(param != NULL);
        int featId = param->getFeatureId(x1);
        if (featId < 0) {
            tx = -1;
            executed = false;
            return;
        }
        tx = featId;
        executed = true;
        degree = 0;
        cg->addNode(this);
    }

  public:
    inline void compute() {
        param->W.value(tx, val);
    }
    //no output losses
    void backward() {
        //assert(param != NULL);
        if (tx >= 0) {
            param->W.loss(tx, loss);
        }
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        SparseC1Node* conv_other = (SparseC1Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class SparseC1Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC1Node* ptr = (SparseC1Node*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC1Node* ptr = (SparseC1Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute SparseC1Node::generate(bool bTrain) {
    SparseC1Execute* exec = new SparseC1Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}



//a sparse feature has two actomic features
class SparseC2Params {
  public:
    SparseParam W;
    int nDim;
    unordered_map<C2Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    SparseC2Params() {
        nDim = 0;
        nVSize = -1;
        bound = 0;
        hash2id.clear();
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        if (nVSize > 0)ada.addParam(&W);
    }

    inline void initialWeights() {
        if (nVSize <= 0) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        W.initial(nDim, nVSize);
    }

    //random initialization
    inline void initial(int outputDim = 1) {
        hash2id.clear();
        bound = 0;
        nVSize = -1;
        nDim = outputDim;
    }

    //important!!! if > nHVSize, using LW, otherwise, using HW
    inline int getFeatureId(const int& id1, const int& id2) {
        if (id1 < 0 || id2 < 0) {
            return -1;
        }
        C2Feat feat;
        feat.setId(id1, id2);
        if (hash2id.find(feat) != hash2id.end()) {
            return hash2id[feat];
        }
        return -1;
    }

    inline void collectFeature(const int& id1, const int& id2) {
        if (id1 < 0 || id2 < 0) {
            return;
        }
        C2Feat feat;
        feat.setId(id1, id2);

        if (hash2id.find(feat) != hash2id.end()) {
            return;
        }

        if (nVSize < 0 && bound < maxCapacity) {
            hash2id[feat] = bound;
            bound++;
            return;
        }

        if (nVSize > 0 && bound < nVSize) {
            hash2id[feat] = bound;
            bound++;
            return;
        }
    }

    inline void setFixed(int base) {
        nVSize = (bound + 1) * base;
        if (nVSize > maxCapacity) nVSize = maxCapacity;
        initialWeights();
    }
};

// a single node;
// input variables are not allocated by current node(s)
class SparseC2Node : public Node {
  public:
    SparseC2Params* param;
    int tx;
    bool executed;

  public:
    SparseC2Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "sparsec2";
    }

    inline void setParam(SparseC2Params* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        tx = -1;
        executed = false;
    }

  public:
    inline void forward(Graph* cg, const int& x1, const int& x2) {
        //assert(param != NULL);
        int featId = param->getFeatureId(x1, x2);
        if (featId < 0) {
            tx = -1;
            executed = false;
            return;
        }
        tx = featId;
        executed = true;
        degree = 0;
        cg->addNode(this);
    }

  public:
    inline void compute() {
        param->W.value(tx, val);
    }
    //no output losses
    void backward() {
        //assert(param != NULL);
        if (tx >= 0) {
            param->W.loss(tx, loss);
        }
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        SparseC2Node* conv_other = (SparseC2Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class SparseC2Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC2Node* ptr = (SparseC2Node*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC2Node* ptr = (SparseC2Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute SparseC2Node::generate(bool bTrain) {
    SparseC2Execute* exec = new SparseC2Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}


//a sparse feature has three actomic features
class SparseC3Params {
  public:
    SparseParam W;
    int nDim;
    unordered_map<C3Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    SparseC3Params() {
        nDim = 0;
        nVSize = -1;
        bound = 0;
        hash2id.clear();
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        if (nVSize > 0)ada.addParam(&W);
    }

    inline void initialWeights() {
        if (nVSize <= 0) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        W.initial(nDim, nVSize);
    }

    //random initialization
    inline void initial(int outputDim = 1) {
        hash2id.clear();
        bound = 0;
        nVSize = -1;
        nDim = outputDim;
    }

    inline int getFeatureId(const int& id1, const int& id2, const int& id3) {
        if (id1 < 0 || id2 < 0 || id3 < 0) {
            return -1;
        }
        C3Feat feat;
        feat.setId(id1, id2, id3);
        if (hash2id.find(feat) != hash2id.end()) {
            return hash2id[feat];
        }

        return -1;
    }

    inline void collectFeature(const int& id1, const int& id2, const int& id3) {
        if (id1 < 0 || id2 < 0 || id3 < 0) {
            return;
        }
        C3Feat feat;
        feat.setId(id1, id2, id3);

        if (hash2id.find(feat) != hash2id.end()) {
            return;
        }

        if (nVSize < 0 && bound < maxCapacity) {
            hash2id[feat] = bound;
            bound++;
            return;
        }

        if (nVSize > 0 && bound < nVSize) {
            hash2id[feat] = bound;
            bound++;
            return;
        }
    }

    inline void setFixed(int base) {
        nVSize = (bound + 1) * base;
        if (nVSize > maxCapacity) nVSize = maxCapacity;
        initialWeights();
    }
};

// a single node;
// input variables are not allocated by current node(s)
class SparseC3Node : public Node {
  public:
    SparseC3Params* param;
    int tx;
    bool executed;

  public:
    SparseC3Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "sparsec3";
    }

    inline void setParam(SparseC3Params* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        tx = -1;
        executed = false;
    }

  public:
    inline void forward(Graph* cg, const int& x1, const int& x2, const int& x3) {
        //assert(param != NULL);
        int featId = param->getFeatureId(x1, x2, x3);
        if (featId < 0) {
            tx = -1;
            executed = false;
            return;
        }
        tx = featId;
        executed = true;
        degree = 0;
        cg->addNode(this);
    }

  public:
    inline void compute() {
        param->W.value(tx, val);
    }
    //no output losses
    void backward() {
        //assert(param != NULL);
        if (tx >= 0) {
            param->W.loss(tx, loss);
        }
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        SparseC3Node* conv_other = (SparseC3Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class SparseC3Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC3Node* ptr = (SparseC3Node*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC3Node* ptr = (SparseC3Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute SparseC3Node::generate(bool bTrain) {
    SparseC3Execute* exec = new SparseC3Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}


//a sparse feature has four actomic features
class SparseC4Params {
  public:
    SparseParam W;
    int nDim;
    unordered_map<C4Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    SparseC4Params() {
        nDim = 0;
        nVSize = -1;
        bound = 0;
        hash2id.clear();
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        if (nVSize > 0)ada.addParam(&W);
    }

    inline void initialWeights() {
        if (nVSize <= 0) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        W.initial(nDim, nVSize);
    }

    //random initialization
    inline void initial(int outputDim = 1) {
        hash2id.clear();
        bound = 0;
        nVSize = -1;
        nDim = outputDim;
    }

    inline int getFeatureId(const int& id1, const int& id2, const int& id3, const int& id4) {
        if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0) {
            return -1;
        }
        C4Feat feat;
        feat.setId(id1, id2, id3, id4);
        if (hash2id.find(feat) != hash2id.end()) {
            return hash2id[feat];
        }

        return -1;
    }

    inline void collectFeature(const int& id1, const int& id2, const int& id3, const int& id4) {
        if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0) {
            return;
        }
        C4Feat feat;
        feat.setId(id1, id2, id3, id4);

        if (hash2id.find(feat) != hash2id.end()) {
            return;
        }

        if (nVSize < 0 && bound < maxCapacity) {
            hash2id[feat] = bound;
            bound++;
            return;
        }

        if (nVSize > 0 && bound < nVSize) {
            hash2id[feat] = bound;
            bound++;
            return;
        }
    }

    inline void setFixed(int base) {
        nVSize = (bound + 1) * base;
        if (nVSize > maxCapacity) nVSize = maxCapacity;
        initialWeights();
    }
};

// a single node;
// input variables are not allocated by current node(s)
class SparseC4Node : public Node {
  public:
    SparseC4Params* param;
    int tx;
    bool executed;

  public:
    SparseC4Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "sparsec4";
    }

    inline void setParam(SparseC4Params* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        tx = -1;
        executed = false;
    }

  public:
    inline void forward(Graph* cg, const int& x1, const int& x2, const int& x3, const int& x4) {
        //assert(param != NULL);
        int featId = param->getFeatureId(x1, x2, x3, x4);
        if (featId < 0) {
            tx = -1;
            executed = false;
            return;
        }
        tx = featId;
        executed = true;
        degree = 0;
        cg->addNode(this);
    }

  public:
    inline void compute() {
        param->W.value(tx, val);
    }
    //no output losses
    void backward() {
        //assert(param != NULL);
        if (tx >= 0) {
            param->W.loss(tx, loss);
        }
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        SparseC4Node* conv_other = (SparseC4Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class SparseC4Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC4Node* ptr = (SparseC4Node*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC4Node* ptr = (SparseC4Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute SparseC4Node::generate(bool bTrain) {
    SparseC4Execute* exec = new SparseC4Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}


//a sparse feature has five actomic features
class SparseC5Params {
  public:
    SparseParam W;
    int nDim;
    unordered_map<C5Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    SparseC5Params() {
        nDim = 0;
        nVSize = -1;
        bound = 0;
        hash2id.clear();
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        if (nVSize > 0)ada.addParam(&W);
    }

    inline void initialWeights() {
        if (nVSize <= 0) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        W.initial(nDim, nVSize);
    }

    //random initialization
    inline void initial(int outputDim = 1) {
        hash2id.clear();
        bound = 0;
        nVSize = -1;
        nDim = outputDim;
    }

    inline int getFeatureId(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5) {
        if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0) {
            return -1;
        }
        C5Feat feat;
        feat.setId(id1, id2, id3, id4, id5);
        if (hash2id.find(feat) != hash2id.end()) {
            return hash2id[feat];
        }

        return -1;
    }

    inline void collectFeature(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5) {
        if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0) {
            return;
        }
        C5Feat feat;
        feat.setId(id1, id2, id3, id4, id5);

        if (hash2id.find(feat) != hash2id.end()) {
            return;
        }

        if (nVSize < 0 && bound < maxCapacity) {
            hash2id[feat] = bound;
            bound++;
            return;
        }

        if (nVSize > 0 && bound < nVSize) {
            hash2id[feat] = bound;
            bound++;
            return;
        }
    }

    inline void setFixed(int base) {
        nVSize = (bound + 1) * base;
        if (nVSize > maxCapacity) nVSize = maxCapacity;
        initialWeights();
    }
};

// a single node;
// input variables are not allocated by current node(s)
class SparseC5Node : public Node {
  public:
    SparseC5Params* param;
    int tx;
    bool executed;

  public:
    SparseC5Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "sparsec5";
    }

    inline void setParam(SparseC5Params* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        tx = -1;
        executed = false;
    }

  public:
    inline void forward(Graph* cg, const int& x1, const int& x2, const int& x3, const int& x4, const int& x5) {
        //assert(param != NULL);
        int featId = param->getFeatureId(x1, x2, x3, x4, x5);
        if (featId < 0) {
            tx = -1;
            executed = false;
            return;
        }
        tx = featId;
        executed = true;
        degree = 0;
        cg->addNode(this);
    }

  public:
    inline void compute() {
        param->W.value(tx, val);
    }
    //no output losses
    void backward() {
        //assert(param != NULL);
        if (tx >= 0) {
            param->W.loss(tx, loss);
        }
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        SparseC5Node* conv_other = (SparseC5Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class SparseC5Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC5Node* ptr = (SparseC5Node*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC5Node* ptr = (SparseC5Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute SparseC5Node::generate(bool bTrain) {
    SparseC5Execute* exec = new SparseC5Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}


//a sparse feature has six actomic features
class SparseC6Params {
  public:
    SparseParam W;
    int nDim;
    unordered_map<C6Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    SparseC6Params() {
        nDim = 0;
        nVSize = -1;
        bound = 0;
        hash2id.clear();
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        if (nVSize > 0)ada.addParam(&W);
    }

    inline void initialWeights() {
        if (nVSize <= 0) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        W.initial(nDim, nVSize);
    }

    //random initialization
    inline void initial(int outputDim = 1) {
        hash2id.clear();
        bound = 0;
        nVSize = -1;
        nDim = outputDim;
    }

    inline int getFeatureId(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5, const int& id6) {
        if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0 || id6 < 0) {
            return -1;
        }
        C6Feat feat;
        feat.setId(id1, id2, id3, id4, id5, id6);
        if (hash2id.find(feat) != hash2id.end()) {
            return hash2id[feat];
        }

        return -1;
    }

    inline void collectFeature(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5, const int& id6) {
        if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0 || id6 < 0) {
            return;
        }
        C6Feat feat;
        feat.setId(id1, id2, id3, id4, id5, id6);

        if (hash2id.find(feat) != hash2id.end()) {
            return;
        }

        if (nVSize < 0 && bound < maxCapacity) {
            hash2id[feat] = bound;
            bound++;
            return;
        }

        if (nVSize > 0 && bound < nVSize) {
            hash2id[feat] = bound;
            bound++;
            return;
        }
    }

    inline void setFixed(int base) {
        nVSize = (bound + 1) * base;
        if (nVSize > maxCapacity) nVSize = maxCapacity;
        initialWeights();
    }
};

// a single node;
// input variables are not allocated by current node(s)
class SparseC6Node : public Node {
  public:
    SparseC6Params* param;
    int tx;
    bool executed;

  public:
    SparseC6Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "sparsec6";
    }

    inline void setParam(SparseC6Params* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        tx = -1;
        executed = false;
    }

  public:
    inline void forward(Graph* cg, const int& x1, const int& x2, const int& x3, const int& x4, const int& x5, const int& x6) {
        //assert(param != NULL);
        int featId = param->getFeatureId(x1, x2, x3, x4, x5, x6);
        if (featId < 0) {
            tx = -1;
            executed = false;
            return;
        }
        tx = featId;
        executed = true;
        degree = 0;
        cg->addNode(this);
    }

  public:
    inline void compute() {
        param->W.value(tx, val);
    }
    //no output losses
    void backward() {
        //assert(param != NULL);
        if (tx >= 0) {
            param->W.loss(tx, loss);
        }
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        SparseC6Node* conv_other = (SparseC6Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class SparseC6Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC6Node* ptr = (SparseC6Node*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC6Node* ptr = (SparseC6Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute SparseC6Node::generate(bool bTrain) {
    SparseC6Execute* exec = new SparseC6Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}

//a sparse feature has seven actomic features
class SparseC7Params {
  public:
    SparseParam W;
    int nDim;
    unordered_map<C7Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    SparseC7Params() {
        nDim = 0;
        nVSize = -1;
        bound = 0;
        hash2id.clear();
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        if (nVSize > 0)ada.addParam(&W);
    }

    inline void initialWeights() {
        if (nVSize <= 0) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        W.initial(nDim, nVSize);
    }

    //random initialization
    inline void initial(int outputDim = 1) {
        hash2id.clear();
        bound = 0;
        nVSize = -1;
        nDim = outputDim;
    }

    inline int getFeatureId(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5, const int& id6, const int& id7) {
        if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0 || id6 < 0 || id7 < 0) {
            return -1;
        }
        C7Feat feat;
        feat.setId(id1, id2, id3, id4, id5, id6, id7);
        if (hash2id.find(feat) != hash2id.end()) {
            return hash2id[feat];
        }

        return -1;
    }

    inline void collectFeature(const int& id1, const int& id2, const int& id3, const int& id4, const int& id5, const int& id6, const int& id7) {
        if (id1 < 0 || id2 < 0 || id3 < 0 || id4 < 0 || id5 < 0 || id6 < 0 || id7 < 0) {
            return;
        }
        C7Feat feat;
        feat.setId(id1, id2, id3, id4, id5, id6, id7);

        if (hash2id.find(feat) != hash2id.end()) {
            return;
        }

        if (nVSize < 0 && bound < maxCapacity) {
            hash2id[feat] = bound;
            bound++;
            return;
        }

        if (nVSize > 0 && bound < nVSize) {
            hash2id[feat] = bound;
            bound++;
            return;
        }
    }

    inline void setFixed(int base) {
        nVSize = (bound + 1) * base;
        if (nVSize > maxCapacity) nVSize = maxCapacity;
        initialWeights();
    }
};

// a single node;
// input variables are not allocated by current node(s)
class SparseC7Node : public Node {
  public:
    SparseC7Params* param;
    int tx;
    bool executed;

  public:
    SparseC7Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "sparsec7";
    }

    inline void setParam(SparseC7Params* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        tx = -1;
        executed = false;
    }

  public:
    inline void forward(Graph* cg, const int& x1, const int& x2, const int& x3, const int& x4, const int& x5, const int& x6, const int& x7) {
        //assert(param != NULL);
        int featId = param->getFeatureId(x1, x2, x3, x4, x5, x6, x7);
        if (featId < 0) {
            tx = -1;
            executed = false;
            return;
        }
        tx = featId;
        executed = true;
        degree = 0;
        cg->addNode(this);
    }

  public:
    inline void compute() {
        param->W.value(tx, val);
    }
    //no output losses
    void backward() {
        //assert(param != NULL);
        if (tx >= 0) {
            param->W.loss(tx, loss);
        }
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        SparseC7Node* conv_other = (SparseC7Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class SparseC7Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC7Node* ptr = (SparseC7Node*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseC7Node* ptr = (SparseC7Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute SparseC7Node::generate(bool bTrain) {
    SparseC7Execute* exec = new SparseC7Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}

class SparseCParams {
  public:
    SparseParam W;
    int nDim;
    unordered_map<CFeat, int> hash2id;
    int nVSize;
    int bound;
  public:
    SparseCParams() {
        nDim = 0;
        nVSize = -1;
        bound = 0;
        hash2id.clear();
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        if (nVSize > 0)ada.addParam(&W);
    }

    inline void initialWeights() {
        if (nVSize <= 0) {
            std::cout << "please check the alphabet" << std::endl;
            return;
        }
        W.initial(nDim, nVSize);
    }

    //random initialization
    inline void initial(int outputDim = 1) {
        hash2id.clear();
        bound = 0;
        nVSize = -1;
        nDim = outputDim;
    }

    inline int getFeatureId(const CFeat& f) {
        if (!f.valid) {
            return -1;
        }
        if (hash2id.find(f) != hash2id.end()) {
            return hash2id[f];
        }

        return -1;
    }

    inline void collectFeature(const CFeat& f) {
        if (!f.valid) {
            return;
        }

        if (hash2id.find(f) != hash2id.end()) {
            return;
        }

        if (nVSize < 0 && bound < maxCapacity) {
            hash2id[f] = bound;
            bound++;
            return;
        }

        if (nVSize > 0 && bound < nVSize) {
            hash2id[f] = bound;
            bound++;
            return;
        }
    }

    inline void setFixed(int base) {
        nVSize = (bound <= 0) ? 1 : bound * base;
        if (nVSize > maxCapacity) {
            nVSize = maxCapacity;
            std::cout << "reach max size" << std::endl;
        }
        initialWeights();
    }
};

// a single node;
// input variables are not allocated by current node(s)
class SparseCNode : public Node {
  public:
    SparseCParams* param;
    vector<int> ins;
    int nSize;

  public:
    SparseCNode() : Node() {
        param = NULL;
        ins.clear();
        nSize = 0;
        node_type = "sparseclist";
    }

    inline void setParam(SparseCParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        ins.clear();
        nSize = 0;
    }

  public:
    inline void forward(Graph* cg, const vector<CFeat*>& xs) {
        //assert(param != NULL);
        int featId, count;
        count = xs.size();
        for (int idx = 0; idx < count; idx++) {
            featId = param->getFeatureId(*(xs[idx]));
            if (featId >= 0) {
                ins.push_back(featId);
                nSize++;
            }
        }
        degree = 0;
        cg->addNode(this);
    }

  public:
    inline void compute() {
        param->W.value(ins, val);
    }

    //no output losses
    void backward() {
        //assert(param != NULL);
        param->W.loss(ins, loss);
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        SparseCNode* conv_other = (SparseCNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class SparseCExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseCNode* ptr = (SparseCNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            SparseCNode* ptr = (SparseCNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute SparseCNode::generate(bool bTrain) {
    SparseCExecute* exec = new SparseCExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}

#endif /* SparseCOP_H_ */
