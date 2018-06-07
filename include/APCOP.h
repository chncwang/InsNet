/*
 * APCOP.h
 *
 *  Created on: Jul 20, 2016
 *      Author: mszhang
 */

#ifndef APCOP_H_
#define APCOP_H_

#include "COPUtils.h"
#include "Alphabet.h"
#include "Node.h"
#include "Graph.h"
#include "APParam.h"

class APC1Params {
  public:
    APParam W;
    int nDim;
    unordered_map<C1Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    APC1Params() {
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

        if(nVSize > 0 && bound < nVSize) {
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
class APC1Node : public Node {
  public:
    APC1Params* param;
    int tx;
    bool executed;

  public:
    APC1Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "apc1";
    }

    inline void setParam(APC1Params* paramInit) {
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
    inline void compute(bool bTrain) {
        param->W.value(tx, val, bTrain);
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

        APC1Node* conv_other = (APC1Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class APC1Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC1Node* ptr = (APC1Node*)batch[idx];
            ptr->compute(bTrain);
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC1Node* ptr = (APC1Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute APC1Node::generate(bool bTrain) {
    APC1Execute* exec = new APC1Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}



//a sparse feature has two actomic features
class APC2Params {
  public:
    APParam W;
    int nDim;
    unordered_map<C2Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    APC2Params() {
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
class APC2Node : public Node {
  public:
    APC2Params* param;
    int tx;
    bool executed;

  public:
    APC2Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "apc2";
    }

    inline void setParam(APC2Params* paramInit) {
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
    inline void compute(bool bTrain) {
        param->W.value(tx, val, bTrain);
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

        APC2Node* conv_other = (APC2Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class APC2Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC2Node* ptr = (APC2Node*)batch[idx];
            ptr->compute(bTrain);
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC2Node* ptr = (APC2Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute APC2Node::generate(bool bTrain) {
    APC2Execute* exec = new APC2Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}


//a sparse feature has three actomic features
class APC3Params {
  public:
    APParam W;
    int nDim;
    unordered_map<C3Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    APC3Params() {
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
class APC3Node : public Node {
  public:
    APC3Params* param;
    int tx;
    bool executed;

  public:
    APC3Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "apc3";
    }

    inline void setParam(APC3Params* paramInit) {
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
    inline void compute(bool bTrain) {
        param->W.value(tx, val, bTrain);
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

        APC3Node* conv_other = (APC3Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class APC3Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC3Node* ptr = (APC3Node*)batch[idx];
            ptr->compute(bTrain);
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC3Node* ptr = (APC3Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute APC3Node::generate(bool bTrain) {
    APC3Execute* exec = new APC3Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}


//a sparse feature has four actomic features
class APC4Params {
  public:
    APParam W;
    int nDim;
    unordered_map<C4Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    APC4Params() {
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
class APC4Node : public Node {
  public:
    APC4Params* param;
    int tx;
    bool executed;

  public:
    APC4Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "apc4";
    }

    inline void setParam(APC4Params* paramInit) {
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
    inline void compute(bool bTrain) {
        param->W.value(tx, val, bTrain);
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

        APC4Node* conv_other = (APC4Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class APC4Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC4Node* ptr = (APC4Node*)batch[idx];
            ptr->compute(bTrain);
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC4Node* ptr = (APC4Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute APC4Node::generate(bool bTrain) {
    APC4Execute* exec = new APC4Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}


//a sparse feature has five actomic features
class APC5Params {
  public:
    APParam W;
    int nDim;
    unordered_map<C5Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    APC5Params() {
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
class APC5Node : public Node {
  public:
    APC5Params* param;
    int tx;
    bool executed;

  public:
    APC5Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "apc5";
    }

    inline void setParam(APC5Params* paramInit) {
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
    inline void compute(bool bTrain) {
        param->W.value(tx, val, bTrain);
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

        APC5Node* conv_other = (APC5Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class APC5Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC5Node* ptr = (APC5Node*)batch[idx];
            ptr->compute(bTrain);
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC5Node* ptr = (APC5Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute APC5Node::generate(bool bTrain) {
    APC5Execute* exec = new APC5Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}


//a sparse feature has six actomic features
class APC6Params {
  public:
    APParam W;
    int nDim;
    unordered_map<C6Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    APC6Params() {
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
class APC6Node : public Node {
  public:
    APC6Params* param;
    int tx;
    bool executed;

  public:
    APC6Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "apc6";
    }

    inline void setParam(APC6Params* paramInit) {
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
    inline void compute(bool bTrain) {
        param->W.value(tx, val, bTrain);
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

        APC6Node* conv_other = (APC6Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class APC6Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC6Node* ptr = (APC6Node*)batch[idx];
            ptr->compute(bTrain);
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC6Node* ptr = (APC6Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute APC6Node::generate(bool bTrain) {
    APC6Execute* exec = new APC6Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}

//a sparse feature has seven actomic features
class APC7Params {
  public:
    APParam W;
    int nDim;
    unordered_map<C7Feat, int> hash2id;
    int nVSize;
    int bound;
  public:
    APC7Params() {
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
class APC7Node : public Node {
  public:
    APC7Params* param;
    int tx;
    bool executed;

  public:
    APC7Node() : Node() {
        param = NULL;
        tx = -1;
        executed = false;
        node_type = "apc7";
    }

    inline void setParam(APC7Params* paramInit) {
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
    inline void compute(bool bTrain) {
        param->W.value(tx, val, bTrain);
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

        APC7Node* conv_other = (APC7Node*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class APC7Execute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC7Node* ptr = (APC7Node*)batch[idx];
            ptr->compute(bTrain);
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APC7Node* ptr = (APC7Node*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute APC7Node::generate(bool bTrain) {
    APC7Execute* exec = new APC7Execute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}

class APCParams {
  public:
    APParam W;
    int nDim;
    unordered_map<CFeat, int> hash2id;
    int nVSize;
    int bound;
  public:
    APCParams() {
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
class APCNode : public Node {
  public:
    APCParams* param;
    vector<int> ins;
    int nSize;

  public:
    APCNode() : Node() {
        param = NULL;
        ins.clear();
        nSize = 0;
        node_type = "apclist";
    }

    inline void setParam(APCParams* paramInit) {
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
    inline void compute(bool bTrain) {
        param->W.value(ins, val, bTrain);
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

        APCNode* conv_other = (APCNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }
};

class APCExecute :public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APCNode* ptr = (APCNode*)batch[idx];
            ptr->compute(bTrain);
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            APCNode* ptr = (APCNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute APCNode::generate(bool bTrain) {
    APCExecute* exec = new APCExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}

#endif /* APCOP_H_ */
