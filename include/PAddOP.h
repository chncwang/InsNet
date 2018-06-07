#ifndef PAddOP
#define PAddOP

/*
*  PAddOP.h:
*  (pointwise) add
*
*  Created on: June 13, 2017
*      Author: mszhang
*/

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class PAddNode : public Node {
  public:
    vector<PNode> ins;

    ~PAddNode() {
        ins.clear();
    }
  public:
    PAddNode() : Node() {
        ins.clear();
        node_type = "point-add";
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
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

};

//#if USE_GPU
//class PAddExecute : public Execute {
//public:
//  bool bTrain;
//public:
//  inline void  forward() {
//    int count = batch.size();
//
//    for (int idx = 0; idx < count; idx++) {
//      PAddNode* ptr = (PAddNode*)batch[idx];
//      ptr->compute();
//      ptr->forward_drop(bTrain);
//    }
//  }
//
//  inline void backward() {
//    int count = batch.size();
//    for (int idx = 0; idx < count; idx++) {
//      PAddNode* ptr = (PAddNode*)batch[idx];
//      ptr->backward_drop();
//      ptr->backward();
//    }
//  }
//};
//
//
//inline PExecute PAddNode::generate(bool bTrain) {
//  PAddExecute* exec = new PAddExecute();
//  exec->batch.push_back(this);
//  exec->bTrain = bTrain;
//  return exec;
//}
//#else
class PAddExecute : public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            PAddNode* ptr = (PAddNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            PAddNode* ptr = (PAddNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};


inline PExecute PAddNode::generate(bool bTrain) {
    PAddExecute* exec = new PAddExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}
//#endif


#endif
