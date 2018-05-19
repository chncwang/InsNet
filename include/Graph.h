#ifndef BasicGraph
#define BasicGraph

/*
*  Graph.h:
*  manage nodes in a neural network model
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/


#include "Eigen/Dense"
#include "Node.h"
#include "MyLib.h"

using namespace Eigen;


// one Node means a vector
// the col should be 1, because we aimed for NLP only
class Graph {
  protected:
    vector<PExecute> execs; //backward
    vector<PNode> nodes; //forward
    vector<PNode> free_nodes;
    vector<PNode> finish_nodes;
    vector<PNode> all_nodes;

  public:
    bool train;

  public:
    Graph() {
        execs.clear();
        execs.clear();
        nodes.clear();
        free_nodes.clear();
    }

    virtual ~Graph() {
        int count = execs.size();
        for (int idx = 0; idx < count; idx++) {
            delete execs[idx];
        }
        execs.clear();
        execs.clear();
        nodes.clear();
        free_nodes.clear();
    }

  public:
    inline void clearValue(const bool& bTrain = false) {
        int count = execs.size();
        for (int idx = 0; idx < count; idx++) {
            delete execs[idx];
        }
        execs.clear();

        count = nodes.size();
        for (int idx = 0; idx < count; idx++) {
            nodes[idx]->clearValue();
        }
        nodes.clear();
        free_nodes.clear();
        finish_nodes.clear();
        all_nodes.clear();

        train = bTrain;
    }

    inline void backward() {
        int count = execs.size();
        for (int idx = count - 1; idx >= 0; idx--) {
            execs[idx]->backward();
        }
    }

    inline void addNode(PNode x) {
        nodes.push_back(x);
        if (x->degree == 0) {
            free_nodes.push_back(x);
        }
        all_nodes.push_back(x);
    }

    //real executation
    inline void compute() {
        int free_count = free_nodes.size();

        while (free_count > 0) {
            vector<PExecute> cur_execs;
            int cur_execs_size = 0;

            for (int idx = 0; idx < free_count; idx++) {
                bool find = false;
                for (int idy = 0; idy < cur_execs_size; idy++) {
                    if (cur_execs[idy]->addNode(free_nodes[idx])) {
                        find = true;
                        break;
                    }
                }

                if (!find) {
                    PExecute new_exec = free_nodes[idx]->generate(train);
                    cur_execs.push_back(new_exec);
                    cur_execs_size++;
                }

            }

            //execute
//#pragma omp parallel for
            for (int idy = 0; idy < cur_execs_size; idy++) {
                cur_execs[idy]->forward();
            }

            for (int idy = 0; idy < cur_execs_size; idy++) {
                execs.push_back(cur_execs[idy]);
            }

            //finished nodes
            vector<PNode> new_free_nodes;
            for (int idx = 0; idx < free_count; idx++) {
                finish_nodes.push_back(free_nodes[idx]);
                int parent_count = free_nodes[idx]->parents.size();
                for (int idy = 0; idy < parent_count; idy++) {
                    free_nodes[idx]->parents[idy]->degree--;
                    if (free_nodes[idx]->parents[idy]->degree == 0) {
                        new_free_nodes.push_back(free_nodes[idx]->parents[idy]);
                    }
                }
            }

            // update free nodes
            free_nodes.clear();
            free_count = new_free_nodes.size();
            for (int idx = 0; idx < free_count; idx++) {
                free_nodes.push_back(new_free_nodes[idx]);
            }

        }

        if (finish_nodes.size() != all_nodes.size()) {
            std::cout << "error: several nodes are not executed, finished: " << finish_nodes.size() << ", all: " << all_nodes.size() << std::endl;
            int total_node_num = all_nodes.size();
            int unprocessed = 0;
            for (int idx = 0; idx < total_node_num; idx++) {
                PNode curNode = all_nodes[idx];
                if (curNode->degree >= 0) {
                    curNode->typeEqual(all_nodes[0]);
                    unprocessed++;
                }
            }
            std::cout << "unprocessed: " << unprocessed << std::endl;
        }
    }

};




// one very useful function to collect pointers of derived nodes
template<typename DerivedNode>
inline vector<PNode> getPNodes(vector<DerivedNode>& inputs, int size) {
    int usedSize = inputs.size();
    if (size >= 0 && size < usedSize) usedSize = size;
    vector<PNode> pnodes;
    for (int idx = 0; idx < usedSize; idx++) {
        pnodes.push_back(&(inputs[idx]));
    }

    return pnodes;
}

template<typename DerivedNode>
inline vector<PNode> getPNodes(DerivedNode inputs[], int size) {
    //int usedSize = inputs.;
    //if (size >= 0 && size < usedSize) usedSize = size;
    int usedSize = size;
    vector<PNode> pnodes;
    for (int idx = 0; idx < usedSize; idx++) {
        pnodes.push_back(&(inputs[idx]));
    }

    return pnodes;
}

template<typename DerivedNode>
inline vector<PNode> getPNodes(vector<DerivedNode>& inputs, int start, int length) {
    int end, tmp_end = start + length;
    if (tmp_end > inputs.size())
        end = inputs.size();
    else
        end = tmp_end;
    //if (size >= 0 && size < usedSize) usedSize = size;
    vector<PNode> pnodes;
    for (int idx = start; idx < end; idx++) {
        pnodes.push_back(&(inputs[idx]));
    }

    return pnodes;
}

template<typename DerivedNode>
inline vector<PNode> getPNodes(DerivedNode inputs[], int size, int start, int length) {
    int end, tmp_end = start + length;
    if (tmp_end > size)
        end = size;
    else
        end = tmp_end;
    //if (size >= 0 && size < usedSize) usedSize = size;
    vector<PNode> pnodes;
    for (int idx = start; idx < end; idx++) {
        pnodes.push_back(&(inputs[idx]));
    }

    return pnodes;
}
#endif
