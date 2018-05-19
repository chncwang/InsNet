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
#include <set>
#include <map>
#include <unordered_map>
#include "profiler.h"
#include <vector>

using namespace Eigen;

int GetDegree(std::map<void*, int> &degree_map, PNode p) {
    auto it = degree_map.find(p);
    if (it == degree_map.end()) {
        degree_map.insert(std::pair<void*, int>(p, p->degree));
        return p->degree;
    } else {
        return it->second;
    }
}

void DecreaseDegree(std::map<void*, int> &degree_map, PNode p) {
    auto it = degree_map.find(p);
    if (it == degree_map.end()) {
        degree_map.insert(std::pair<void*, int>(p, p->degree - 1));
    } else {
        --(it->second);
    }
}

struct SelfHash {
    size_t operator()(size_t hash) const {
        return hash;
    }
};

typedef std::unordered_map<size_t, vector<PNode>, SelfHash> NodeMap;

void Insert(const PNode node, NodeMap& node_map) {
    size_t x_hash = node->typeHashCode();
    auto it = node_map.find(x_hash);
    if (it == node_map.end()) {
        std::vector<PNode> v = {node};
        node_map.insert(std::make_pair<size_t, std::vector<PNode>>(std::move(x_hash), std::move(v)));
    } else {
        it->second.push_back(node);
    }
}

int Size(const NodeMap &map) {
    int sum = 0;
    for (auto it : map) {
        sum += it.second.size();
    }
    return sum;
}

// one Node means a vector
// the col should be 1, because we aimed for NLP only
class Graph {
  protected:
    vector<PExecute> execs; //backward
    vector<PNode> nodes; //forward
    NodeMap free_nodes;
    vector<PNode> finish_nodes;
    vector<PNode> all_nodes;

  public:
    bool train;
    dtype drop_factor;

  public:
    Graph() {
        drop_factor = 1.0;
    }

    virtual ~Graph() {
        int count = execs.size();
        for (int idx = 0; idx < count; idx++) {
            delete execs.at(idx);
        }
        execs.clear();
        nodes.clear();
        free_nodes.clear();
    }


    void setDropFactor(dtype cur_drop_factor) {
        drop_factor = cur_drop_factor;
        if (drop_factor <= 0) drop_factor = 0;
        if (drop_factor >= 1.0) drop_factor = 1.0;
    }

  public:
    void clearValue(const bool& bTrain = false) {
        NodeMap node_map;
        for (Node *node : nodes) {
            Insert(node, node_map);
        }
        for (auto it : node_map) {
            PExecute new_exec = it.second.at(0)->generate(train,
                    drop_factor);
            new_exec->batch = it.second;
            new_exec->clearValue();
            delete new_exec;
        }


        int count = execs.size();
        for (int idx = 0; idx < count; idx++) {
            delete execs.at(idx);
        }
        execs.clear();
        nodes.clear();
        free_nodes.clear();
        finish_nodes.clear();
        all_nodes.clear();

        train = bTrain;
    }

    void backward() {
        int count = execs.size();
        for (int idx = count - 1; idx >= 0; idx--) {
            execs.at(idx)->backward();
        }
    }

    void addNode(PNode x) {
        static int index;
        x->node_index = index++;
        nodes.push_back(x);
        if (x->degree == 0) {
            Insert(x, free_nodes);
        }
        all_nodes.push_back(x);
    }

    //real executation
    void compute() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();

        while (Size(free_nodes) > 0) {
            vector<PExecute> cur_execs;
            for (auto it : free_nodes) {
                PExecute new_exec = it.second.at(0)->generate(train,
                        drop_factor);
                new_exec->batch = it.second;
                cur_execs.push_back(new_exec);
            }

            for (PExecute e : cur_execs) {
                //profiler.BeginEvent("forward");
                e->forwardFully();
                //profiler.EndEvent();
                execs.push_back(e);
            }

            //finished nodes
            NodeMap new_free_nodes;
            for (auto vec_it : free_nodes) {
                for (auto free_node_it : vec_it.second) {
                    finish_nodes.push_back(free_node_it);
                    for (auto parent_it : free_node_it->parents) {
                        if (parent_it->degree <= 0) {
                            abort();
                        }
                        parent_it->degree--;
                        if (parent_it->degree == 0) {
                            Insert(parent_it, new_free_nodes);
                        }
                    }
                }
            }

            // update free nodes
            free_nodes = std::move(new_free_nodes);
        }

        if (finish_nodes.size() != all_nodes.size()) {
            std::cout << "error: several nodes are not executed, finished: " << finish_nodes.size() << ", all: " << all_nodes.size() << std::endl;
            int total_node_num = all_nodes.size();
            int unprocessed = 0;
            for (int idx = 0; idx < total_node_num; idx++) {
                PNode curNode = all_nodes.at(idx);
                if (curNode->degree > 0) {
                    std::cout << "unprocessed node:" << curNode->node_type << " degree:" << curNode->degree << std::endl;
                    unprocessed++;
                }
            }
            std::cout << "unprocessed: " << unprocessed << std::endl;
            abort();
        }
    }
};


// one very useful function to collect pointers of derived nodes
template<typename DerivedNode>
vector<PNode> getPNodes(vector<DerivedNode>& inputs, int size) {
    int usedSize = inputs.size();
    if (size >= 0 && size < usedSize) usedSize = size;
    vector<PNode> pnodes;
    for (int idx = 0; idx < usedSize; idx++) {
        pnodes.push_back(&(inputs.at(idx)));
    }

    return pnodes;
}

template<typename DerivedNode>
vector<PNode> getPNodes(DerivedNode inputs[], int size) {
    //int usedSize = inputs.;
    //if (size >= 0 && size < usedSize) usedSize = size;
    int usedSize = size;
    vector<PNode> pnodes;
    for (int idx = 0; idx < usedSize; idx++) {
        pnodes.push_back(&(inputs.at(idx)));
    }

    return pnodes;
}

template<typename DerivedNode>
vector<PNode> getPNodes(vector<DerivedNode>& inputs, int start, int length) {
    int end, tmp_end = start + length;
    if (tmp_end > inputs.size())
        end = inputs.size();
    else
        end = tmp_end;
    //if (size >= 0 && size < usedSize) usedSize = size;
    vector<PNode> pnodes;
    for (int idx = start; idx < end; idx++) {
        pnodes.push_back(&(inputs.at(idx)));
    }

    return pnodes;
}

template<typename DerivedNode>
vector<PNode> getPNodes(DerivedNode inputs[], int size, int start, int length) {
    int end, tmp_end = start + length;
    if (tmp_end > size)
        end = size;
    else
        end = tmp_end;
    //if (size >= 0 && size < usedSize) usedSize = size;
    vector<PNode> pnodes;
    for (int idx = start; idx < end; idx++) {
        pnodes.push_back(&(inputs.at(idx)));
    }

    return pnodes;
}
#endif
