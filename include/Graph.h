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
#if USE_GPU
    void *host_memory = NULL;
    void *device_memory = NULL;
#endif

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


    inline void setDropFactor(dtype cur_drop_factor) {
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

        //std::set<PNode> uncleared_nodes;
        //for (PNode p : nodes) {
        //    uncleared_nodes.insert(p);
        //}
        //while (!uncleared_nodes.empty()) {
        //    PNode p = NULL;
        //    PExecute cur_exec;
        //    for (PNode pp : nodes) {
        //        auto find = uncleared_nodes.find(pp);
        //        if (p == NULL && find != uncleared_nodes.end()) {
        //            p = pp;
        //            cur_exec = p->generate(bTrain, -1);
        //            uncleared_nodes.erase(find);
        //        } else if (p != NULL && find != uncleared_nodes.end()) {
        //            if (p->typeEqual(*find)) {
        //                cur_exec->addNode(*find);
        //                uncleared_nodes.erase(find);
        //            }
        //        }
        //    }
        //    cur_exec->clearValue();
        //}

        nodes.clear();
        free_nodes.clear();
        finish_nodes.clear();
        all_nodes.clear();

        train = bTrain;
    }

    inline void backward() {
        int count = execs.size();
        for (int idx = count - 1; idx >= 0; idx--) {
            execs.at(idx)->backward();
        }
    }

    inline void addNode(PNode x) {
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
                e->forward();
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
                if (curNode->degree >= 0) {
                    curNode->typeEqual(all_nodes.at(0));
                    unprocessed++;
                }
            }
            std::cout << "unprocessed: " << unprocessed << std::endl;
            abort();
        }
    }

#if USE_GPU
    void computeNodeInfo(std::vector<std::vector<NodeInfo>> &graph_node_info) const {
        if (!graph_node_info.empty()) {
            abort();
        }

        std::map<void *, int> degree_map;
        NodeMap copied_free_nodes = free_nodes;
        std::vector<PNode> copied_finished_nodes = finish_nodes;
        int free_count = Size(copied_free_nodes);

        while (Size(copied_free_nodes) > 0) {
            vector<PExecute> cur_execs;

            for (auto it : copied_free_nodes) {
                PExecute new_exec = it.second.at(0)->generate(train,
                        drop_factor);
                new_exec->batch = it.second;
                cur_execs.push_back(new_exec);
            }

            for (PExecute e : cur_execs) {
                std::vector<NodeInfo> node_info_vec;
                for (PNode p : e->batch) {
                    NodeInfo info;
                    p->toNodeInfo(info);
                    node_info_vec.push_back(info);
                }
                graph_node_info.push_back(node_info_vec);
            }

            for (PExecute e : cur_execs) {
                delete e;
            }

            //finished nodes
            NodeMap new_free_nodes;
            for (auto vec_it : copied_free_nodes) {
                for (auto free_node_it : vec_it.second) {
                    copied_finished_nodes.push_back(free_node_it);
                    for (auto parent_it : free_node_it->parents) {
                        DecreaseDegree(degree_map, parent_it);
                        int degree = GetDegree(degree_map, parent_it);
                        if (degree < 0) {
                            abort();
                        }
                        if (degree == 0) {
                            Insert(parent_it, new_free_nodes);
                        }
                    }
                }
            }

            copied_free_nodes = std::move(new_free_nodes);
        }

        if (copied_finished_nodes.size() != all_nodes.size()) {
            int total_node_num = all_nodes.size();
            int unprocessed = 0;
            for (int idx = 0; idx < total_node_num; idx++) {
                PNode curNode = all_nodes.at(idx);
                if (GetDegree(degree_map, curNode) >= 0) {
                    curNode->typeEqual(all_nodes.at(0));
                    unprocessed++;
                }
            }
            abort();
        }
    }
#endif
};


// one very useful function to collect pointers of derived nodes
template<typename DerivedNode>
inline vector<PNode> getPNodes(vector<DerivedNode>& inputs, int size) {
    int usedSize = inputs.size();
    if (size >= 0 && size < usedSize) usedSize = size;
    vector<PNode> pnodes;
    for (int idx = 0; idx < usedSize; idx++) {
        pnodes.push_back(&(inputs.at(idx)));
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
        pnodes.push_back(&(inputs.at(idx)));
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
        pnodes.push_back(&(inputs.at(idx)));
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
        pnodes.push_back(&(inputs.at(idx)));
    }

    return pnodes;
}
#endif
