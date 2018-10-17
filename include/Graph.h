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

template<typename NodePointer>
using NodeMap = std::unordered_map<size_t, vector<NodePointer>, SelfHash>;

template<typename NodePointer>
void Insert(const NodePointer node, NodeMap<NodePointer>& node_map) {
    size_t x_hash = node->typeHashCode();
    auto it = node_map.find(x_hash);
    if (it == node_map.end()) {
        std::vector<NodePointer> v = {node};
        node_map.insert(std::make_pair<size_t, std::vector<NodePointer>>(std::move(x_hash),
                    std::move(v)));
    } else {
        it->second.push_back(node);
    }
}

template<typename NodePointer>
int Size(const NodeMap<NodePointer> &map) {
    int sum = 0;
    for (auto it : map) {
        sum += it.second.size();
    }
    return sum;
}

template <typename T>
std::vector<Node *> toWeakPointers(std::vector<T> &x) {
    std::vector<Node *> nodes;
    for (T& t : x) {
        nodes.push_back(t.get());
    }
    return nodes;
}

// one Node means a vector
// the col should be 1, because we aimed for NLP only
template <typename NodePointer>
class TemplateGraph {
  protected:
    vector<PExecute> execs; //backward
    vector<NodePointer> nodes; //forward
    NodeMap<NodePointer> free_nodes;
    std::map<size_t, std::pair<int, int>> node_type_depth;
    vector<NodePointer> finish_nodes;
    vector<NodePointer> all_nodes;

  public:
    dtype drop_factor;
    bool train;

  public:
    TemplateGraph() {
        drop_factor = 1.0;
    }

    virtual ~TemplateGraph() {
        int count = execs.size();
        for (int idx = 0; idx < count; idx++) {
            delete execs.at(idx);
        }
    }

    void setDropFactor(dtype cur_drop_factor) {
        drop_factor = cur_drop_factor;
        if (drop_factor <= 0) drop_factor = 0;
        if (drop_factor >= 1.0) drop_factor = 1.0;
    }

    void backward() {
        int count = execs.size();
        for (int idx = count - 1; idx >= 0; idx--) {
            execs.at(idx)->backward();
        }
    }

    void addNode(NodePointer x) {
        static int index;
        x->node_index = index++;
        nodes.push_back(x);
        if (x->degree == 0) {
            Insert(x, free_nodes);
        }
        all_nodes.push_back(x);

        size_t x_type_hash = x->typeHashCode();
        auto it = node_type_depth.find(x_type_hash);
        if (it == node_type_depth.end()) {
            node_type_depth.insert(std::pair<size_t, std::pair<int, int>>(
                        x_type_hash, std::pair<int, int>(x->depth, 1)));
        } else {
            it->second.first += x->depth;
            it->second.second++;
        }
    }

    //real executation
    void compute(bool log = false) {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();

        int i = 0;
        while (Size(free_nodes) > 0) {
            if (log)
            std::cout << "i:" << i++ << std::endl;
            float min_avg_depth = 100000000;
            std::vector<NodePointer> shallow_nodes;
            size_t min_hash = 0;
            for (auto it : free_nodes) {
                size_t type_hash = it.first;
                auto depth_it = node_type_depth.find(type_hash);
                float avg_depth = (float)depth_it->second.first /
                    depth_it->second.second;
                if (avg_depth < min_avg_depth) {
                    min_avg_depth = avg_depth;
                    shallow_nodes = it.second;
                    min_hash = type_hash;
                }
            }
            NodePointer first_node = shallow_nodes.at(0);
            if (log) {
                std::cout << "Graph compute first_node node type:" << first_node->node_type <<
                    std::endl;
                std::cout << "node size:" << shallow_nodes.size() << std::endl;
            }
            PExecute cur_exec = first_node->generate(train, drop_factor);
            cur_exec->batch = toWeakPointers(shallow_nodes);
            free_nodes.erase(min_hash);

            //profiler.BeginEvent("forward");
            cur_exec->forwardFully();
            //profiler.EndEvent();
            execs.push_back(cur_exec);

            //finished nodes
            for (Node *free_node : cur_exec->batch) {
                finish_nodes.push_back(NodePointer(free_node));
                for (auto parent_it : free_node->parents) {
                    if (parent_it->degree <= 0) {
                        abort();
                    }
                    parent_it->degree--;
                    if (parent_it->degree == 0) {
                        Insert(NodePointer(parent_it), free_nodes);
                    }
                }
            }
        }

        if (finish_nodes.size() != all_nodes.size()) {
            std::cout << "error: several nodes are not executed, finished: " << finish_nodes.size() << ", all: " << all_nodes.size() << std::endl;
            int total_node_num = all_nodes.size();
            int unprocessed = 0;
            for (int idx = 0; idx < total_node_num; idx++) {
                NodePointer curNode = all_nodes.at(idx);
                if (curNode->degree > 0) {
                    std::cout << "unprocessed node:" << curNode->node_type <<
                        " degree:" << curNode->degree <<
                        " name:" << curNode->node_name <<
                        std::endl;
                    unprocessed++;
                }
            }
            std::cout << "unprocessed: " << unprocessed << std::endl;
            abort();
        }
    }
};

struct Graph : public TemplateGraph<std::shared_ptr<Node>> {
    void addNode(Node *node) {
        TemplateGraph<std::shared_ptr<Node>>::addNode(std::shared_ptr<Node>(node));
    }
};

#endif
