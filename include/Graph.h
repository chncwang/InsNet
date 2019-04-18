#ifndef BasicGraph
#define BasicGraph

#include "Eigen/Dense"
#include "Node.h"
#include "MyLib.h"
#include <set>
#include <map>
#include <memory>
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

class Graph {
protected:
    vector<PExecute> execs;
    vector<Node *> nodes;
    NodeMap free_nodes;
    std::map<size_t, std::pair<int, int>> node_type_depth;
    vector<PNode> finish_nodes;
    vector<PNode> all_nodes;

public:
    Graph() = default;

    virtual ~Graph() {
        int count = execs.size();
        for (int idx = 0; idx < count; idx++) {
            delete execs.at(idx);
        }

        for (Node *n : nodes) {
            delete n;
        }
    }

    void backward() {
        int count = execs.size();
        for (int idx = count - 1; idx >= 0; idx--) {
            execs.at(idx)->backwardFully();
        }
    }

    void addNode(Node *x) {
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

    void compute(bool log = false) {
//        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();

        int i = 0;
        while (Size(free_nodes) > 0) {
            if (log)
            std::cout << "i:" << i++ << std::endl;
            float min_avg_depth = 100000000;
            std::vector<Node*> shallow_nodes;
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
            Node *first_node = shallow_nodes.at(0);
            if (log) {
                std::cout << "Graph compute first_node node type:" << first_node->node_type <<
                    std::endl;
                std::cout << "node size:" << shallow_nodes.size() << std::endl;
            }
            PExecute cur_exec = first_node->generate();
            cur_exec->batch = std::move(shallow_nodes);
            free_nodes.erase(min_hash);

            //profiler.BeginEvent("forward");
            cur_exec->forwardFully();
            //profiler.EndEvent();
            execs.push_back(cur_exec);

            //finished nodes
            for (Node* free_node : cur_exec->batch) {
                finish_nodes.push_back(free_node);
                for (auto parent_it : free_node->parents) {
                    if (parent_it->degree <= 0) {
                        abort();
                    }
                    parent_it->degree--;
                    if (parent_it->degree == 0) {
                        Insert(parent_it, free_nodes);
                    }
                }
            }
        }

        if (finish_nodes.size() != all_nodes.size()) {
            std::cout << "error: several nodes are not executed, finished: " << finish_nodes.size() << ", all: " << all_nodes.size() << std::endl;
            int total_node_num = all_nodes.size();
            int unprocessed = 0;
            for (int idx = 0; idx < total_node_num; idx++) {
                PNode curNode = all_nodes.at(idx);
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

#endif
