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
        degree_map.insert(std::pair<void*, int>(p, p->getDegree()));
        return p->getDegree();
    } else {
        return it->second;
    }
}

void DecreaseDegree(std::map<void*, int> &degree_map, PNode p) {
    auto it = degree_map.find(p);
    if (it == degree_map.end()) {
        degree_map.insert(std::pair<void*, int>(p, p->getDegree() - 1));
    } else {
        --(it->second);
    }
}

typedef std::unordered_map<string, vector<PNode>> NodeMap;

void Insert(const PNode node, NodeMap& node_map) {
    string x_hash = node->typeSignature();
    auto it = node_map.find(x_hash);
    if (it == node_map.end()) {
        std::vector<PNode> v = {node};
        node_map.insert(std::make_pair<string, std::vector<PNode>>(std::move(x_hash), std::move(v)));
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

class Graph : public NodeContainer {
public:
    Graph(bool eager = false) : eager_(eager) {}

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

    void addNode(Node *x) override {
        if (x == nullptr) {
            cerr << "x is nullptr" << endl;
            abort();
        }
        static int index;
        x->setNodeIndex(index++);
        nodes.push_back(x);
        if (x->getDegree() == 0) {
            Insert(x, free_nodes);
        }
        all_nodes.push_back(x);

        string x_type_hash = x->typeSignature();
        auto it = node_type_depth.find(x_type_hash);
        if (it == node_type_depth.end()) {
            node_type_depth.insert(std::pair<string, std::pair<int, int>>(
                        x_type_hash, std::pair<int, int>(x->getDepth(), 1)));
        } else {
            it->second.first += x->getDepth();
            it->second.second++;
        }

        if (eager_) {
            compute();
        }
    }

    void compute() {
        while (Size(free_nodes) > 0) {
            float min_avg_depth = 100000000;
            std::vector<Node*> shallow_nodes;
            string min_hash;
            for (auto it : free_nodes) {
                string type_hash = it.first;
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
            PExecutor cur_exec = first_node->generate();
            cur_exec->batch = std::move(shallow_nodes);
            free_nodes.erase(min_hash);
#if USE_GPU
            clearNodes(cur_exec->batch, cur_exec->getDim());
#endif
            cur_exec->forwardFully();
            execs.push_back(cur_exec);

            for (Node* free_node : cur_exec->batch) {
                finish_nodes.push_back(free_node);
                for (auto parent_it : free_node->getParents()) {
                    if (parent_it->getDegree() <= 0) {
                        abort();
                    }
                    parent_it->setDegree(parent_it->getDegree() - 1);
                    if (parent_it->getDegree() == 0) {
                        Insert(parent_it, free_nodes);
                    }
                }
            }
        }

        if (finish_nodes.size() != all_nodes.size()) {
            std::cerr << "error: several nodes are not executed, finished: " <<
                finish_nodes.size() << ", all: " << all_nodes.size() << std::endl;
            int total_node_num = all_nodes.size();
            int unprocessed = 0;
            for (int idx = 0; idx < total_node_num; idx++) {
                PNode curNode = all_nodes.at(idx);
                if (curNode->getDegree() > 0) {
                    std::cerr << "unprocessed node:" << curNode->getNodeType() <<
                        " degree:" << curNode->getDegree() <<
                        " name:" << curNode->getNodeName() <<
                        std::endl;
                    unprocessed++;
                }
            }
            std::cerr << "unprocessed: " << unprocessed << std::endl;
            abort();
        }
    }

protected:
    vector<PExecutor> execs;
    vector<Node *> nodes;
    NodeMap free_nodes;
    std::map<string, std::pair<int, int>> node_type_depth;
    vector<PNode> finish_nodes;
    vector<PNode> all_nodes;

private:
    bool eager_ = false;
};

#endif
