#include "n3ldg-plus/computation-graph/graph.h"

using std::map;
using std::pair;
using std::make_pair;
using std::string;
using std::vector;
using std::cerr;
using std::endl;

namespace n3ldg_plus {

namespace {

void Insert(NodeAbs *node, NodeMap& node_map) {
    string x_hash = node->cachedTypeSig();
    auto it = node_map.find(x_hash);
    if (it == node_map.end()) {
        vector<NodeAbs *> v = {node};
        node_map.insert(make_pair<string, vector<NodeAbs *>>(move(x_hash),
                    move(v)));
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

}

Graph::Graph(bool eager, bool calculate_flops, bool calculate_activations) : eager_(eager),
    calculate_flops_(calculate_flops), calculate_activations_(calculate_activations) {}

Graph::~Graph() {
    int count = execs.size();
    for (int idx = 0; idx < count; idx++) {
        delete execs.at(idx);
    }

    if (globalPoolEnabled()) {
        for (NodeAbs *node : finish_nodes) {
            if (node->isBatched()) {
                for (Node *inner : node->batch()) {
                    if (!inner->isPooled()) {
                        delete inner;
                    }
                }
                delete node;
            } else if (!node->isPooled()) {
                delete node;
            }
        }
        auto &refs = globalPoolReferences();
        for (auto &e : refs) {
            e->second = 0;
        }
    } else {
        for (NodeAbs *node : finish_nodes) {
            if (node->isBatched()) {
                for (Node *x : node->batch()) {
                    delete x;
                }
                delete node;
            } else {
                delete node;
            }
        }
    }
}

void Graph::backward() {
    int count = execs.size();
    for (int idx = count - 1; idx >= 0; idx--) {
        execs.at(idx)->backwardFully();
    }
}

void Graph::addNode(NodeAbs *x) {
    if (x == nullptr) {
        cerr << "x is nullptr" << endl;
        abort();
    }
    if (x->getDegree() == 0) {
        Insert(x, free_nodes);
    }
    ++all_nodes_count;

    string x_type_hash = x->cachedTypeSig();
    auto it = node_type_depth.find(x_type_hash);
    if (it == node_type_depth.end()) {
        node_type_depth.insert(pair<string, pair<int, int>>(
                    x_type_hash, pair<int, int>(x->getDepth(), 1)));
    } else {
        it->second.first += x->getDepth();
        it->second.second++;
    }

    if (eager_) {
        forward();
    }
}

void Graph::forward() {
    while (true) {
        if (Size(free_nodes) <= 0) {
            break;
        }
        float min_avg_depth = 100000000;
        vector<NodeAbs *> shallow_nodes;
        string min_hash;
        for (auto it : free_nodes) {
            string type_hash = it.first;
            auto depth_it = node_type_depth.find(type_hash);
            if (depth_it == node_type_depth.end()) {
                cerr << fmt::format("type not found in depth map:{}\n", type_hash);
                abort();
            }
            float avg_depth = (float)depth_it->second.first / depth_it->second.second;
            if (avg_depth < min_avg_depth) {
                min_avg_depth = avg_depth;
                shallow_nodes = it.second;
                min_hash = type_hash;
            }
        }
        //            cout << "type:" <<min_hash << " " << shallow_nodes.size() << endl;
        NodeAbs *first_node = shallow_nodes.front();
        Executor *cur_exec = first_node->generate();
        cur_exec->batch.clear();
        cur_exec->topo_nodes = shallow_nodes;
        if (first_node->isBatched()) {
            for (NodeAbs *node : shallow_nodes) {
                auto &v = node->batch();
                for (Node *atom : v) {
                    cur_exec->batch.push_back(atom);
                }
            }
        } else {
            cur_exec->batch.reserve(shallow_nodes.size());
            for (NodeAbs *node : shallow_nodes) {
                cur_exec->batch.push_back(dynamic_cast<Node *>(node));
            }
        }
        free_nodes.erase(min_hash);
#if USE_GPU
        profiler.BeginEvent("clear nodes");
        clearNodes(cur_exec->batch);
        profiler.EndCudaEvent();
#endif
        //            cout << "type:" << cur_exec->getSignature() << " " << cur_exec->batch.size() << endl << endl;

        cur_exec->forwardFully();
        if (eager_) {
            for (Node *node : cur_exec->batch) {
                node->getVal().checkIsNumber();
            }
        }
        if (calculate_flops_) {
#if !USE_GPU
            const auto &it = flops_table_.find(cur_exec->getNodeType());
            if (it == flops_table_.end()) {
                flops_table_.insert(make_pair(cur_exec->getNodeType(),
                            cur_exec->calculateFLOPs()));
            } else {
                it->second += cur_exec->calculateFLOPs();
            }
#endif
        }
        if (calculate_activations_) {
#if !USE_GPU
            activations_ += cur_exec->calculateActivations();
#endif
        }

        execs.push_back(cur_exec);

        int depth_sum = 0;
        for (NodeAbs* free_node : cur_exec->topo_nodes) {
            finish_nodes.push_back(free_node);
            depth_sum += free_node->getDepth();
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

        auto &it = node_type_depth.at(cur_exec->topo_nodes.front()->cachedTypeSig());
        it.first -= depth_sum;
        it.second -= cur_exec->topo_nodes.size();
        if (it.first < 0 || it.second < 0) {
            cerr << fmt::format("Graph compute - it first:{} second:{}", it.first,
                    it.second);
            abort();
        }
    }

    if (finish_nodes.size() != all_nodes_count) {
        cerr << "error: several nodes are not executed, finished: " <<
            finish_nodes.size() << ", all: " << all_nodes_count << endl;
        abort();
    }
}

void Graph::addFLOPs(int64_t flops, const string &name) {
    if (calculate_flops_) {
        const auto &it = flops_table_.find(name);
        if (it == flops_table_.end()) {
            flops_table_.insert(make_pair(name, flops));
        } else {
            it->second += flops;
        }
    }
}

}
