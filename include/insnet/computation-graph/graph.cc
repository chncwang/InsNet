#include "insnet/computation-graph/graph.h"
#include "insnet/util/profiler.h"

using std::map;
using std::pair;
using std::make_pair;
using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

namespace insnet {

namespace {

void Insert(NodeAbs *node, NodeMap& node_map) {
    const string &x_hash = node->cachedTypeSig();
    auto it = node_map.find(x_hash);
    if (it == node_map.end()) {
        vector<NodeAbs *> v = {node};
        string x_hash_copy = x_hash;
        node_map.insert(make_pair<string, vector<NodeAbs *>>(move(x_hash_copy), move(v)));
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

Graph::Graph(ModelStage stage, bool eager, bool calculate_flops, bool calculate_activations) :
    NodeContainer(stage), eager_(eager), calculate_flops_(calculate_flops),
    calculate_activations_(calculate_activations) {}

Graph::~Graph() {
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("graph_destructor");

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
            for (Node *node : *e.first) {
                node->val().releaseMemory();
                node->grad().releaseMemory();
            }
            *e.second = 0;
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
    profiler.EndEvent();
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
    x->setNodeContainer(*this);
    if (x->getDegree() == 0) {
        Insert(x, free_nodes);
    }
    ++all_nodes_count;

    if (eager_) {
        forward();
    }
}

void Graph::forward() {
    while (true) {
        Profiler &profiler = Profiler::Ins();
        profiler.BeginEvent("dynamic_batching");
        if (Size(free_nodes) <= 0) {
            profiler.EndEvent();
            break;
        }
        auto free_nodes_begin = free_nodes.begin();
        NodeAbs *first_node = free_nodes_begin->second.front();
        Executor *cur_exec = first_node->generate();
        cur_exec->batch.clear();
        cur_exec->topo_nodes = free_nodes_begin->second;
        if (first_node->isBatched()) {
            for (NodeAbs *node : free_nodes_begin->second) {
                auto &v = node->batch();
                for (Node *atom : v) {
                    cur_exec->batch.push_back(atom);
                }
            }
        } else {
            cur_exec->batch.reserve(free_nodes_begin->second.size());
            for (NodeAbs *node : free_nodes_begin->second) {
                cur_exec->batch.push_back(dynamic_cast<Node *>(node));
            }
        }
        free_nodes.erase(free_nodes_begin->first);

        profiler.EndEvent();
        cur_exec->forwardFully();
        profiler.BeginEvent("dynamic_batching");
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
        profiler.EndEvent();
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
