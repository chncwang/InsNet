#ifndef BasicGraph
#define BasicGraph

#include "eigen/Eigen/Dense"
#include "n3ldg-plus/computation-graph/node.h"
#include "n3ldg-plus/util/util.h"
#include "n3ldg-plus/util/profiler.h"
#include <set>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace n3ldg_plus {

int GetDegree(std::map<void*, int> &degree_map, NodeAbs *p) {
    auto it = degree_map.find(p);
    if (it == degree_map.end()) {
        degree_map.insert(std::pair<void*, int>(p, p->getDegree()));
        return p->getDegree();
    } else {
        return it->second;
    }
}

void DecreaseDegree(std::map<void*, int> &degree_map, NodeAbs *p) {
    auto it = degree_map.find(p);
    if (it == degree_map.end()) {
        degree_map.insert(std::pair<void*, int>(p, p->getDegree() - 1));
    } else {
        --(it->second);
    }
}

typedef std::unordered_map<std::string, std::vector<NodeAbs *>> NodeMap;

void Insert(NodeAbs *node, NodeMap& node_map) {
    std::string x_hash = node->cachedTypeSig();
    auto it = node_map.find(x_hash);
    if (it == node_map.end()) {
        std::vector<NodeAbs *> v = {node};
        node_map.insert(std::make_pair<std::string, std::vector<NodeAbs *>>(std::move(x_hash),
                    std::move(v)));
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
    Graph(bool eager = false, bool calculate_flops = false, bool calculate_activations = false) :
        eager_(eager), calculate_flops_(calculate_flops),
        calculate_activations_(calculate_activations) {}

    Graph (const Graph &graph) = delete;

    virtual ~Graph() {
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

    void backward() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent("computation backward");
        int count = execs.size();
        for (int idx = count - 1; idx >= 0; idx--) {
            execs.at(idx)->backwardFully();
        }
        profiler.EndCudaEvent();
    }

    void addNode(NodeAbs *x) override {
        if (x == nullptr) {
            std::cerr << "x is nullptr" << std::endl;
            abort();
        }
        if (x->getDegree() == 0) {
            Insert(x, free_nodes);
        }
        ++all_nodes_count;

        std::string x_type_hash = x->cachedTypeSig();
        auto it = node_type_depth.find(x_type_hash);
        if (it == node_type_depth.end()) {
            node_type_depth.insert(std::pair<std::string, std::pair<int, int>>(
                        x_type_hash, std::pair<int, int>(x->getDepth(), 1)));
        } else {
            it->second.first += x->getDepth();
            it->second.second++;
        }

        if (eager_) {
            forward();
        }
    }

    void forward() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();

        while (true) {
            profiler.BeginEvent("computation plan");
            if (Size(free_nodes) <= 0) {
                profiler.EndEvent();
                break;
            }
            float min_avg_depth = 100000000;
            std::vector<NodeAbs *> shallow_nodes;
            std::string min_hash;
            for (auto it : free_nodes) {
                std::string type_hash = it.first;
                auto depth_it = node_type_depth.find(type_hash);
                if (depth_it == node_type_depth.end()) {
                    std::cerr << fmt::format("type not found in depth map:{}\n", type_hash);
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
            profiler.EndEvent();
#if USE_GPU
            profiler.BeginEvent("clear nodes");
            clearNodes(cur_exec->batch);
            profiler.EndCudaEvent();
#endif
//            cout << "type:" << cur_exec->getSignature() << " " << cur_exec->batch.size() << endl << endl;

            profiler.BeginEvent("computation forward");
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
            profiler.EndEvent();

            profiler.BeginEvent("computation plan");
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
                std::cerr << fmt::format("Graph compute - it first:{} second:{}", it.first,
                        it.second);
                abort();
            }

            profiler.EndEvent();
        }

        if (finish_nodes.size() != all_nodes_count) {
            std::cerr << "error: several nodes are not executed, finished: " <<
                finish_nodes.size() << ", all: " << all_nodes_count << std::endl;
            abort();
        }
    }

    const std::map<std::string, int64_t> &getFLOPs() const {
        return flops_table_;
    }

    int64_t getActivations() const {
        return activations_;
    }

    void addFLOPs(int64_t flops, const std::string &name) {
        if (calculate_flops_) {
            const auto &it = flops_table_.find(name);
            if (it == flops_table_.end()) {
                flops_table_.insert(make_pair(name, flops));
            } else {
                it->second += flops;
            }
        }
    }

protected:
    std::vector<Executor *> execs;
    NodeMap free_nodes;
    std::map<std::string, std::pair<int, int>> node_type_depth;
    std::vector<NodeAbs *> finish_nodes;

private:
    bool eager_ = false;
    bool calculate_flops_ = false;
    std::map<std::string, int64_t> flops_table_;

    bool calculate_activations_ = false;
    int64_t activations_ = 0;
    int all_nodes_count = 0;
};

}

#endif
