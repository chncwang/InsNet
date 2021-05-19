#ifndef N3LDG_PLUS_GRAPH_H
#define N3LDG_PLUS_GRAPH_H

#include <unordered_map>
#include "n3ldg-plus/computation-graph/node.h"

namespace n3ldg_plus {

typedef std::unordered_map<std::string, std::vector<NodeAbs *>> NodeMap;

/// \brief The computation graph.
class Graph : public NodeContainer {
public:
    Graph(ModelStage stage = ModelStage::TRAINING, bool eager = false,
            bool calculate_flops = false,
            bool calculate_activations = false);

    Graph (const Graph &graph) = delete;

    virtual ~Graph();

    void forward();

    void backward();

    void addNode(NodeAbs *x) override;

    const std::map<std::string, int64_t> &getFLOPs() const {
        return flops_table_;
    }

    int64_t getActivations() const {
        return activations_;
    }

    void addFLOPs(int64_t flops, const std::string &name);

protected:
    std::vector<Executor *> execs;
    NodeMap free_nodes;
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
