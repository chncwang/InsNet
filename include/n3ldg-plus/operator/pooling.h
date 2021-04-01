#ifndef N3LDG_PLUS_POOLING_H
#define N3LDG_PLUS_POOLING_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {


Node *maxPool(Graph &graph, std::vector<Node *> &inputs);

Node *minPool(Graph &graph, std::vector<Node *> &inputs);

Node *sumPool(Graph &graph, std::vector<Node *> &inputs);

Node *averagePool(Graph &graph, std::vector<Node *> &inputs);

}

#endif
