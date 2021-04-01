#ifndef N3LDG_PLUS_CONCAT
#define N3LDG_PLUS_CONCAT

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *concat(Graph &graph, const std::vector<Node*> &inputs, int col = 1);

Node *concat(Graph &graph, BatchedNode &inputs, int col = 1);

BatchedNode *concatInBatch(Graph &graph, const std::vector<BatchedNode *> &inputs);

}

#endif
