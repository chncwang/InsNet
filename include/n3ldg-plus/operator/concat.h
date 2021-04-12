#ifndef N3LDG_PLUS_CONCAT
#define N3LDG_PLUS_CONCAT

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *concat(const std::vector<Node*> &inputs, int col = 1);

Node *concat(BatchedNode &inputs, int col = 1);

BatchedNode *concatInBatch(const std::vector<BatchedNode *> &inputs);

}

#endif
