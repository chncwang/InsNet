#ifndef N3LDG_PLUS_SPLIT_H
#define N3LDG_PLUS_SPLIT_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node* split(Node &input, int dim, int offset, int col = 1);

BatchedNode* split(BatchedNode &input, int dim, int offset);

BatchedNode *split(BatchedNode &input, int dim, const std::vector<int> &offsets);

BatchedNode *split(Node &input, int row, const std::vector<int> &offsets, int col = 1);

}

#endif
