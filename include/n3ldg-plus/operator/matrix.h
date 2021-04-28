#ifndef N3LDG_PLUS_MATRX_NODE_H
#define N3LDG_PLUS_MATRX_NODE_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *concatToMatrix(const std::vector<Node *> &inputs);

BatchedNode *tranMatrixMulMatrix(BatchedNode &a, BatchedNode &b, int input_row,
        bool use_lower_triangle_mask = false);

Node *matrixMulMatrix(Node &a, Node &b, int k);

BatchedNode *matrixMulMatrix(BatchedNode &a, BatchedNode &b, int k);

}
#endif
