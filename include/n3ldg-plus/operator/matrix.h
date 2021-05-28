#ifndef N3LDG_PLUS_MATRX_NODE_H
#define N3LDG_PLUS_MATRX_NODE_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

BatchedNode *tranMatrixMulMatrix(BatchedNode &a, BatchedNode &b, int input_row,
        bool use_lower_triangle_mask = false);

Node *matmul(Node &a, Node &b, int b_row, bool transpose_a = false,
        bool use_lower_triangular_mask = false);

BatchedNode *matrixMulMatrix(BatchedNode &a, BatchedNode &b, int k);

}
#endif
