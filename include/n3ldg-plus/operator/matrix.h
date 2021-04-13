#ifndef N3LDG_PLUS_MATRX_NODE_H
#define N3LDG_PLUS_MATRX_NODE_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *concatToMatrix(const ::std::vector<Node *> &inputs);

Node *concatToMatrix(BatchedNode &input);

BatchedNode *concatToMatrix(BatchedNode &input, int group);

Node *matrixAndVectorMulti(Node &matrix, Node &vec);

BatchedNode *matrixAndVectorMulti(Node &matrix, BatchedNode &vec, int *dim = nullptr);

BatchedNode *matrixAndVectorMulti(BatchedNode &matrix, BatchedNode &vec, int *dim = nullptr);

Node *tranMatrixMulVector(Node &matrix, Node &vec, int dim);

Node *tranMatrixMulVector(Node &matrix, Node &vec);

BatchedNode *tranMatrixMulVector(Node &matrix, BatchedNode &vec,
        const ::std::vector<int> *dims = nullptr);

BatchedNode *tranMatrixMulVector(BatchedNode &matrix, BatchedNode &vec,
        const ::std::vector<int> *dims = nullptr);

BatchedNode *tranMatrixMulMatrix(BatchedNode &a, BatchedNode &b, int input_row,
        bool use_lower_triangle_mask = false);

BatchedNode *matrixMulMatrix(BatchedNode &a, BatchedNode &b, int k);

}
#endif
