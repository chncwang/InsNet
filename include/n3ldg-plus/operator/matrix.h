#ifndef N3LDG_PLUS_MATRX_NODE_H
#define N3LDG_PLUS_MATRX_NODE_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *concatToMatrix(Graph &graph, const std::vector<Node *> &inputs);

Node *concatToMatrix(Graph &graph, BatchedNode &input);

BatchedNode *concatToMatrix(Graph &graph, BatchedNode &input, int group);

Node *matrixAndVectorMulti(Graph &graph, Node &matrix, Node &vec);

BatchedNode *matrixAndVectorMulti(Graph &graph, Node &matrix, BatchedNode &vec,
        int *dim = nullptr);

BatchedNode *matrixAndVectorMulti(Graph &graph, BatchedNode &matrix, BatchedNode &vec,
        int *dim = nullptr);

Node *tranMatrixMulVector(Graph &graph, Node &matrix, Node &vec, int dim);

Node *tranMatrixMulVector(Graph &graph, Node &matrix, Node &vec);

BatchedNode *tranMatrixMulVector(Graph &graph, Node &matrix, BatchedNode &vec,
        const std::vector<int> *dims = nullptr);

BatchedNode *tranMatrixMulVector(Graph &graph, BatchedNode &matrix, BatchedNode &vec,
        const std::vector<int> *dims = nullptr);

BatchedNode *tranMatrixMulMatrix(Graph &graph, BatchedNode &a, BatchedNode &b, int input_row,
        bool use_lower_triangle_mask = false);

BatchedNode *matrixMulMatrix(Graph &graph, BatchedNode &a, BatchedNode &b, int k);

}
#endif
