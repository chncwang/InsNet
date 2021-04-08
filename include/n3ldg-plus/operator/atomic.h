#ifndef N3LDG_PLUS_ATOMIC_H
#define N3LDG_PLUS_ATOMIC_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *maxScalar(Graph &graph, Node &input, int input_col);

BatchedNode *maxScalar(Graph &graph, BatchedNode &input, int input_col);

Node *tanh(Graph &graph, Node &input);

Node *sigmoid(Graph &graph, Node &input);

Node *relu(Graph &graph, Node &input);

BatchedNode *relu(Graph &graph, BatchedNode &input);

Node *sqrt(Graph &graph, Node &input);

BatchedNode *sqrt(Graph &graph, BatchedNode &input);

Node *scalarToVector(Graph &graph, Node &input, int row);

BatchedNode *scalarToVector(Graph &graph, BatchedNode &input, int row);

BatchedNode *scalarToVector(Graph &graph, BatchedNode &input, const ::std::vector<int> &rows);

Node *vectorSum(Graph &graph, Node &input,  int input_col);

BatchedNode *vectorSum(Graph &graph, BatchedNode &input,  int input_col);

Node *exp(Graph &graph, Node &input);

BatchedNode *exp(Graph &graph, BatchedNode &input);

Node *dropout(Graph &graph, Node &input, dtype dropout, bool is_training);

BatchedNode *dropout(Graph &graph, BatchedNode &input, dtype dropout, bool is_training);

Node *scaled(Graph &graph, Node &input, dtype factor);

BatchedNode *scaled(Graph &graph, BatchedNode &input, const ::std::vector<dtype> &factors);

BatchedNode *scaled(Graph &graph, BatchedNode &input, dtype factor);

}

#endif
