#ifndef N3LDG_PLUS_ATOMIC_H
#define N3LDG_PLUS_ATOMIC_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *maxScalar(Node &input, int input_col);

BatchedNode *maxScalar(BatchedNode &input, int input_col);

Node *tanh(Node &input);

Node *sigmoid(Node &input);

Node *relu(Node &input);

BatchedNode *relu(BatchedNode &input);

Node *sqrt(Node &input);

BatchedNode *sqrt(BatchedNode &input);

Node *scalarToVector(Node &input, int row);

BatchedNode *scalarToVector(BatchedNode &input, int row);

BatchedNode *scalarToVector(BatchedNode &input, const ::std::vector<int> &rows);

Node *vectorSum(Node &input,  int input_col);

BatchedNode *vectorSum(BatchedNode &input,  int input_col);

Node *exp(Node &input);

BatchedNode *exp(BatchedNode &input);

Node *dropout(Node &input, dtype dropout, bool is_training);

BatchedNode *dropout(BatchedNode &input, dtype dropout, bool is_training);

Node *scaled(Node &input, dtype factor);

BatchedNode *scaled(BatchedNode &input, const ::std::vector<dtype> &factors);

BatchedNode *scaled(BatchedNode &input, dtype factor);

}

#endif
