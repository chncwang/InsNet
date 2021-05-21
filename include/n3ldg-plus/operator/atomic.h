#ifndef N3LDG_PLUS_ATOMIC_H
#define N3LDG_PLUS_ATOMIC_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

/// \ingroup operator
/// Find the row-wise max scalars of the input tensor.
///
/// **The operators that returns the equal size of tensors will be executed in batch.** But this batching rule seems not reasonable enough and needs to be modified.
/// \param input The input tensor.
/// \param row The row number for which the row-wise max should be calculated. Note that the input tensor's size should be divisible by the row number.
Node *max(Node &input, int row);

Node *tanh(Node &input);

Node *sigmoid(Node &input);

Node *relu(Node &input);

Node *sqrt(Node &input);

Node *scalarToVector(Node &input, int row);

BatchedNode *scalarToVector(BatchedNode &input, int row);

BatchedNode *scalarToVector(BatchedNode &input, const ::std::vector<int> &rows);

Node *vectorSum(Node &input,  int input_col);

BatchedNode *vectorSum(BatchedNode &input,  int input_col);

Node *exp(Node &input);

BatchedNode *exp(BatchedNode &input);

Node *dropout(Node &input, dtype dropout);

BatchedNode *dropout(BatchedNode &input, dtype dropout);

Node *scaled(Node &input, dtype factor);

BatchedNode *scaled(BatchedNode &input, const ::std::vector<dtype> &factors);

BatchedNode *scaled(BatchedNode &input, dtype factor);

}

#endif
