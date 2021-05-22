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
/// \return The tensor of maximal values. Its size is input.size() / row;
Node *max(Node &input, int row);

/// \ingroup operator
/// The tanh activation function.
///
/// **All tanh operators will be executed in batch.**
/// \param input The input tensor.
/// \return The result tensor. Its size is equal to input.size().
Node *tanh(Node &input);

/// \ingroup operator
/// The sigmoid activation function.
///
/// **All sigmoid operators will be executed in batch.**
/// \param input The input tensor.
/// \return The result tensor. Its size is equal to input.size().
Node *sigmoid(Node &input);

/// \ingroup operator
/// The relu activation function.
///
/// **All relu operators will be executed in batch.**
/// \param input The input tensor.
/// \return The result tensor. Its size is equal to input.size().
Node *relu(Node &input);

/// \ingroup operator
/// The pointwise sqrt function.
///
/// **All sqrt operators will be executed in batch.**
/// \param input The input tensor.
/// \return The result tensor. Its size is equal to input.size().
Node *sqrt(Node &input);

/// \ingroup operator
/// Expand the input tensor in the row-wise direction.
///
/// For example, expand([0.1, 0.2], 3) will return [0.1, 0.1, 0.1, 0.2, 0.2, 0.2].
/// We consider expansion in the column-wise direction is rarely used, but if it is really your case, you can use *concat* and pass the same input tensor.
///
/// **The operators whose input tensor's sizes are equal will be executed in batch.** But this batching rule seems not reasonable enough and needs to be modified.
/// \param input The input tensor.
/// \param row The row number to expand with.
/// \return The expanded tensor. Its size is equal to input.size() * row.
Node *expand(Node &input, int row);

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
