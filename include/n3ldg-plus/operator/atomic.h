#ifndef N3LDG_PLUS_ATOMIC_H
#define N3LDG_PLUS_ATOMIC_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

/// \ingroup operator
/// Find the row-wise max scalars of the input tensor.
///
/// **The max operators with the same returned tensor's size will be executed in batch.** But this batching rule seems not reasonable enough and needs to be modified.
/// \param input The input tensor.
/// \param row The row number for which the row-wise max should be calculated. Note that the input tensor's size should be divisible by the row number.
/// \return The tensor of maximal values. Its size is *input.size() / row*;
Node *max(Node &input, int row);

/// \ingroup operator
/// The tanh activation function.
///
/// **All tanh operators will be executed in batch.**
/// \param input The input tensor.
/// \return The result tensor. Its size is equal to *input.size()*.
Node *tanh(Node &input);

/// \ingroup operator
/// The sigmoid activation function.
///
/// **All sigmoid operators will be executed in batch.**
/// \param input The input tensor.
/// \return The result tensor. Its size is equal to *input.size()*.
Node *sigmoid(Node &input);

/// \ingroup operator
/// The relu activation function.
///
/// **All relu operators will be executed in batch.**
/// \param input The input tensor.
/// \return The result tensor. Its size is equal to *input.size()*.
Node *relu(Node &input);

/// \ingroup operator
/// The pointwise sqrt function.
///
/// **All sqrt operators will be executed in batch.**
/// \param input The input tensor.
/// \return The result tensor. Its size is equal to *input.size()*.
Node *sqrt(Node &input);

/// \ingroup operator
/// Expand the input tensor in the row-wise direction.
///
/// For example, expandRowwisely([0.1, 0.2], 3) will return [0.1, 0.1, 0.1, 0.2, 0.2, 0.2].
///
/// **The operators whose input tensor's sizes are equal will be executed in batch.** But this batching rule seems not reasonable enough and needs to be modified.
/// \param input The input tensor.
/// \param row The row number to expand with.
/// \return The expanded tensor. Its size is equal to *input.size() * row*.
Node *expandRowwisely(Node &input, int row);

/// \ingroup operator
/// Sum up the input tensor's elements in the row-wise direction.
///
/// For example, sum([0.1, 0.1, 0.1, 0.2, 0.2, 0.2], 3) will return [0.3, 0.6].
/// 
/// If you want to sum up in the column-wise direction, use *sumPool* instead.
///
/// **The operators that returns the equal size of tensors will be executed in batch.** But this batching rule seems not reasonable enough and needs to be modified.
/// \param input The input tensor.
/// \param input_row The input tensor's row number.
/// \return The result tensor. Its size is equal to *input.size() / input_row*.
Node *sum(Node &input,  int input_row);

/// \ingroup operator
/// The pointwise exp function.
///
/// **All exp operators will be executed in batch.**
/// \param input The input tensor.
/// \return The result tensor. Its size is equal to *input.size()*.
Node *exp(Node &input);

/// \ingroup operator
/// The dropout function. In particular, if the dropout value is no greater than 1e-10, it will return the input tensor directly.
///
/// If the graph is set to the training stage, it drop out all elements independently with the probability *p*.
/// Otherwise it scales all elements by (1 - p).
///
/// **The operators with the equal dropout probability will be executed in batch.**
/// For example, dropout([0.1, 0.1], 0.1) and dropout([0.2, 0.2, 0.2], 0.1) will be executed in batch, but dropout([0.1, 0.1], 0.1) and dropout([0.2, 0.2], 0.2) will not.
/// \param input The input tensor.
/// \param p The dropout probability.
/// \return The result tensor. Its size is equal to *input.size()*.
Node *dropout(Node &input, dtype p);

/// \ingroup operator
/// It multiplies the input tensor by the factor.
//
/// For example, mul([0.1, 0.1], 2) will return [0.2, 0.2].
///
/// **All mul operators will be executed in batch.**
/// \param input The input tensor.
/// \param factor The number to multiply with.
/// \return The multiplied tensor. Its size is equal to *input.size()*.
Node *mul(Node &input, dtype factor);

BatchedNode *mul(BatchedNode &input, dtype factor);


}

#endif
