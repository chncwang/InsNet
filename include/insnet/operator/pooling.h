#ifndef INSNET_POOLING_H
#define INSNET_POOLING_H

#include "insnet/computation-graph/graph.h"

namespace insnet {

/// \ingroup operator
/// Find the column-wise max pooling.
///
/// For example, maxPool([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2) returns [0.5, 0.6], and maxPool([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 3) returns [0.4, 0.5, 0.6].
///
/// **The operators with the equal row number will be executed in batch.** For example, maxPool([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2) and maxPool([0.1, 0.2], 2) will be executed in batch. This guarantees that the *maxPool* operators in the same mini-batch will generally be executed in batch.
/// \param input The input tensor.
/// \param row The row number. Note that the input tensor's size should be divisible by the row number.
/// \return The result tensor. Its size is *row*.
Node *maxPool(Node &input, int row);

/// \ingroup operator
/// Find the column-wise min pooling.
///
/// It is the shortcut of *mul(\*maxPool(\*mul(input, -1), row), -1)*.
Node *minPool(Node &input, int row);

/// \ingroup operator
/// Find the column-wise sum pooling.
///
/// For example, sumPool([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2) returns [0.9, 1.2], and sumPool([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 3) returns [0.5, 0.7, 0.9].
///
/// **As similar to maxPool, the operators with the equal row number will be executed in batch.**
/// \param input The input tensor.
/// \param row The row number. Note that the input tensor's size should be divisible by the row number.
/// \return The result tensor. Its size is *row*.
Node *sumPool(Node &input, int row);

/// \ingroup operator
/// Find the column-wise sum pooling.
///
/// It is the shortcut of *mul(\*sumPool(input, row), static_cast<dtype>(row) / input.size())*.
Node *avgPool(Node &input, int row);

}

#endif
