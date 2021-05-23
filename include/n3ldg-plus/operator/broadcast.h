#ifndef N3LDG_PLUS_BROADCAST_H
#define N3LDG_PLUS_BROADCAST_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

/// \ingroup operator
/// Expand the input tensor in the column-wise direction.
///
/// For example, expandColumnwisely([0.1, 0.2], 3) will return [0.1, 0.2, 0.1, 0.2, 0.1, 0.2].
///
/// **The operators whose input tensor's sizes are equal will be executed in batch.**
/// \param input The input tensor.
/// \param col The column number to expand with.
/// \return The expanded tensor. Its size is equal to input.size() * col.
Node *expandColumnwisely(Node &input, int col);

}

#endif
