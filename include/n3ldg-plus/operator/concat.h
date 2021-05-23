#ifndef N3LDG_PLUS_CONCAT
#define N3LDG_PLUS_CONCAT

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

/// \ingroup operator
/// Concaternate input matrices into the result matrix with a specified column number.
///
/// For example, cat({[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]}) will return [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4], and cat({[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], 2) will return [0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.3, 0.4].
///
/// **The operators whose column number and input matrix sizes are equal one by one will be executed in batch.**
/// For example, cat({[0.1, 0.2, 0.3, 0.4], [0.1], [0.1, 0.2]}) and cat({[0, 0, 0, 0], [0], [0, 0]}) will be executed in batch.
/// \param inputs The input matrices
/// \param col The column number of both the input matrices and the result matrix.
/// \return The result matrix. Its size is equal to the sum of all input matrix sizes.
Node *cat(const std::vector<Node*> &inputs, int col = 1);

Node *cat(BatchedNode &inputs, int col = 1);

}

#endif
