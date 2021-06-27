#ifndef INSNET_CONCAT
#define INSNET_CONCAT

#include "insnet/computation-graph/graph.h"

namespace insnet {

/// \ingroup operator
/// Concaternate input matrices into the result matrix with a specified column number.
///
/// For example, cat({[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]}) will return [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4], and cat({[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], 2) will return [0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.3, 0.4].
///
/// **The operators whose column number and input matrix sizes are equal one by one will be executed in batch.**
/// For example, cat({[0.1, 0.2, 0.3, 0.4], [0.1], [0.1, 0.2]}) and cat({[0, 0, 0, 0], [0], [0, 0]}) will be executed in batch.
///
/// **The operators whose column number is 1 and input tensor sizes are all the same will also be executed in batch.** This rule is especially useful when concaternating RNN hidden states.
/// For example, cat({[0.1, 0.2], [0.1, 0], [0.1, 0.2]}) and cat({[0, 0], [0, 0]}) will be executed in batch because their input tensors have the same size of 2, though they have different number of input tensors.
/// \param inputs The input matrices
/// \param col The column number of both the input matrices and the result matrix. *The default value is 1.*
/// \return The result matrix. Its size is equal to the sum of all input matrix sizes.
Node *cat(const std::vector<Node*> &inputs, int col = 1);

Node *cat(BatchedNode &inputs, int col = 1);

}

#endif
