#ifndef INSNET_SOFTMAX_NODE
#define INSNET_SOFTMAX_NODE

#include <insnet/computation-graph/graph.h>

namespace insnet {

/// \ingroup operator
/// The row-wise softmax operator.
///
/// For example, softmax([0.1, 0.1, 0.2, 0.2], 2) returns [0.5, 0.5, 0.5, 0.5]
///
/// **All the operators will be executed in batch.** This guarantees that all self-attention and cross-attention in the same layer will be executed in batch.
/// \param input The input tensor.
/// \param row The row number. Note that the input tensor's size should be divisible by the row number.
/// \return The result tensor. Its size is *input.size()*;
Node* softmax(Node &input, int row);

/// \ingroup operator
/// The row-wise softmax operator of the input vector.
///
/// This is the shorcut of *softmax(input, input.size())*.
inline Node* softmax(Node &input) {
    return softmax(input, input.size());
}

BatchedNode* softmax(BatchedNode &input, int row);

inline BatchedNode* softmax(BatchedNode &input) {
    return softmax(input, input.size());
}

}

#endif
