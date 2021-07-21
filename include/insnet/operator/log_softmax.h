#ifndef INSNET_LOG_SOFTMAX_NODE
#define INSNET_LOG_SOFTMAX_NODE

#include <insnet/computation-graph/graph.h>

namespace insnet {

/// \ingroup operator
/// The row-wise log softmax operator.
///
/// For example, logSoftmax([5, 0, 0, -5], 2) returns [-0.0067, -5.0067, -0.0067, -5.0067]
///
/// **The operators with the same row will be executed in batch.**
/// \param input The input tensor.
/// \param row The row number. Note that the input tensor's size should be divisible by the row number.
/// \return The result tensor. Its size is *input.size()*;
Node* logSoftmax(Node &input, int row);

/// \ingroup operator
/// The row-wise log softmax operator of the input vector.
///
/// This is the shorcut of *logSoftmax(input, input.size())*.
inline Node* logSoftmax(Node &input) {
    return logSoftmax(input, input.size());
}

Node *log(Node &input);

}

#endif
