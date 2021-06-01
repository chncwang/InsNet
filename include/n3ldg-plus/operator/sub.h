#ifndef N3LDG_PLUS_SUB_H
#define N3LDG_PLUS_SUB_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

/// \ingroup operator
/// Subtract an input tensor by another.
///
/// **All sub operators will be executed in batch.**
/// \param inputs The input tensors to be added. Note that their sizes should be equal.
/// \return The sum of the input tensors. Its size is equal to the size of any input tensor.
Node *sub(Node &minuend, Node &subtrahend);

}

#endif
