#ifndef N3LDG_PLUS_ADD_H
#define N3LDG_PLUS_ADD_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

/// \ingroup operator
/// Add input tensors.
///
/// **The operators that have the same number of inputs will be executed in batch.** For example, [0.1, 0.1] + [0.2, 0.2] and [0.3, 0.3, 0.3] + [0.4, 0.4, 0.4] will be executed in batch, but [0.1, 0.1] + [0.2, 0.2] and [0.1, 0.1] + [0.2, 0.2] + [0.3, 0.3] are not. If the latter is your case, use *sumPool* instead.
/// \param inputs The input tensors to be added. Note that their sizes should be equal.
/// \return The sum of the input tensors. Its size is equal to the size of any input tensor.
Node *add(const std::vector<Node*> &inputs);

}

#endif
