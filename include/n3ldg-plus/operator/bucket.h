#ifndef N3LDG_PLUS_BUCKET_H
#define N3LDG_PLUS_BUCKET_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

/// \ingroup operator
/// Initialize a tensor with the computation graph, specified size and value.
///
/// For example, tensor(graph, 2, 0) will return [0, 0].
///
/// **The operators passed with the equal size will be executed in batch.**
/// \param graph The computation graph.
/// \param size The result tensor's size.
/// \param value The value to initialize the tensor.
/// \return The result tensor. Its size is equal to *size*.
Node *tensor(Graph &graph, int size, dtype value);

/// \ingroup operator
/// Initialize a tensor with the computation graph and list.
///
/// For example, tensor(graph, {0.1, 0.2}) will return [0.1, 0.2].
///
/// **The operators passed with the equal size of list will be executed in batch.**
/// \param graph The computation graph.
/// \param list The list to initialize the tensor.
/// \return The result tensor. Its size is equal to *list.size()*.
Node *tensor(Graph &graph, const std::vector<dtype> &list);

}

#endif
