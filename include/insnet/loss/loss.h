#ifndef N3LDGPLUS_LOSS_H
#define N3LDGPLUS_LOSS_H

#include "insnet/computation-graph/node.h"

namespace insnet {

/// \ingroup operator
/// Return indexes of row-wise max values.
///
/// For example, argmax({[0.1, 0.2], [0.1, 0.2, 0.3, 0.4]}, 2) returns {{1}, {1, 1}}.
///
/// **It is not differentiable and will be executed eagerly.**
/// \param nodes The input matrices. their sizes can be different but should be divisible by row.
/// \param row The row number of nodes.
/// \return The result indexes.
std::vector<std::vector<int>> argmax(const std::vector<Node *> &nodes, int row);

dtype NLLoss(std::vector<Node *> &nodes, int row,
        const std::vector<std::vector<int>> &answers,
        dtype factor);

dtype KLDivergenceLoss(std::vector<Node *> &nodes,
        const std::vector<std::shared_ptr<std::vector<dtype>>> &answers,
        dtype factor);

float binrayLikelihoodLoss(std::vector<Node *> &nodes,
        const std::vector<std::vector<int>> &answers,
        dtype factor);
}

#endif
