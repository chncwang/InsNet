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
/// \param nodes The input matrices. their sizes can be variant but should all be divisible by row.
/// \param row The row number of nodes.
/// \return The result indexes.
std::vector<std::vector<int>> argmax(const std::vector<Node *> &nodes, int row);

/// \ingroup loss
/// The negative log likelihood loss.
///
/// It returns the loss and add gradients to probs.
///
/// **It will be executed eagerly.**
/// \param probs The probability matrices. their sizes can be variant but should all be divisible by row. **Note that we may change this argument to log probabilities in the future to guarantee numerical stability.**
/// \param row The row number of probability matrices.
/// \param answers The answers. The inner vector's sizes should be equal to probs' size one by one.
/// \param factor The factor that the loss will be multiplied with. Specifically, pass 1.0 if you want sum reduction, or 1.0 / n if you want average reduction, where n is the sum of answer sizes.
/// \return The loss.
dtype NLLLoss(std::vector<Node *> &probs, int row, const std::vector<std::vector<int>> &answers,
        dtype factor);

dtype KLDivLoss(std::vector<Node *> &nodes,
        const std::vector<std::shared_ptr<std::vector<dtype>>> &answers,
        dtype factor);

float binrayLikelihoodLoss(std::vector<Node *> &nodes,
        const std::vector<std::vector<int>> &answers,
        dtype factor);
}

#endif
