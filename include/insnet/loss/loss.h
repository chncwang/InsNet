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
/// It returns the loss and accumulate gradients to probs.
///
/// **It will be executed eagerly.**
/// \param log_probs The natural log probability matrices. their sizes can be variant but should all be divisible by row.
/// \param row The row number of probability matrices.
/// \param answers The answers. The inner vector's sizes should be equal to probs' size one by one.
/// \param factor The factor that the loss will be multiplied with. Specifically, pass 1.0 if you want sum reduction, or 1.0 / n if you want average reduction, where n is the sum of answer sizes.
/// \return The loss.
dtype NLLLoss(std::vector<Node *> &log_probs, int row, const std::vector<std::vector<int>> &answers,
        dtype factor);

/// \ingroup loss
/// The KL divergence loss.
///
/// It returns the loss and accumulate gradients to probs.
///
/// **It will be executed eagerly.**
/// \param probs The probability vectors. **Note that for the current version they are all vectors of the same row and we may change it to support matrices of variant sizes in the future.**
/// \param answers The answers.
/// \param factor The factor that the loss will be multiplied with.
/// \return The loss.
dtype KLDivLoss(std::vector<Node *> &probs, const std::vector<std::vector<dtype> *> &answers,
        dtype factor);

/// \ingroup loss
/// The binary cross entropy loss.
///
/// It returns the loss and accumulate gradients to probs.
///
/// **It will be executed eagerly.**
/// \param probs The probability vectors. **Note that for the current version they are all vectors of the same row and we may change it to support matrices of variant sizes in the future.**
/// \param answers The answers. Their values should be either 0 or 1.
/// \param factor The factor that the loss will be multiplied with.
/// \return The loss.
float BCELoss(std::vector<Node *> &probs, const std::vector<std::vector<int> *> &answers,
        dtype factor);
}

#endif
