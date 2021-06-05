#ifndef N3LDGPLUS_LOSS_H
#define N3LDGPLUS_LOSS_H

#include "insnet/computation-graph/node.h"

namespace insnet {

std::vector<std::vector<int>> predict(const std::vector<Node *> &nodes, int row);

dtype NLLoss(std::vector<Node *> &nodes, int row,
        const std::vector<std::vector<int>> &answers,
        dtype factor);

std::pair<float, std::vector<int>> KLDivergenceLoss(std::vector<Node *> &nodes,
        const std::vector<std::shared_ptr<std::vector<dtype>>> &answers,
        dtype factor);

float binrayLikelihoodLoss(std::vector<Node *> &nodes,
        const std::vector<std::vector<int>> &answers,
        dtype factor);
}

#endif