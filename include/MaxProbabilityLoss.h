#ifndef N3LDG_MAX_PROBABILITY_LOSS_H
#define N3LDG_MAX_PROBABILITY_LOSS_H

#include <vector>
#include "MyLib.h"

dtype MaxLogProbabilityLoss(std::vector<Node *> &nodes, const std::vector<int> &answers,
        int batchsize) {
    dtype loss = 0.0f;
    dtype reverse_batchsize = 1.0 / batchsize;

    for (int i = 0; i < nodes.size(); ++i) {
        Node &node = *nodes.at(i);
        dtype max = node.val.v[0];
        for (int j = 1; j < node.dim; ++j) {
            if (node.val.v[j] > max) {
                max = node.val.v[j];
            }
        }
        Tensor1D exp;
        exp.init(node.dim);
        exp.vec() = (node.val.vec() - max).exp();
        dtype sum = static_cast<Eigen::Tensor<dtype, 0>>(exp.vec().sum())(0);
        node.loss.vec() += exp.vec() / sum;
        int answer = answers.at(i);
        node.loss.v[answer] -= 1.0f;
        node.loss.vec().unaryExpr([=](dtype x)->dtype {return x * reverse_batchsize;});
        loss += (log(sum) - node.val.v[answer] + max) * reverse_batchsize;
    }

    return loss;
}

#endif
