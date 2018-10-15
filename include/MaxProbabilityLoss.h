#ifndef N3LDG_MAX_PROBABILITY_LOSS_H
#define N3LDG_MAX_PROBABILITY_LOSS_H

#include <vector>
#include <utility>

#include "MyLib.h"

std::pair<dtype, std::vector<int>> MaxLogProbabilityLoss(std::vector<Node *> &nodes,
        const std::vector<int> &answers,
        int batchsize) {
    dtype loss = 0.0f;
    dtype reverse_batchsize = 1.0 / batchsize;
    std::vector<int> results;

    for (int i = 0; i < nodes.size(); ++i) {
        Node &node = *nodes.at(i);
        dtype max = node.val.v[0];
        int max_j = 0;
        for (int j = 1; j < node.dim; ++j) {
            //std::cout << "j:" << node.val.v[j] << std::endl;
            if (node.val.v[j] > max) {
                max = node.val.v[j];
                max_j = j;
            }
        }
        results.push_back(max_j);

        Tensor1D exp;
        exp.init(node.dim);
        exp.vec() = (node.val.vec() - max).exp();
        dtype sum = static_cast<Eigen::Tensor<dtype, 0>>(exp.vec().sum())(0);
        //std::cout << "sum:" << sum << std::endl;

        Tensor1D loss_tensor;
        loss_tensor.init(node.dim);
        loss_tensor.vec() = exp.vec() / sum;
        int answer = answers.at(i);
        loss_tensor.v[answer] -= 1.0f;
        node.loss.vec() = loss_tensor.vec().unaryExpr(
                [=](dtype x)->dtype {return x * reverse_batchsize;});
        //std::cout << "max:" << max << " answer v:" << node.val.v[answer] << std::endl;
        loss += (log(sum) - node.val.v[answer] + max) * reverse_batchsize;
    }

    return std::make_pair(loss, std::move(results));
}

#endif
