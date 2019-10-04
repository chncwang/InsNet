#ifndef N3LDG_MAX_PROBABILITY_LOSS_H
#define N3LDG_MAX_PROBABILITY_LOSS_H

#include <vector>
#include <utility>
#include "Node.h"

#include "MyLib.h"

std::pair<dtype, std::vector<int>> cpuMaxLogProbabilityLoss(std::vector<Node *> &nodes,
        const std::vector<int> &answers,
        int batchsize) {
    dtype loss = 0.0f;
    dtype reverse_batchsize = 1.0 / batchsize;
    std::vector<int> results;

    for (int i = 0; i < nodes.size(); ++i) {
        Node &node = *nodes.at(i);
        auto tuple = toExp(node);
        std::pair<int, dtype> &max_pair = std::get<1>(tuple);
        results.push_back(max_pair.first);
        dtype sum = std::get<2>(tuple);
        Tensor1D &exp = *std::get<0>(tuple);

        Tensor1D loss_tensor;
        loss_tensor.init(node.getDim());
        loss_tensor.vec() = exp.vec() / sum;
        int answer = answers.at(i);
        loss_tensor.v[answer] -= 1.0f;
        node.loss().vec() = loss_tensor.vec().unaryExpr(
                [=](dtype x)->dtype {return x * reverse_batchsize;});
        loss += (log(sum) - node.getVal().v[answer] + max_pair.second) * reverse_batchsize;
    }

    return std::make_pair(loss, std::move(results));
}

std::pair<dtype, std::vector<int>> maxLogProbabilityLoss(std::vector<Node *> &nodes,
        const std::vector<int> &answers,
        int batchsize) {
#if USE_GPU
    vector<const dtype*> vals;
    vector<dtype*> losses;
    for (Node *node : nodes) {
        vals.push_back(node->getVal().value);
        losses.push_back(node->getLoss().value);
    }
#if TEST_CUDA
    cpuMaxLogProbabilityLoss(nodes, answers, batchsize);
#endif
    return n3ldg_cuda::SoftMaxLoss(vals, nodes.size(),
            nodes.front()->getDim(), answers, batchsize, losses);
#else
    return cpuMaxLogProbabilityLoss(nodes, answers, batchsize);
#endif
}

#endif
