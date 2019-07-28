#ifndef N3LDG_MAX_PROBABILITY_LOSS_H
#define N3LDG_MAX_PROBABILITY_LOSS_H

#include <vector>
#include <utility>

#include "MyLib.h"

#if USE_GPU
std::pair<dtype, std::vector<int>> MaxLogProbabilityLoss(const std::vector<Node*> &result_nodes,
        const std::vector<int> ids,
        int batchsize) {
    vector<const dtype *> vals;
    vector<dtype*> losses;
    for (const Node *node : result_nodes) {
        vals.push_back(node->getVal().value);
        losses.push_back(node->getLoss().value);
    }
    int dim = result_nodes.at(0)->getDim();
    auto result = n3ldg_cuda::SoftMaxLoss(vals, vals.size(), dim, ids, batchsize, losses);
    auto result_ids = result.second;
    for (int id : result_ids) {
        if (id >= dim) {
            cerr << boost::format("id:%1% dim:%2%") % id % dim << endl;
            abort();
        }
    }
#if TEST_CUDA
    auto cpu_result = MaxLogProbabilityLoss(result_nodes, word_ids,
            hyper_params.batch_size);
    cout << format("result loss:%1% cpu_result loss:%2%") % result.first %
        cpu_result.first << endl;
    if (abs(result.first - cpu_result.first) > 0.001) {
        abort();
    }

    for (const Node *node : result_nodes) {
        n3ldg_cuda::Assert(node->getLoss().verify("cross entropy loss"));
    }
#endif
    return result;
}
#else
std::pair<dtype, std::vector<int>> MaxLogProbabilityLoss(std::vector<Node *> &nodes,
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
        //std::cout << "sum:" << sum << std::endl;

        Tensor1D loss_tensor;
        loss_tensor.init(node.getDim());
        loss_tensor.vec() = exp.vec() / sum;
        int answer = answers.at(i);
        loss_tensor.v[answer] -= 1.0f;
        node.loss().vec() = loss_tensor.vec().unaryExpr(
                [=](dtype x)->dtype {return x * reverse_batchsize;});
        //std::cout << "max:" << max << " answer v:" << node.val.v[answer] << std::endl;
        loss += (log(sum) - node.getVal().v[answer] + max_pair.second) * reverse_batchsize;
    }

    return std::make_pair(loss, std::move(results));
}

#endif

#endif
