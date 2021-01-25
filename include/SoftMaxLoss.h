#ifndef _SOFTMAXLOSS_H_
#define _SOFTMAXLOSS_H_

#include "MyLib.h"
#include "Metric.h"
#include "Node.h"

dtype softMaxLoss(AtomicNode * x, const vector<dtype> &answer, Metric& eval,
    dtype batchsize) {
    int nDim = x->getDim();
    int labelsize = answer.size();
    if (labelsize != nDim) {
        std::cerr << "softmax_loss error: dim size invalid" << std::endl;
        return -1.0;
    }

    NRVec<dtype> scores(nDim);

    dtype cost = 0.0;
    int optLabel = -1;
    for (int i = 0; i < nDim; ++i) {
        if (answer[i] >= 0) {
            if (optLabel < 0 || x->val()[i] > x->val()[optLabel])
                optLabel = i;
        }
    }

    dtype sum1 = 0, sum2 = 0, maxScore = x->val()[optLabel];
    for (int i = 0; i < nDim; ++i) {
        scores[i] = -1e10;
        if (answer[i] >= 0) {
            scores[i] = exp(x->val()[i] - maxScore);
            if (answer[i] == 1)
                sum1 += scores[i];
            sum2 += scores[i];
        }
    }
    cost += (log(sum2) - log(sum1)) / batchsize;
    if (answer[optLabel] == 1)
        eval.correct_label_count++;
    eval.overall_label_count++;

    for (int i = 0; i < nDim; ++i) {
        if (answer[i] >= 0) {
            x->loss()[i] = (scores[i] / sum2 - answer[i]) / batchsize;
        }
    }

    return cost;

}

dtype softMaxLoss(AtomicNode &node, int answer, Metric &metric, int batchsize) {
    std::vector<dtype> fit_answer;
    for (int i = 0; i < node.getDim(); ++i) {
        fit_answer.push_back(i == answer);
    }
    return softMaxLoss(&node, fit_answer, metric, batchsize);
}

#if USE_GPU
dtype softMaxLoss(const std::vector<AtomicNode *> &x, const std::vector<int> &answers,
        n3ldg_cuda::DeviceInt &correct,
        int batchsize = 1) {
    cerr << "unsupported operation" << endl;
    abort();
}
#endif

dtype cost(AtomicNode * x, const vector<dtype> &answer, int batchsize = 1) {
    int nDim = x->getDim();
    int labelsize = answer.size();
    if (labelsize != nDim) {
        std::cerr << "softmax_loss error: dim size invalid" << std::endl;
        return -1.0;
    }

    NRVec<dtype> scores(nDim);

    dtype cost = 0.0;

    int optLabel = -1;
    for (int i = 0; i < nDim; ++i) {
        if (answer[i] >= 0) {
            if (optLabel < 0 || x->val()[i] > x->val()[optLabel])
                optLabel = i;
        }
    }

    dtype sum1 = 0, sum2 = 0, maxScore = x->val()[optLabel];
    for (int i = 0; i < nDim; ++i) {
        scores[i] = -1e10;
        if (answer[i] >= 0) {
            scores[i] = exp(x->val()[i] - maxScore);
            if (answer[i] == 1)
                sum1 += scores[i];
            sum2 += scores[i];
        }
    }
    cost += (log(sum2) - log(sum1)) / batchsize;
    return cost;
}


#endif /* _SOFTMAXLOSS_H_ */
