#ifndef N3LDGPLUS_LOSS_H
#define N3LDGPLUS_LOSS_H

#include <Node.h>

dtype cpuCrossEntropyLoss(vector<Node *> &nodes, const vector<int> &answers, int batchsize) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        int answer = answers.at(i);
        nodes.at(i)->loss()[answer] -= 1 / nodes.at(i)->getVal()[answer] / batchsize;
        loss -= log(nodes.at(i)->getVal()[answer]);
    }
    return loss;
}

dtype crossEntropyLoss(vector<Node *> &nodes, const vector<int> &answers, int batchsize) {
    if (nodes.size() != answers.size()) {
        cerr << boost::format("crossEntropyLoss - node size is %1%, but answer size is %2%") %
            nodes.size() % answers.size() << endl;
        abort();
    }
    validateEqualNodeDims(nodes);
#if USE_GPU
    vector<dtype*> vals, losses;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    transform(nodes.begin(), nodes.end(), back_inserter(losses), gpu_get_node_loss);
    dtype loss = n3ldg_cuda::CrossEntropyLoss(vals, answers, nodes.size(), batchsize,
            losses);
#if TEST_CUDA
    dtype cpu_loss = cpuCrossEntropyLoss(nodes, answers, batchsize);
    for (Node *node : nodes) {
        n3ldg_cuda::Assert(node->loss().verify("crossEntropyLoss"));
    }
    cout << boost::format("cpu loss:%1% gpu:%2%") % cpu_loss % loss << endl;
#endif
    return loss;
#else
    return cpuCrossEntropyLoss(nodes, answers, batchsize);
#endif
}

vector<int> cpuPredict(const vector<Node *> &nodes) {
    vector<int> result;
    for (Node *node : nodes) {
        result.push_back(std::max_element(node->getVal().v,
                    node->getVal().v + node->getDim()) - node->getVal().v);
    }
    return result;
}

#if USE_GPU

vector<int> gpuPredict(const vector<Node *> &nodes) {
    vector<dtype*> vals;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    return n3ldg_cuda::Predict(vals, nodes.size(), nodes.front()->getDim());
}

#endif

vector<int> predict(const vector<Node *> &nodes) {
    validateEqualNodeDims(nodes);
#if USE_GPU
    return gpuPredict(nodes);
#else
    return cpuPredict(nodes);
#endif
}

#endif
