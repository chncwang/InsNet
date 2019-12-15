#ifndef N3LDGPLUS_LOSS_H
#define N3LDGPLUS_LOSS_H

#include <Node.h>

dtype cpuCrossEntropyLoss(vector<Node *> &nodes, const vector<int> &answers, dtype factor) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        int answer = answers.at(i);
        nodes.at(i)->loss()[answer] -= 1 / nodes.at(i)->getVal()[answer] * factor;
        loss -= log(nodes.at(i)->getVal()[answer]);
    }
    return loss * factor;
}

dtype crossEntropyLoss(vector<Node *> &nodes, const vector<int> &answers, dtype factor) {
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
    dtype loss = n3ldg_cuda::CrossEntropyLoss(vals, answers, nodes.size(), factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuCrossEntropyLoss(nodes, answers, factor);
    for (Node *node : nodes) {
        n3ldg_cuda::Assert(node->loss().verify("crossEntropyLoss"));
    }
    cout << boost::format("cpu loss:%1% gpu:%2%") % cpu_loss % loss << endl;
#endif
    return loss;
#else
    return cpuCrossEntropyLoss(nodes, answers, factor);
#endif
}

float cpuMultiCrossEntropyLoss(vector<Node *> &nodes, const vector<vector<int>> &answers,
        dtype factor) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        Node &node = *nodes.at(i);
        const auto &answer = answers.at(i);
        for (int j = 0; j < node.getDim(); ++j) {
            dtype val = node.getVal()[j];
            node.loss()[j] += (answer.at(j) ?  -1 / val : 1 / (1 - val)) * factor;
            loss += (answer.at(j) ? -log(val): -log(1 - val));
        }
    }
    return loss * factor;
}

float multiCrossEntropyLoss(vector<Node *> &nodes, const vector<vector<int>> &answers,
        dtype factor) {
    if (nodes.size() != answers.size()) {
        cerr << "multiCrossEntropyLoss - nodes size is not equal to answers size" << endl;
        abort();
    }
    validateEqualNodeDims(nodes);
#if USE_GPU
    vector<dtype *> vals, losses;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    transform(nodes.begin(), nodes.end(), back_inserter(losses), gpu_get_node_loss);
    dtype gpu_loss = n3ldg_cuda::MultiCrossEntropyLoss(vals, answers, nodes.size(),
            nodes.front()->getDim(), factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuMultiCrossEntropyLoss(nodes, answers, factor);
    cout << "multiCrossEntropyLoss - gpu loss:" << gpu_loss << " cpu_loss:" << cpu_loss << endl;
    for (Node *node : nodes) {
        n3ldg_cuda::Assert(node->getLoss().verify("multiCrossEntropyLoss"));
    }
#endif
    return gpu_loss;
#else
    return cpuMultiCrossEntropyLoss(nodes, answers, factor);
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
