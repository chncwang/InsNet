#ifndef N3LDGPLUS_LOSS_H
#define N3LDGPLUS_LOSS_H

#include <Node.h>

void cpuCrossEntropyLoss(vector<Node *> &nodes, const vector<int> &answers) {
    for (int i = 0; i < nodes.size(); ++i) {
        int answer = answers.at(i);
        nodes.at(i)->loss()[answer] = - 1 / nodes.at(i)->getVal()[answer];
    }
}

void crossEntropyLoss(vector<Node *> &nodes, const vector<int> &answers) {
    if (nodes.size() != answers.size()) {
        cerr << boost::format("crossEntropyLoss - node size is %1%, but answer size is %2%") %
            nodes.size() % answers.size() << endl;
        abort();
    }
    validateEqualNodeDims(nodes);
#if USE_GPU
    vector<dtype*> vals, losses;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), get_node_val);
    transform(nodes.begin(), nodes.end(), back_inserter(losses), get_node_loss);
    n3ldg_cuda::CrossEntropyLoss(vals, answers, nodes.size(), nodes.front()->getDim(), losses);
#if TEST_CUDA
    cpuCrossEntropyLoss(nodes, answers);
    for (Node *node : nodes) {
        n3ldg_cuda::Assert(node->val().verify("crossEntropyLoss"));
    }
#endif
#else
    cpuCrossEntropyLoss(nodes, answers);
#endif
}

vector<int> cpuPredict(const vector<Node *> &nodes) {
    vector<int> result;
    for (Node *node : nodes) {
        result.push_back(*std::max(node->getVal().v, node->getVal().v + node->getDim()));
    }
    return result;
}

vector<int> gpuPredict(const vector<Node *> &nodes) {
    vector<dtype*> vals;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), get_node_val);
    return n3ldg_cuda::Predict(vals, nodes.size(), nodes.front()->getDim());
}

vector<int> predict(const vector<Node *> &nodes) {
    validateEqualNodeDims(nodes);
#if USE_GPU
    return gpuPredict(nodes);
#else
    return cpuPredict(nodes);
#endif
}

#endif
