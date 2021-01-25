#ifndef N3LDGPLUS_LOSS_H
#define N3LDGPLUS_LOSS_H

#include <Node.h>

vector<int> cpuPredict(const vector<AtomicNode *> &nodes) {
    vector<int> result;
    for (AtomicNode *node : nodes) {
        result.push_back(std::max_element(node->getVal().v,
                    node->getVal().v + node->getDim()) - node->getVal().v);
    }
    return result;
}

#if USE_GPU

vector<int> gpuPredict(const vector<AtomicNode *> &nodes) {
    vector<dtype*> vals;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    return n3ldg_cuda::Predict(vals, nodes.size(), nodes.front()->getDim());
}

#endif

vector<int> predict(const vector<AtomicNode *> &nodes) {
    validateEqualNodeDims(nodes);
#if USE_GPU
    return gpuPredict(nodes);
#else
    return cpuPredict(nodes);
#endif
}

dtype cpuCrossEntropyLoss(vector<AtomicNode *> &nodes, const vector<int> &answers, dtype factor) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        int answer = answers.at(i);
        nodes.at(i)->loss()[answer] -= 1 / nodes.at(i)->getVal()[answer] * factor;
        loss -= log(nodes.at(i)->getVal()[answer]);
    }
    return loss * factor;
}

dtype crossEntropyLoss(vector<AtomicNode *> &nodes, const vector<int> &answers, dtype factor) {
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
    dtype loss = n3ldg_cuda::CrossEntropyLoss(vals, const_cast<vector<int>&>(answers),
            nodes.size(), factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuCrossEntropyLoss(nodes, answers, factor);
    for (AtomicNode *node : nodes) {
        n3ldg_cuda::Assert(node->loss().verify("crossEntropyLoss"));
    }
    cout << boost::format("cpu loss:%1% gpu:%2%") % cpu_loss % loss << endl;
#endif
    return loss;
#else
    return cpuCrossEntropyLoss(nodes, answers, factor);
#endif
}

float cpuMultiCrossEntropyLoss(vector<AtomicNode *> &nodes, const vector<vector<int>> &answers,
        dtype factor) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        AtomicNode &node = *nodes.at(i);
        const auto &answer = answers.at(i);
        for (int j = 0; j < node.getDim(); ++j) {
            dtype val = node.getVal()[j];
            node.loss()[j] += (answer.at(j) ?  -1 / val : 1 / (1 - val)) * factor;
            loss += (answer.at(j) ? -log(val): -log(1 - val));
        }
    }
    return loss * factor;
}

float cpuKLLoss(vector<AtomicNode *> &nodes, const vector<shared_ptr<vector<dtype>>> &answers,
        dtype factor) {
    cout << "node size:" << nodes.size() << endl;
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        AtomicNode *node = nodes.at(i);
        const auto &answer = answers.at(i);
        if (answer->size() < node->getDim()) {
            cerr << boost::format("cpuKLLoss - answer size is %1%, but node dim is %2%") %
                answer->size() % node->getDim() << endl;
            abort();
        }
        for (int j = 0; j < answer->size(); ++j) {
            loss -= answer->at(j) * log(node->getVal()[j]);
            node->loss()[j] -= factor * answer->at(j) / node->getVal()[j];
        }
    }

    return loss * factor;
}

pair<float, vector<int>> KLLoss(vector<AtomicNode *> &nodes,
        const vector<shared_ptr<vector<dtype>>> &answers,
        dtype factor) {
    if (nodes.size() != answers.size()) {
        cerr << "KLLoss - nodes size is not equal to answers size" << endl;
        abort();
    }
    validateEqualNodeDims(nodes);
#if USE_GPU
    vector<dtype *> vals, losses;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    transform(nodes.begin(), nodes.end(), back_inserter(losses), gpu_get_node_loss);
    dtype gpu_loss = n3ldg_cuda::KLCrossEntropyLoss(vals,
            const_cast<vector<shared_ptr<vector<dtype>>>&>(answers),
            nodes.size(), nodes.front()->getDim(), factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuKLLoss(nodes, answers, factor);
    cout << "KLLoss - gpu loss:" << gpu_loss << " cpu_loss:" << cpu_loss << endl;
    for (AtomicNode *node : nodes) {
        n3ldg_cuda::Assert(node->getLoss().verify("multiCrossEntropyLoss"));
    }
#endif
    dtype loss = gpu_loss;
#else
    dtype loss = cpuKLLoss(nodes, answers, factor);
#endif
    auto predicted_ids = predict(nodes);
    pair<float, vector<int>> result = make_pair(loss, predicted_ids);
    return result;
}

float multiCrossEntropyLoss(vector<AtomicNode *> &nodes, const vector<vector<int>> &answers,
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
    dtype gpu_loss = n3ldg_cuda::MultiCrossEntropyLoss(vals,
            const_cast<vector<vector<int>>&>(answers), nodes.size(),
            nodes.front()->getDim(), factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuMultiCrossEntropyLoss(nodes, answers, factor);
    cout << "multiCrossEntropyLoss - gpu loss:" << gpu_loss << " cpu_loss:" << cpu_loss << endl;
    for (AtomicNode *node : nodes) {
        n3ldg_cuda::Assert(node->getLoss().verify("multiCrossEntropyLoss"));
    }
#endif
    return gpu_loss;
#else
    return cpuMultiCrossEntropyLoss(nodes, answers, factor);
#endif
}

#endif
