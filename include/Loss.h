#ifndef N3LDGPLUS_LOSS_H
#define N3LDGPLUS_LOSS_H

#include <Node.h>

inline vector<vector<int>> cpuPredict(const vector<Node *> &nodes, int row) {
    vector<vector<int>> result;
    result.reserve(nodes.size());
    for (Node *node : nodes) {
        int col = node->getDim() / row;
        if (col * row != node->getDim()) {
            cerr << boost::format("cpuPredict - row:%1% node dim:%2%\n") % row % node->getDim();
            abort();
        }
        vector<int> ids;
        ids.reserve(col);
        for (int i = 0; i < col; ++i) {
            ids.push_back(std::max_element(node->getVal().v + i * row,
                    node->getVal().v + (i + 1) * row) - node->getVal().v - i * row);
        }
        result.push_back(ids);
    }
    return result;
}

#if USE_GPU

inline vector<vector<int>> gpuPredict(const vector<Node *> &nodes, int row) {
    vector<dtype*> vals;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    vector<int> cols;
    cols.reserve(nodes.size());
    for (Node *node : nodes) {
        cols.push_back(node->getDim() / row);
    }
    return n3ldg_cuda::Predict(vals, nodes.size(), cols, row);
}

#endif

inline vector<vector<int>> predict(const vector<Node *> &nodes, int row) {
#if USE_GPU
    return gpuPredict(nodes, row);
#else
    return cpuPredict(nodes, row);
#endif
}

inline dtype cpuCrossEntropyLoss(vector<Node *> &nodes, int row,
        const vector<vector<int>> &answers_vector,
        dtype factor) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        const auto &answers = answers_vector.at(i);
        int col = nodes.at(i)->getDim() / row;
        if (col * row != nodes.at(i)->getDim()) {
            cerr << boost::format("cpuCrossEntropyLoss row:%1% node dim:%2%") % row %
                nodes.at(i)->getDim() << endl;
            abort();
        }
        for (int j = 0; j < col; ++j) {
            int answer = answers.at(j);
            nodes.at(i)->loss()[row * j + answer] -=
                1 / nodes.at(i)->getVal()[row * j + answer] * factor;
            loss -= log(nodes.at(i)->getVal()[row * j + answer]);
        }
    }
    return loss * factor;
}

inline dtype crossEntropyLoss(vector<Node *> &nodes, int row, const vector<vector<int>> &answers,
        dtype factor) {
    if (nodes.size() != answers.size()) {
        cerr << boost::format("crossEntropyLoss - node size is %1%, but answer size is %2%") %
            nodes.size() % answers.size() << endl;
        abort();
    }
#if USE_GPU
#if TEST_CUDA
    for (Node *node : nodes) {
        if (!node->loss().verify("crossEntropyLoss grad")) {
            node->loss().print();
            cout << node->loss().toString() << endl;
            abort();
        }
        if (!node->val().verify("crossEntropyLoss val")) {
            node->val().print();
            cout << node->val().toString() << endl;
            abort();
        }
    }
#endif
    vector<dtype*> vals, losses;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    transform(nodes.begin(), nodes.end(), back_inserter(losses), gpu_get_node_loss);
    dtype loss = n3ldg_cuda::CrossEntropyLoss(vals, answers, nodes.size(), row, factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuCrossEntropyLoss(nodes, row, answers, factor);
    for (Node *node : nodes) {
        if (!node->loss().verify("crossEntropyLoss")) {
            node->loss().print();
            cout << node->loss().toString() << endl;
            abort();
        }
    }
    cout << boost::format("cpu loss:%1% gpu:%2%") % cpu_loss % loss << endl;
#endif
    return loss;
#else
    return cpuCrossEntropyLoss(nodes, row, answers, factor);
#endif
}

inline float cpuMultiCrossEntropyLoss(vector<Node *> &nodes, const vector<vector<int>> &answers,
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

inline float cpuKLLoss(vector<Node *> &nodes, const vector<shared_ptr<vector<dtype>>> &answers,
        dtype factor) {
    cout << "node size:" << nodes.size() << endl;
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        Node *node = nodes.at(i);
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

inline pair<float, vector<int>> KLLoss(vector<Node *> &nodes,
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
    for (Node *node : nodes) {
        n3ldg_cuda::Assert(node->getLoss().verify("multiCrossEntropyLoss"));
    }
#endif
    dtype loss = gpu_loss;
#else
    dtype loss = cpuKLLoss(nodes, answers, factor);
#endif
    auto predicted_ids = predict(nodes, nodes.front()->getDim());
    vector<int> ids;
    for (const auto &x : predicted_ids) {
        if (x.size() != 1) {
            cerr << boost::format("KLLoss x size:%1%\n") % x.size();
            abort();
        }
        ids.push_back(x.front());
    }
    pair<float, vector<int>> result = make_pair(loss, ids);
    return result;
}

inline float multiCrossEntropyLoss(vector<Node *> &nodes, const vector<vector<int>> &answers,
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
    for (Node *node : nodes) {
        n3ldg_cuda::Assert(node->getLoss().verify("multiCrossEntropyLoss"));
    }
#endif
    return gpu_loss;
#else
    return cpuMultiCrossEntropyLoss(nodes, answers, factor);
#endif
}

#endif
