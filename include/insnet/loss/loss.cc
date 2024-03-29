#include "insnet/loss/loss.h"

using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::pair;
using std::max_element;

namespace insnet {

namespace {

vector<vector<int>> cpuPredict(const vector<Node *> &nodes, int row) {
    vector<vector<int>> result;
    result.reserve(nodes.size());
    for (Node *node : nodes) {
        int col = node->size() / row;
        if (col * row != node->size()) {
            cerr << fmt::format("cpuPredict - row:{} node dim:{}\n", row, node->size());
            abort();
        }
        vector<int> ids;
        ids.reserve(col);
        for (int i = 0; i < col; ++i) {
            ids.push_back(max_element(node->getVal().v + i * row,
                    node->getVal().v + (i + 1) * row) - node->getVal().v - i * row);
        }
        result.push_back(ids);
    }
    return result;
}

#if USE_GPU

auto gpu_get_node_val = [](Node *node) {
    return node->val().value;
};

auto gpu_get_node_loss = [](Node *node) {
    return node->grad().value;
};

vector<vector<int>> gpuPredict(const vector<Node *> &nodes, int row) {
    vector<dtype*> vals;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    vector<int> cols;
    cols.reserve(nodes.size());
    for (Node *node : nodes) {
        cols.push_back(node->size() / row);
    }
    return cuda::Predict(vals, nodes.size(), cols, row);
}

#endif
}

vector<vector<int>> argmax(const vector<Node *> &nodes, int row) {
#if USE_GPU
    return gpuPredict(nodes, row);
#else
    return cpuPredict(nodes, row);
#endif
}

namespace {

dtype cpuNLLLoss(vector<Node *> &nodes, int row, const vector<vector<int>> &answers_vector,
        dtype factor) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        const auto &answers = answers_vector.at(i);
        int col = nodes.at(i)->size() / row;
        if (col * row != nodes.at(i)->size()) {
            cerr << fmt::format("cpuCrossEntropyLoss row:{} node dim:{}\n", row,
                nodes.at(i)->size());
            abort();
        }
        for (int j = 0; j < col; ++j) {
            int answer = answers.at(j);
            nodes.at(i)->grad()[row * j + answer] -= factor;
            dtype v = nodes.at(i)->getVal()[row * j + answer]; 
            loss -= v;
        }
    }
    return loss * factor;
}

}

dtype NLLLoss(vector<Node *> &nodes, int row, const vector<vector<int>> &answers,
        dtype factor) {
    if (nodes.size() != answers.size()) {
        cerr << fmt::format("crossEntropyLoss - node size is {}, but answer size is {}\n",
            nodes.size(), answers.size());
        abort();
    }

    initAndZeroGrads(nodes);

#if USE_GPU
#if TEST_CUDA
    for (Node *node : nodes) {
        if (!node->grad().verify("crossEntropyLoss grad")) {
            node->grad().print();
            cout << node->grad().toString() << endl;
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
    dtype loss = cuda::NLLLoss(vals, answers, nodes.size(), row, factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuNLLLoss(nodes, row, answers, factor);
    for (Node *node : nodes) {
        if (!node->grad().verify("crossEntropyLoss")) {
            node->grad().print();
            cout << node->grad().toString() << endl;
            abort();
        }
    }
    cout << fmt::format("cpu loss:{} gpu:{}\n", cpu_loss, loss);
#endif
    return loss;
#else
    return cpuNLLLoss(nodes, row, answers, factor);
#endif
}

namespace {

float cpuBinaryLikelihoodLoss(vector<Node *> &nodes, const vector<vector<int> *> &answers,
        dtype factor) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        Node &node = *nodes.at(i);
        const auto &answer = answers.at(i);
        for (int j = 0; j < node.size(); ++j) {
            dtype val = node.getVal()[j];
            node.grad()[j] += (answer->at(j) ?  -1 / val : 1 / (1 - val)) * factor;
            loss += (answer->at(j) ? -log(val): -log(1 - val));
        }
    }
    return loss * factor;
}

float cpuKLDivergenceLoss(vector<Node *> &nodes, const vector<vector<dtype> *> &answers,
        dtype factor) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        Node *node = nodes.at(i);
        const auto &answer = answers.at(i);
        if (answer->size() < node->size()) {
            cerr << fmt::format("cpuKLLoss - answer size is {}, but node dim is {}",
                answer->size(), node->size());
            abort();
        }
        for (int j = 0; j < answer->size(); ++j) {
            loss -= answer->at(j) * log(node->getVal()[j]);
            node->grad()[j] -= factor * answer->at(j) / node->getVal()[j];
        }
    }

    return loss * factor;
}

}

dtype KLDivLoss(vector<Node *> &nodes, const vector<vector<dtype> *> &answers, dtype factor) {
    if (nodes.size() != answers.size()) {
        cerr << "KLLoss - nodes size is not equal to answers size" << endl;
        abort();
    }
    initAndZeroGrads(nodes);
    validateEqualNodeDims(nodes);
#if USE_GPU
    vector<dtype *> vals, losses;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    transform(nodes.begin(), nodes.end(), back_inserter(losses), gpu_get_node_loss);
    dtype gpu_loss = cuda::KLCrossEntropyLoss(vals, const_cast<vector<vector<dtype> *> &>(answers),
            nodes.size(), nodes.front()->size(), factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuKLDivergenceLoss(nodes, answers, factor);
    cout << "KLLoss - gpu loss:" << gpu_loss << " cpu_loss:" << cpu_loss << endl;
    for (Node *node : nodes) {
        cuda::Assert(node->getGrad().verify("multiCrossEntropyLoss"));
    }
#endif
    dtype loss = gpu_loss;
#else
    dtype loss = cpuKLDivergenceLoss(nodes, answers, factor);
#endif
    return loss;
}

float BCELoss(vector<Node *> &nodes, const vector<vector<int> *> &answers, dtype factor) {
    if (nodes.size() != answers.size()) {
        cerr << "multiCrossEntropyLoss - nodes size is not equal to answers size" << endl;
        abort();
    }
    initAndZeroGrads(nodes);
    validateEqualNodeDims(nodes);
#if USE_GPU
    vector<dtype *> vals, losses;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    transform(nodes.begin(), nodes.end(), back_inserter(losses), gpu_get_node_loss);
    dtype gpu_loss = cuda::MultiCrossEntropyLoss(vals, const_cast<vector<vector<int> *>&>(answers),
            nodes.size(), nodes.front()->size(), factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuBinaryLikelihoodLoss(nodes, answers, factor);
    cout << "multiCrossEntropyLoss - gpu loss:" << gpu_loss << " cpu_loss:" << cpu_loss << endl;
    for (Node *node : nodes) {
        cuda::Assert(node->getGrad().verify("multiCrossEntropyLoss"));
    }
#endif
    return gpu_loss;
#else
    return cpuBinaryLikelihoodLoss(nodes, answers, factor);
#endif
}

}
