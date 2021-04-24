#include "n3ldg-plus/loss/loss.h"

using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using std::shared_ptr;
using std::pair;
using std::max_element;

namespace n3ldg_plus {

namespace {

vector<vector<int>> cpuPredict(const vector<Node *> &nodes, int row) {
    vector<vector<int>> result;
    result.reserve(nodes.size());
    for (Node *node : nodes) {
        int col = node->getDim() / row;
        if (col * row != node->getDim()) {
            cerr << fmt::format("cpuPredict - row:{} node dim:{}\n", row, node->getDim());
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
    return node->loss().value;
};

vector<vector<int>> gpuPredict(const vector<Node *> &nodes, int row) {
    vector<dtype*> vals;
    transform(nodes.begin(), nodes.end(), back_inserter(vals), gpu_get_node_val);
    vector<int> cols;
    cols.reserve(nodes.size());
    for (Node *node : nodes) {
        cols.push_back(node->getDim() / row);
    }
    return cuda::Predict(vals, nodes.size(), cols, row);
}

#endif
}

vector<vector<int>> predict(const vector<Node *> &nodes, int row) {
#if USE_GPU
    return gpuPredict(nodes, row);
#else
    return cpuPredict(nodes, row);
#endif
}

namespace {

dtype cpuLikelihoodLoss(vector<Node *> &nodes, int row, const vector<vector<int>> &answers_vector,
        dtype factor) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        const auto &answers = answers_vector.at(i);
        int col = nodes.at(i)->getDim() / row;
        if (col * row != nodes.at(i)->getDim()) {
            cerr << fmt::format("cpuCrossEntropyLoss row:{} node dim:{}\n", row,
                nodes.at(i)->getDim());
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

}

dtype NLLoss(vector<Node *> &nodes, int row, const vector<vector<int>> &answers,
        dtype factor) {
    if (nodes.size() != answers.size()) {
        cerr << fmt::format("crossEntropyLoss - node size is {}, but answer size is {}\n",
            nodes.size(), answers.size());
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
    dtype loss = cuda::CrossEntropyLoss(vals, answers, nodes.size(), row, factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuLikelihoodLoss(nodes, row, answers, factor);
    for (Node *node : nodes) {
        if (!node->loss().verify("crossEntropyLoss")) {
            node->loss().print();
            cout << node->loss().toString() << endl;
            abort();
        }
    }
    cout << fmt::format("cpu loss:{} gpu:{}\n", cpu_loss, loss);
#endif
    return loss;
#else
    return cpuLikelihoodLoss(nodes, row, answers, factor);
#endif
}

namespace {

float cpuBinaryLikelihoodLoss(vector<Node *> &nodes, const vector<vector<int>> &answers,
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

float cpuKLDivergenceLoss(vector<Node *> &nodes,
        const vector<shared_ptr<vector<dtype>>> &answers,
        dtype factor) {
    dtype loss = 0;
    for (int i = 0; i < nodes.size(); ++i) {
        Node *node = nodes.at(i);
        const auto &answer = answers.at(i);
        if (answer->size() < node->getDim()) {
            cerr << fmt::format("cpuKLLoss - answer size is {}, but node dim is {}",
                answer->size(), node->getDim());
            abort();
        }
        for (int j = 0; j < answer->size(); ++j) {
            loss -= answer->at(j) * log(node->getVal()[j]);
            node->loss()[j] -= factor * answer->at(j) / node->getVal()[j];
        }
    }

    return loss * factor;
}

}

pair<float, vector<int>> KLDivergenceLoss(vector<Node *> &nodes,
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
    dtype gpu_loss = cuda::KLCrossEntropyLoss(vals,
            const_cast<vector<shared_ptr<vector<dtype>>>&>(answers),
            nodes.size(), nodes.front()->getDim(), factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuKLDivergenceLoss(nodes, answers, factor);
    cout << "KLLoss - gpu loss:" << gpu_loss << " cpu_loss:" << cpu_loss << endl;
    for (Node *node : nodes) {
        cuda::Assert(node->getLoss().verify("multiCrossEntropyLoss"));
    }
#endif
    dtype loss = gpu_loss;
#else
    dtype loss = cpuKLDivergenceLoss(nodes, answers, factor);
#endif
    auto predicted_ids = predict(nodes, nodes.front()->getDim());
    vector<int> ids;
    for (const auto &x : predicted_ids) {
        if (x.size() != 1) {
            cerr << fmt::format("KLLoss x size:{}\n", x.size());
            abort();
        }
        ids.push_back(x.front());
    }
    pair<float, vector<int>> result = make_pair(loss, ids);
    return result;
}

float binrayLikelihoodLoss(vector<Node *> &nodes, const vector<vector<int>> &answers,
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
    dtype gpu_loss = cuda::MultiCrossEntropyLoss(vals,
            const_cast<vector<vector<int>>&>(answers), nodes.size(),
            nodes.front()->getDim(), factor, losses);
#if TEST_CUDA
    dtype cpu_loss = cpuBinaryLikelihoodLoss(nodes, answers, factor);
    cout << "multiCrossEntropyLoss - gpu loss:" << gpu_loss << " cpu_loss:" << cpu_loss << endl;
    for (Node *node : nodes) {
        cuda::Assert(node->getLoss().verify("multiCrossEntropyLoss"));
    }
#endif
    return gpu_loss;
#else
    return cpuBinaryLikelihoodLoss(nodes, answers, factor);
#endif
}

}
