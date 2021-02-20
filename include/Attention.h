#ifndef ATTENTION_BUILDER
#define ATTENTION_BUILDER

#include "MyLib.h"
#include "Node.h"
#include "UniOP.h"
#include "Graph.h"
#include "AttentionHelp.h"
#include "AtomicOP.h"
#include <memory>
#include <boost/format.hpp>

namespace n3ldg_plus {

pair<BatchedNode *, BatchedNode *> dotAttention(Graph &graph, Node& key_matrix, Node& value_matrix,
        BatchedNode& guide,
        const vector<int> *matrix_cols = nullptr) {
    BatchedNode *raw_weights = n3ldg_plus::tranMatrixMulVector(graph, key_matrix, guide,
            matrix_cols);
    BatchedNode *scaled_weight = n3ldg_plus::scaled(graph, *raw_weights,
            1.0 / ::sqrt((dtype)guide.getDim()));
    scaled_weight = n3ldg_plus::softmax(graph, *scaled_weight);
    int dim = guide.getDim();
    BatchedNode *hidden = n3ldg_plus::matrixAndVectorMulti(graph, value_matrix, *scaled_weight,
            &dim);
    return make_pair(hidden, scaled_weight);
}

pair<BatchedNode *, BatchedNode *> dotAttention(Graph &graph, BatchedNode &key_matrix,
        BatchedNode &value_matrix,
        BatchedNode& guide,
        const vector<int> *matrix_cols = nullptr) {
    BatchedNode *raw_weights = n3ldg_plus::tranMatrixMulVector(graph, key_matrix, guide,
            matrix_cols);
    BatchedNode *scaled_weight = n3ldg_plus::scaled(graph, *raw_weights,
            1.0 / ::sqrt((dtype)guide.getDim()));
    scaled_weight = n3ldg_plus::softmax(graph, *scaled_weight);
    int dim = guide.getDim();
    BatchedNode *hidden = n3ldg_plus::matrixAndVectorMulti(graph, value_matrix, *scaled_weight,
            &dim);
    return make_pair(hidden, scaled_weight);
}

pair<BatchedNode *, BatchedNode *> dotAttention(Graph &graph, BatchedNode &key_matrix,
        BatchedNode &value_matrix,
        BatchedNode &query_matrix,
        int matrix_col) {
    BatchedNode *raw_weights = n3ldg_plus::tranMatrixMulMatrix(graph, key_matrix, query_matrix,
            matrix_col);
    int dim = key_matrix.getDim() / matrix_col;
    BatchedNode *scaled_weight = n3ldg_plus::scaled(graph, *raw_weights,
            1.0 / ::sqrt((dtype)dim));
    vector<int> offsets(matrix_col);
    for (int i = 0; i < matrix_col; ++i) {
        offsets.at(i) = i * matrix_col;
    }
    scaled_weight = n3ldg_plus::split(graph, *scaled_weight, matrix_col, offsets);
    scaled_weight = n3ldg_plus::softmax(graph, *scaled_weight);
    BatchedNode *hidden = n3ldg_plus::matrixAndVectorMulti(graph, value_matrix, *scaled_weight,
            &dim);
    return make_pair(hidden, scaled_weight);
}

Node * dotAttentionWeights(Graph &cg, Node& key_matrix, Node& guide) {
    Node *raw_weights = n3ldg_plus::tranMatrixMulVector(cg, key_matrix, guide);
    Node *scaled_weight = n3ldg_plus::scaled(cg, *raw_weights,
            1.0 / ::sqrt((dtype)guide.getDim()));
    scaled_weight = n3ldg_plus::softmax(cg, *scaled_weight, 1);
    return scaled_weight;
}

}

struct AdditiveAttentionParams : public N3LDGSerializable, TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    UniParams k, q, w3t;

    AdditiveAttentionParams(const string &name) : k(name + "-k"), q(name + "-q"),
    w3t(name + "-w3t") {}

    Json::Value toJson() const override {
        Json::Value json;
        json["k"] = k.toJson();
        json["q"] = q.toJson();
        json["w3t"] = w3t.toJson();
        return json;
    }


    void fromJson(const Json::Value &json) override {
        k.fromJson(json["k"]);
        q.fromJson(json["q"]);
        w3t.fromJson(json["w3t"]);
    }

    void init(int k_size, int q_size) {
        int out = std::max(k_size, q_size);
        k.init(out, k_size, false);
        q.init(out, q_size, false);
        w3t.init(1, out, false);
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override {
        return {&k, &q, &w3t};
    }

    virtual std::string name() const {
        return "AdditiveAttention";
    }
#endif

protected:
    std::vector<Tunable<BaseParam> *> tunableComponents() override {
        return {&k, &q, &w3t};
    }
};

class AdditiveAttentionBuilder {
public:
    vector<Node *> _weights;
    Node* _hidden;

    void forward(Graph &graph, AdditiveAttentionParams &params, vector<Node *>& values,
            Node& guide) {
        using namespace n3ldg_plus;
        if (values.empty()) {
            std::cerr << "empty inputs for attention operation" << std::endl;
            abort();
        }

        Node *q = linear(graph, guide, params.q);

        for (int idx = 0; idx < values.size(); idx++) {
            Node *k = linear(graph, *values.at(idx), params.k);
            Node *sum = add(graph, {k, q});
            Node *nonlinear = tanh(graph, *sum);
            Node *w = linear(graph, *nonlinear, params.w3t);
            _weights.push_back(w);
        }

        _hidden = attention(graph, values, _weights);
    }
};

namespace n3ldg_plus {

vector<Node *> additiveAttentionWeights(Graph &graph, AdditiveAttentionParams &params,
        vector<Node *> &values,
        Node& guide) {
    Node *q = linear(graph, guide, params.q);
    vector<Node *> weights;

    for (int idx = 0; idx < values.size(); idx++) {
        Node *k = linear(graph, *values.at(idx), params.k);
        Node *sum = add(graph, {k, q});
        Node *nonlinear = tanh(graph, *sum);
        Node *w = linear(graph, *nonlinear, params.w3t);
        weights.push_back(w);
    }
    return weights;
}

}

#endif
