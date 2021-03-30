#ifndef ATTENTION_BUILDER
#define ATTENTION_BUILDER

#include "n3ldg-plus/operator/linear.h"
#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/mul.h"
#include "n3ldg-plus/operator/matrix.h"
#include "n3ldg-plus/operator/softmax.h"
#include "n3ldg-plus/operator/add.h"
#include "n3ldg-plus/computation-graph/graph.h"

#include <memory>

namespace n3ldg_plus {

std::pair<BatchedNode *, BatchedNode *> dotAttention(Graph &graph, Node& key_matrix,
        Node& value_matrix,
        BatchedNode& guide,
        const std::vector<int> *matrix_cols = nullptr) {
    BatchedNode *raw_weights = tranMatrixMulVector(graph, key_matrix, guide,
            matrix_cols);
    BatchedNode *scaled_weight = scaled(graph, *raw_weights,
            1.0 / ::sqrt((dtype)guide.getDim()));
    scaled_weight = softmax(graph, *scaled_weight);
    int dim = guide.getDim();
    BatchedNode *hidden = matrixAndVectorMulti(graph, value_matrix, *scaled_weight,
            &dim);
    return std::make_pair(hidden, scaled_weight);
}

std::pair<BatchedNode *, BatchedNode *> dotAttention(Graph &graph, BatchedNode &key_matrix,
        BatchedNode &value_matrix,
        BatchedNode& guide,
        const std::vector<int> *matrix_cols = nullptr) {
    BatchedNode *raw_weights = tranMatrixMulVector(graph, key_matrix, guide,
            matrix_cols);
    BatchedNode *scaled_weight = scaled(graph, *raw_weights,
            1.0 / ::sqrt((dtype)guide.getDim()));
    scaled_weight = softmax(graph, *scaled_weight);
    int dim = guide.getDim();
    BatchedNode *hidden = matrixAndVectorMulti(graph, value_matrix, *scaled_weight,
            &dim);
    return std::make_pair(hidden, scaled_weight);
}

std::pair<BatchedNode *, BatchedNode *> dotAttention(Graph &graph, BatchedNode &key_matrix,
        BatchedNode &value_matrix,
        BatchedNode &query_matrix,
        int q_col,
        bool is_decoder) {
    int row = query_matrix.getDim() / q_col;
    BatchedNode *raw_weights = tranMatrixMulMatrix(graph, key_matrix, query_matrix,
            row, is_decoder);
    BatchedNode *scaled_weight = scaled(graph, *raw_weights,
            1.0 / ::sqrt((dtype)row));
    scaled_weight = softmax(graph, *scaled_weight, q_col);
    int v_col = value_matrix.getDim() / row;
    BatchedNode *hidden = matrixMulMatrix(graph, value_matrix, *scaled_weight, v_col);
    return std::make_pair(hidden, scaled_weight);
}

Node * dotAttentionWeights(Graph &cg, Node& key_matrix, Node& guide) {
    Node *raw_weights = tranMatrixMulVector(cg, key_matrix, guide);
    Node *scaled_weight = scaled(cg, *raw_weights,
            1.0 / ::sqrt((dtype)guide.getDim()));
    scaled_weight = softmax(cg, *scaled_weight, 1);
    return scaled_weight;
}

struct AdditiveAttentionParams : TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    LinearParam k, q, w3t;

    AdditiveAttentionParams(const std::string &name) : k(name + "-k"), q(name + "-q"),
    w3t(name + "-w3t") {}

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(k, q, w3t);
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

}

#endif
