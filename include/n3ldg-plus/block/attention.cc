#include "n3ldg-plus/block/attention.h"

#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/matrix.h"
#include "n3ldg-plus/operator/softmax.h"

using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::max;

namespace n3ldg_plus {

pair<BatchedNode *, BatchedNode *> dotAttention(Node& key_matrix,
        Node& value_matrix,
        BatchedNode& guide,
        const vector<int> *matrix_cols) {
    BatchedNode *raw_weights = tranMatrixMulVector(key_matrix, guide, matrix_cols);
    BatchedNode *scaled_weight = scaled(*raw_weights, 1.0 / ::sqrt((dtype)guide.getDim()));
    scaled_weight = softmax(*scaled_weight);
    int dim = guide.getDim();
    BatchedNode *hidden = matrixAndVectorMulti(value_matrix, *scaled_weight,
            &dim);
    return make_pair(hidden, scaled_weight);
}

pair<BatchedNode *, BatchedNode *> dotAttention(BatchedNode &key_matrix,
        BatchedNode &value_matrix,
        BatchedNode& guide,
        const vector<int> *matrix_cols) {
    BatchedNode *raw_weights = tranMatrixMulVector(key_matrix, guide, matrix_cols);
    BatchedNode *scaled_weight = scaled(*raw_weights, 1.0 / ::sqrt((dtype)guide.getDim()));
    scaled_weight = softmax(*scaled_weight);
    int dim = guide.getDim();
    BatchedNode *hidden = matrixAndVectorMulti(value_matrix, *scaled_weight, &dim);
    return make_pair(hidden, scaled_weight);
}

pair<BatchedNode *, BatchedNode *> dotAttention(BatchedNode &key_matrix,
        BatchedNode &value_matrix,
        BatchedNode &query_matrix,
        int q_col,
        bool is_decoder) {
    int row = query_matrix.getDim() / q_col;
    BatchedNode *raw_weights = tranMatrixMulMatrix(key_matrix, query_matrix, row, is_decoder);
    BatchedNode *scaled_weight = scaled(*raw_weights, 1.0 / ::sqrt((dtype)row));
    scaled_weight = softmax(*scaled_weight, q_col);
    int v_col = value_matrix.getDim() / row;
    BatchedNode *hidden = matrixMulMatrix(value_matrix, *scaled_weight, v_col);
    return make_pair(hidden, scaled_weight);
}

Node * dotAttentionWeights(Node& key_matrix, Node& guide) {
    Node *raw_weights = tranMatrixMulVector(key_matrix, guide);
    Node *scaled_weight = scaled(*raw_weights, 1.0 / ::sqrt((dtype)guide.getDim()));
    scaled_weight = softmax(*scaled_weight, 1);
    return scaled_weight;
}

AdditiveAttentionParams::AdditiveAttentionParams(const string &name) : k(name + "-k"),
    q(name + "-q"), w3t(name + "-w3t") {}

void AdditiveAttentionParams::init(int k_size, int q_size) {
    int out = max(k_size, q_size);
    k.init(out, k_size, false);
    q.init(out, q_size, false);
    w3t.init(1, out, false);
}

vector<Tunable<BaseParam> *> AdditiveAttentionParams::tunableComponents() {
    return {&k, &q, &w3t};
}

}
