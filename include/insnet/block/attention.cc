#include "insnet/block/attention.h"

#include "insnet/operator/add.h"
#include "insnet/operator/atomic.h"
#include "insnet/operator/broadcast.h"
#include "insnet/operator/matrix.h"
#include "insnet/operator/softmax.h"

using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::min;

namespace insnet {

pair<BatchedNode *, BatchedNode *> dotAttention(BatchedNode &key_matrix,
        BatchedNode &value_matrix,
        BatchedNode &query_matrix,
        int row,
        bool is_decoder) {
    BatchedNode *raw_weights = tranMatrixMulMatrix(key_matrix, query_matrix, row, is_decoder);
    BatchedNode *scaled_weight = mul(*raw_weights, 1.0 / ::sqrt((dtype)row));
    int v_col = value_matrix.size() / row;
    scaled_weight = softmax(*scaled_weight, v_col);
    BatchedNode *hidden = matrixMulMatrix(value_matrix, *scaled_weight, v_col);
    return make_pair(hidden, scaled_weight);
}

AdditiveAttentionParams::AdditiveAttentionParams(const string &name) : k(name + "-k"),
    q(name + "-q"), vt(name + "-vt") {}

void AdditiveAttentionParams::init(int k_size, int q_size) {
    int out = min(k_size, q_size);
    k.init(out, k_size, false);
    q.init(out, q_size, false);
    vt.init(1, out, false);
}

vector<Tunable<BaseParam> *> AdditiveAttentionParams::tunableComponents() {
    return {&k, &q, &vt};
}

pair<Node *, Node *> additiveAttention(Node &guide, Node &value, int value_col,
        AdditiveAttentionParams &params) {
    Node *value_matrix = linear(value, params.k);
    Node *q = linear(guide, params.q);
    q = expandColumnwisely(*q, value_col);
    Node *sum = add({q, value_matrix});
    sum = tanh(*sum);
    Node *score = linear(*sum, params.vt);
    Node *weight = softmax(*score);
    Node *result = matmul(value, *weight, weight->size());
    return make_pair(result, weight);
}

}
