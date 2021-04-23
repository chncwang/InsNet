#ifndef N3LDG_PLUS_ATTENTION_H
#define N3LDG_PLUS_ATTENTION_H

#include "n3ldg-plus/operator/linear.h"
#include "n3ldg-plus/computation-graph/graph.h"
#include "n3ldg-plus/param/base-param.h"

namespace n3ldg_plus {

std::pair<BatchedNode *, BatchedNode *> dotAttention(Node& key_matrix, Node& value_matrix,
        BatchedNode& guide,
        const std::vector<int> *matrix_cols = nullptr);

std::pair<BatchedNode *, BatchedNode *> dotAttention(BatchedNode &key_matrix,
        BatchedNode &value_matrix,
        BatchedNode& guide,
        const std::vector<int> *matrix_cols = nullptr);

std::pair<BatchedNode *, BatchedNode *> dotAttention(BatchedNode &key_matrix,
        BatchedNode &value_matrix,
        BatchedNode &query_matrix,
        int q_col,
        bool is_decoder);

Node * dotAttentionWeights(Node& key_matrix, Node& guide);

struct AdditiveAttentionParams : TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
    LinearParam k, q, vt;

    AdditiveAttentionParams(const std::string &name);

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(k, q, vt);
    }

    void init(int k_size, int q_size);

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override {
        return {&k, &q, &vt};
    }
#endif

protected:
    std::vector<Tunable<BaseParam> *> tunableComponents() override;
};

std::pair<Node *, Node *> additiveAttention(Node &guide, Node &value, int value_col,
        AdditiveAttentionParams &params);

}

#endif
