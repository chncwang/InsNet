#ifndef N3LDG_PLUS_GRU_H
#define N3LDG_PLUS_GRU_H

#include "n3ldg-plus/operator/linear.h"

namespace n3ldg_plus {

struct GRUParams : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
    LinearParams update_input;
    LinearParams update_hidden;
    LinearParams reset_input;
    LinearParams reset_hidden;
    LinearParams candidate_input;
    LinearParams candidate_hidden;

    GRUParams(const std::string &name);

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(update_input, update_hidden, reset_input, reset_hidden, candidate_input,
                candidate_hidden);
    }

    void init(int out_size, int in_size);

    int inDim() {
        return update_input.W().row();
    }

    int outDim() {
        return update_input.W().col();
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override;
#endif

protected:
    std::vector<Tunable<BaseParam> *> tunableComponents() override;
};

/// \ingroup module
/// Return the next GRU state.
///
/// **The operators inside guarantee that gru with the same params and dropout value will be executed in batch.**
/// \param last_state The last hidden state.
/// \param input The input vector.
/// \param params The GRU parameters.
/// \param dropout The dropout value.
Node *gru(Node &last_state, Node &input, GRUParams &params, dtype dropout);

std::vector<Node *> gru(Node &initial_state, const std::vector<Node *> &inputs, GRUParams &params,
        dtype dropout_value);

}

#endif
