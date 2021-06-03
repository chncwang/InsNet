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
/// Return the next GRU hidden state.
///
/// **The operators inside guarantee that gru with the same params and dropout value will be executed in batch.**
/// \param last_state The last hidden state.
/// \param input The input vector.
/// \param params The GRU parameters.
/// \param dropout The dropout value. The dropout will be added when returning the result vector.
/// \return The next hidden state.
Node *gru(Node &last_state, Node &input, GRUParams &params, dtype dropout);

/// \ingroup module
/// Return GRU hidden states.
///
/// It is implemented using gru(Node &, Node &, GRUParams &, dtype).
///
/// \param initial_state The initial hidden state commonly remarked as h_0.
/// \param inputs The input vectors.
/// \param params The GRU parameters.
/// \param dropout The dropout value.
/// \return The hidden states.
std::vector<Node *> gru(Node &initial_state, const std::vector<Node *> &inputs, GRUParams &params,
        dtype dropout);

}

#endif
