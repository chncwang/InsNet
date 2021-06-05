#ifndef INSNET_LSTM_H
#define INSNET_LSTM_H

#include "insnet/operator/linear.h"

namespace insnet {

struct LSTMParams : TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
    LinearParams input_hidden;
    LinearParams input_input;
    LinearParams output_hidden;
    LinearParams output_input;
    LinearParams forget_hidden;
    LinearParams forget_input;
    LinearParams cell_hidden;
    LinearParams cell_input;

    LSTMParams(const std::string &name);

    void init(int out_dim, int in_dim);

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(input_hidden, input_input, output_hidden, output_input, forget_hidden, forget_input,
                cell_hidden, cell_input);
    }

    int inDim() {
        return input_input.W().row();
    }

    int outDim() {
        return input_input.W().col();
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override;
#endif

protected:
    std::vector<Tunable<BaseParam> *> tunableComponents() override;
};

struct LSTMState {
    Node *hidden;
    Node *cell;
};

/// \ingroup module
/// Return the next LSTM hidden state, i.e., hi and ci.
///
/// **The operators inside guarantee that lstm with the same params and dropout value will be executed in batch.**
/// \param last_state The last lstm state containing h_i and c_i.
/// \param input The input vector.
/// \param params The LSTM parameters.
/// \param dropout The dropout value. The dropout will be added when returning the hidden vector.
/// \return The next LSTM state.
LSTMState lstm(LSTMState &last_state, Node &input, LSTMParams &params, dtype dropout);

/// \ingroup module
/// Return LSTM hidden states.
///
/// It is implemented using lstm(LSTMState &, Node &, LSTMParams &, dtype).
///
/// \param initial_state The initial state commonly remarked as h_0 and c_0.
/// \param inputs The input vectors.
/// \param params The LSTM parameters.
/// \param dropout The dropout value.
/// \return The hidden states.
std::vector<Node *> lstm(LSTMState &initial_state, const std::vector<Node *> &inputs,
        LSTMParams &params,
        dtype dropout);

}

#endif
