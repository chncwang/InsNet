#ifndef N3LDG_PLUS_LSTM_H
#define N3LDG_PLUS_LSTM_H

#include "n3ldg-plus/operator/linear.h"

namespace n3ldg_plus {

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

LSTMState lstm(LSTMState &last_state, Node &input, LSTMParams &params, dtype dropout);

std::vector<Node *> lstm(LSTMState &initial_state, const std::vector<Node *> &inputs,
        LSTMParams &params,
        dtype dropout);

}

#endif
