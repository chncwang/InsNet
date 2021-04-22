#ifndef N3LDG_PLUS_LSTM_H
#define N3LDG_PLUS_LSTM_H

#include "n3ldg-plus/operator/linear.h"

namespace n3ldg_plus {

struct LSTMParam : TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
    LinearParam input_hidden;
    LinearParam input_input;
    LinearParam output_hidden;
    LinearParam output_input;
    LinearParam forget_hidden;
    LinearParam forget_input;
    LinearParam cell_hidden;
    LinearParam cell_input;

    LSTMParam(const ::std::string &name);

    void init(int nOSize, int nISize);

    int inDim() {
        return input_input.W().outDim();
    }

    int outDim() {
        return input_input.W().inDim();
    }

#if USE_GPU
    ::std::vector<Transferable *> transferablePtrs() override;
#endif

protected:
    ::std::vector<Tunable<BaseParam> *> tunableComponents() override;
};

class LSTMBuilder {
public:
    int size() const {
        return hiddens_.size();
    }

    void step(LSTMParam &lstm_params, Node &input, Node &h0, Node &c0, dtype dropout_value);

    const ::std::vector<Node *> &cells() {
        return cells_;
    }

    const ::std::vector<Node *> &hiddens() {
        return hiddens_;
    }

private:
    ::std::vector<Node*> cells_;
    ::std::vector<Node*> hiddens_;
};

}

#endif
