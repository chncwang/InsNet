#include "n3ldg-plus/block/lstm.h"
#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/mul.h"
#include "n3ldg-plus/operator/add.h"

using std::string;
using std::vector;

namespace n3ldg_plus {

LSTMParam::LSTMParam(const string &name) : input_hidden(name + "-input_hidden"),
    input_input(name + "-input_input"), output_hidden(name + "-output_hidden"),
    output_input(name + "-output_input"), forget_hidden(name + "-forget_hidden"),
    forget_input(name + "-forget_input"), cell_hidden(name + "-cell_hidden"),
    cell_input(name + "-cell_input") {}

void LSTMParam::init(int nOSize, int nISize) {
    input_hidden.init(nOSize, nOSize, false);
    input_input.init(nOSize, nISize, true);
    output_hidden.init(nOSize, nOSize, false);
    output_input.init(nOSize, nISize, true);
    forget_hidden.init(nOSize, nOSize, false);
    forget_input.init(nOSize, nISize, true);
    cell_hidden.init(nOSize, nOSize, false);
    cell_input.init(nOSize, nISize, true);
    forget_input.b().val().assignAll(1.0f);
}

#if USE_GPU
vector<cuda::Transferable *> LSTMParam::transferablePtrs() {
    return {&input_hidden, &input_input, &output_hidden, &output_input, &forget_input,
        &forget_hidden, &cell_hidden, &cell_input};
}
#endif

vector<Tunable<BaseParam> *> LSTMParam::tunableComponents() {
    return {&input_hidden, &input_input, &output_hidden, &output_input, &forget_hidden,
        &forget_input, &cell_hidden, &cell_input};
}

void LSTMBuilder::step(LSTMParam &lstm_params, Node &input, Node &h0, Node &c0,
        dtype dropout_value) {
    Node *last_hidden, *last_cell;
    int len = hiddens_.size();
    if (len == 0) {
        last_hidden = &h0;
        last_cell = &c0;
    } else {
        last_hidden = hiddens_.at(len - 1);
        last_cell = cells_.at(len - 1);
    }

    using namespace n3ldg_plus;
    Node *inputgate_hidden = linear(*last_hidden, lstm_params.input_hidden);
    Node *inputgate_input = linear(input, lstm_params.input_input);
    Node *inputgate_add = add({inputgate_hidden, inputgate_input});
    Node *inputgate = sigmoid(*inputgate_add);

    Node *forgetgate_hidden = linear(*last_hidden, lstm_params.forget_hidden);
    Node *forgetgate_input = linear(input, lstm_params.forget_input);
    Node *forgetgate_add = add({forgetgate_hidden, forgetgate_input});
    Node *forgetgate = sigmoid(*forgetgate_add);

    Node *outputgate_hidden = linear(*last_hidden, lstm_params.output_hidden);
    Node *outputgate_input = linear(input, lstm_params.output_input);
    Node *outputgate_add = add({outputgate_hidden, outputgate_input});
    Node *outputgate = sigmoid(*outputgate_add);

    Node *halfcell_hidden = linear(*last_hidden, lstm_params.cell_hidden);
    Node *halfcell_input = linear(input, lstm_params.cell_input);
    Node *halfcell_add = add({halfcell_hidden, halfcell_input});
    Node *halfcell = tanh(*halfcell_add);
    Node *inputfilter = pointwiseMultiply(*inputgate, *halfcell);
    Node *forgetfilter = pointwiseMultiply(*last_cell, *forgetgate);
    Node *cell = add({inputfilter, forgetfilter});
    Node *halfhidden = tanh(*cell);
    Node *hidden = pointwiseMultiply(*halfhidden, *outputgate);
    hidden = dropout(*hidden, dropout_value);
    hiddens_.push_back(hidden);
    cells_.push_back(cell);
}

}
