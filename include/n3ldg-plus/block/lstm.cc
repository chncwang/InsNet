#include "n3ldg-plus/block/lstm.h"
#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/mul.h"
#include "n3ldg-plus/operator/add.h"

using std::string;
using std::vector;

namespace n3ldg_plus {

LSTMParams::LSTMParams(const string &name) : input_hidden(name + "-input_hidden"),
    input_input(name + "-input_input"), output_hidden(name + "-output_hidden"),
    output_input(name + "-output_input"), forget_hidden(name + "-forget_hidden"),
    forget_input(name + "-forget_input"), cell_hidden(name + "-cell_hidden"),
    cell_input(name + "-cell_input") {}

void LSTMParams::init(int out_dim, int in_dim) {
    input_hidden.init(out_dim, out_dim, false);
    input_input.init(out_dim, in_dim, true);
    output_hidden.init(out_dim, out_dim, false);
    output_input.init(out_dim, in_dim, true);
    forget_hidden.init(out_dim, out_dim, false);
    forget_input.init(out_dim, in_dim, true);
    cell_hidden.init(out_dim, out_dim, false);
    cell_input.init(out_dim, in_dim, true);
    forget_input.b().val().assignAll(1.0f);
}

#if USE_GPU
vector<cuda::Transferable *> LSTMParams::transferablePtrs() {
    return {&input_hidden, &input_input, &output_hidden, &output_input, &forget_input,
        &forget_hidden, &cell_hidden, &cell_input};
}
#endif

vector<Tunable<BaseParam> *> LSTMParams::tunableComponents() {
    return {&input_hidden, &input_input, &output_hidden, &output_input, &forget_hidden,
        &forget_input, &cell_hidden, &cell_input};
}

LSTMState lstm(LSTMState &last_state, Node &input, LSTMParams &params, dtype dropout_value) {
    Node &last_hidden = *last_state.hidden;
    Node &last_cell = *last_state.cell;
    Node *inputgate_hidden = linear(last_hidden, params.input_hidden);
    Node *inputgate_input = linear(input, params.input_input);
    Node *inputgate_add = add({inputgate_hidden, inputgate_input});
    Node *inputgate = sigmoid(*inputgate_add);

    Node *forgetgate_hidden = linear(last_hidden, params.forget_hidden);
    Node *forgetgate_input = linear(input, params.forget_input);
    Node *forgetgate_add = add({forgetgate_hidden, forgetgate_input});
    Node *forgetgate = sigmoid(*forgetgate_add);

    Node *outputgate_hidden = linear(last_hidden, params.output_hidden);
    Node *outputgate_input = linear(input, params.output_input);
    Node *outputgate_add = add({outputgate_hidden, outputgate_input});
    Node *outputgate = sigmoid(*outputgate_add);

    Node *halfcell_hidden = linear(last_hidden, params.cell_hidden);
    Node *halfcell_input = linear(input, params.cell_input);
    Node *halfcell_add = add({halfcell_hidden, halfcell_input});
    Node *halfcell = tanh(*halfcell_add);
    Node *inputfilter = pointwiseMultiply(*inputgate, *halfcell);
    Node *forgetfilter = pointwiseMultiply(last_cell, *forgetgate);
    Node *cell = add({inputfilter, forgetfilter});
    Node *halfhidden = tanh(*cell);
    Node *hidden = pointwiseMultiply(*halfhidden, *outputgate);
    hidden = dropout(*hidden, dropout_value);
    return {hidden, cell};
}

vector<Node *> lstm(LSTMState &initial_state, const vector<Node *> &inputs, LSTMParams &params,
        dtype dropout_value) {
    LSTMState last_state = initial_state;
    vector<Node *> results;

    for (Node *input : inputs) {
        last_state = lstm(last_state, *input, params, dropout_value);
        results.push_back(last_state.hidden);
    }

    return results;
}

}
