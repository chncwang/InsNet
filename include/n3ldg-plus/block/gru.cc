#include "n3ldg-plus/block/gru.h"

#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/mul.h"
#include "n3ldg-plus/operator/add.h"
#include "n3ldg-plus/operator/bucket.h"
#include "n3ldg-plus/operator/sub.h"

using std::vector;
using std::string;

namespace n3ldg_plus {

GRUParams::GRUParams(const string &name) : update_input(name + "-update_input"),
    update_hidden(name + "-update_hidden"), reset_input(name + "reset_input"),
    reset_hidden(name + "reset_hidden"), candidate_input(name + "candidate_input"),
    candidate_hidden(name + "candidate_hidden") {}

void GRUParams::init(int out_size, int in_size) {
    update_input.init(out_size, in_size);
    update_hidden.init(out_size, out_size);
    reset_input.init(out_size, in_size);
    reset_hidden.init(out_size, out_size);
    candidate_input.init(out_size, in_size);
    candidate_hidden.init(out_size, out_size);
}

#if USE_GPU
vector<cuda::Transferable *> GRUParams::transferablePtrs() {
    return {&update_input, &update_hidden, &reset_input, &reset_hidden, &candidate_input,
        &candidate_hidden};
}
#endif

vector<Tunable<BaseParam> *> GRUParams::tunableComponents() {
    return {&update_input, &update_hidden, &reset_input, &reset_hidden, &candidate_input,
        &candidate_hidden};
}

Node *gru(Node &last_state, Node &input, GRUParams &params, dtype dropout_value) {
    Node *update_input = linear(input, params.update_input);
    Node *update_hidden = linear(last_state, params.update_hidden);
    Node *update_gate = add({update_input, update_hidden});
    update_gate = sigmoid(*update_gate);

    Node *reset_input = linear(input, params.reset_input);
    Node *reset_hidden = linear(last_state, params.reset_hidden);
    Node *reset_gate = add({reset_input, reset_hidden});
    reset_gate = sigmoid(*reset_gate);

    Node *candidate_input = linear(input, params.candidate_input);
    Node *updated_hidden = mul(*reset_gate, last_state);
    Node *candidate_hidden = linear(*updated_hidden, params.candidate_hidden);
    Node *candidate = add({candidate_input, candidate_hidden});
    candidate = tanh(*candidate);

    int hidden_dim = last_state.size();
    Graph &graph = dynamic_cast<Graph&>(input.getNodeContainer());
    Node *one = tensor(graph, hidden_dim, 1);
    Node *reversal_update = sub(*one, *update_gate);
    Node *passed_last_state = mul(*reversal_update, last_state);
    Node *updated_candidate = mul(*update_gate, *candidate);
    Node *h = add({passed_last_state, updated_candidate});
    return dropout(*h, dropout_value);
}

vector<Node *> gru(Node &initial_state, const vector<Node *> &inputs, GRUParams &params,
        dtype dropout_value) {
    Node *last_state = &initial_state;
    vector<Node *> results;

    for (Node *input : inputs) {
        last_state = gru(*last_state, *input, params, dropout_value);
        results.push_back(last_state);
    }

    return results;
}

}
