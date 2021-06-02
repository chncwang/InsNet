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
