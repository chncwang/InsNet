#include "n3ldg-plus/block/gru.h"

#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/mul.h"
#include "n3ldg-plus/operator/add.h"
#include "n3ldg-plus/operator/bucket.h"
#include "n3ldg-plus/operator/sub.h"

using std::vector;
using std::string;

namespace n3ldg_plus {

GRUParam::GRUParam(const string &name) : update_input(name + "-update_input"),
    update_hidden(name + "-update_hidden"), reset_input(name + "reset_input"),
    reset_hidden(name + "reset_hidden"), candidate_input(name + "candidate_input"),
    candidate_hidden(name + "candidate_hidden") {}

void GRUParam::init(int out_size, int in_size) {
    update_input.init(out_size, in_size);
    update_hidden.init(out_size, out_size);
    reset_input.init(out_size, in_size);
    reset_hidden.init(out_size, out_size);
    candidate_input.init(out_size, in_size);
    candidate_hidden.init(out_size, out_size);
}

#if USE_GPU
vector<cuda::Transferable *> GRUParam::transferablePtrs() {
    return {&update_input, &update_hidden, &reset_input, &reset_hidden, &candidate_input,
        &candidate_hidden};
}
#endif

vector<Tunable<BaseParam> *> GRUParam::tunableComponents() {
    return {&update_input, &update_hidden, &reset_input, &reset_hidden, &candidate_input,
        &candidate_hidden};
}

void GRUBuilder::step(GRUParam &gru_params, Node &input, Node &h0, dtype dropout) {
    int len = hiddens_.size();
    Node *last_hidden = len == 0 ? &h0 : hiddens_.at(len - 1);
    using namespace n3ldg_plus;

    Node *update_input = linear(input, gru_params.update_input);
    Node *update_hidden = linear(*last_hidden, gru_params.update_hidden);
    Node *update_gate = add({update_input, update_hidden});
    update_gate = sigmoid(*update_gate);

    Node *reset_input = linear(input, gru_params.reset_input);
    Node *reset_hidden = linear(*last_hidden, gru_params.reset_hidden);
    Node *reset_gate = add({reset_input, reset_hidden});
    reset_gate = sigmoid(*reset_gate);

    Node *candidate_input = linear(input, gru_params.candidate_input);
    Node *updated_hidden = pointwiseMultiply(*reset_gate, *last_hidden);
    Node *candidate_hidden = linear(*updated_hidden, gru_params.candidate_hidden);
    Node *candidate = add({candidate_input, candidate_hidden});
    candidate = tanh(*candidate);

    int hidden_dim = h0.size();
    Graph &graph = dynamic_cast<Graph&>(input.getNodeContainer());
    Node *one = tensor(graph, hidden_dim, 1);
    Node *reversal_update = sub(*one, *update_gate);
    Node *passed_last_hidden = pointwiseMultiply(*reversal_update, *last_hidden);
    Node *updated_candidate = pointwiseMultiply(*update_gate, *candidate);
    Node *h = add({passed_last_hidden, updated_candidate});
    hiddens_.push_back(h);
}

}
