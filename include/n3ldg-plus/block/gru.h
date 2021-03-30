#ifndef N3LDG_PLUS_GRU
#define N3LDG_PLUS_GRU

#include "n3ldg-plus/computation-graph/graph.h"
#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/mul.h"
#include "n3ldg-plus/operator/add.h"
#include "n3ldg-plus/operator/bucket.h"
#include "n3ldg-plus/operator/linear.h"
#include "n3ldg-plus/operator/sub.h"

#include <memory>

namespace n3ldg_plus {

struct GRUParam : public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    LinearParam update_input;
    LinearParam update_hidden;
    LinearParam reset_input;
    LinearParam reset_hidden;
    LinearParam candidate_input;
    LinearParam candidate_hidden;

    GRUParam(const std::string &name) : update_input(name + "-update_input"),
    update_hidden(name + "-update_hidden"), reset_input(name + "reset_input"),
    reset_hidden(name + "reset_hidden"), candidate_input(name + "candidate_input"),
    candidate_hidden(name + "candidate_hidden") {}

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(update_input, update_hidden, reset_input, reset_hidden, candidate_input,
                candidate_hidden);
    }

    void init(int out_size, int in_size) {
        update_input.init(out_size, in_size);
        update_hidden.init(out_size, out_size);
        reset_input.init(out_size, in_size);
        reset_hidden.init(out_size, out_size);
        candidate_input.init(out_size, in_size);
        candidate_hidden.init(out_size, out_size);
    }

    int inDim() {
        return update_input.W().outDim();
    }

    int outDim() {
        return update_input.W().inDim();
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override {
        return {&update_input, &update_hidden, &reset_input, &reset_hidden, &candidate_input,
            &candidate_hidden};
    }

    virtual std::string name() const {
        return "GRUParams";
    }
#endif

protected:
    std::vector<Tunable<BaseParam> *> tunableComponents() override {
        return {&update_input, &update_hidden, &reset_input, &reset_hidden, &candidate_input,
            &candidate_hidden};
    }
};

class DynamicGRUBuilder {
public:
    std::vector<Node*> hiddens;

    int size() {
        return hiddens.size();
    }

    void connect(Graph &graph, GRUParam &gru_params, Node &input, Node &h0, dtype dropout,
            bool is_training) {
        int len = hiddens.size();
        Node *last_hidden = len == 0 ? &h0 : hiddens.at(len - 1);
        using namespace n3ldg_plus;

        Node *update_input = linear(graph, input, gru_params.update_input);
        Node *update_hidden = linear(graph, *last_hidden, gru_params.update_hidden);
        Node *update_gate = add(graph, {update_input, update_hidden});
        update_gate = sigmoid(graph, *update_gate);

        Node *reset_input = linear(graph, input, gru_params.reset_input);
        Node *reset_hidden = linear(graph, *last_hidden, gru_params.reset_hidden);
        Node *reset_gate = add(graph, {reset_input, reset_hidden});
        reset_gate = sigmoid(graph, *reset_gate);

        Node *candidate_input = linear(graph, input, gru_params.candidate_input);
        Node *updated_hidden = pointwiseMultiply(graph, *reset_gate, *last_hidden);
        Node *candidate_hidden = linear(graph, *updated_hidden, gru_params.candidate_hidden);
        Node *candidate = add(graph, {candidate_input, candidate_hidden});
        candidate = tanh(graph, *candidate);

        int hidden_dim = h0.getDim();
        Node *one = bucket(graph, hidden_dim, 1);
        Node *reversal_update = sub(graph, *one, *update_gate);
        Node *passed_last_hidden = pointwiseMultiply(graph, *reversal_update, *last_hidden);
        Node *updated_candidate = pointwiseMultiply(graph, *update_gate, *candidate);
        Node *h = add(graph, {passed_last_hidden, updated_candidate});
        hiddens.push_back(h);
    }

private:
};

}

#endif
